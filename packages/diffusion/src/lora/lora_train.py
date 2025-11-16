import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import random
from typing import List, Optional

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import numpy as np

    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import logging as dlogging
    from peft import LoraConfig as PeftLoraConfig
except Exception as e:
    torch = None
    F = None
    Dataset = None
    DataLoader = None
    Image = None
    np = None
    StableDiffusionXLPipeline = None
    dlogging = None
    PeftLoraConfig = None

from .lora_data import build_manifest

@dataclass
class LoraTrainConfig:
    model_id: str
    output_dir: str
    train_jsonl: str
    resolution: int = 1024
    rank: int = 8
    lr: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_train_steps: int = 0  # 0 == dry run
    checkpoint_steps: int = 50
    mixed_precision: str = "fp16"
    use_8bit_adam: bool = True
    seed: int = 42

def _write_config(cfg: LoraTrainConfig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(cfg), indent=2))

def _assert_reqs():
    if torch is None or StableDiffusionXLPipeline is None or PeftLoraConfig is None:
        raise RuntimeError("Required packages not available. Please install: torch, diffusers, transformers, peft, accelerate, safetensors,")

class JsonlImageDataset(Dataset):
    def __init__(self, jsonl_path: Path, resolution: int = 1024):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        self.resolution = int(resolution)

    def __len__(self):
        return len(self.items)

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        path = Path(rec["image"])
        caption = rec.get("caption", "")
        with Image.open(path) as im:
            tensor = self._preprocess(im)
        return {"pixel_values": tensor, "caption": caption}

def _inject_unet_lora(pipe: "StableDiffusionXLPipeline", rank: int = 8) -> int:
    pipe.unet.requires_grad_(False)
    unet_lora_cfg = PeftLoraConfig(
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]
    )
    pipe.unet.add_adapter(unet_lora_cfg)
    return sum(p.requires_grad for p in pipe.unet.parameters())

def _collect_lora_params(pipe: "StableDiffusionXLPipeline"):
    return [p for p in pipe.unet.parameters() if p.requires_grad]

def _encode_prompts(pipe: "StableDiffusionXLPipeline", captions: List[str], device: str):
    with torch.no_grad():
        out = pipe.encode_prompt(
            prompt=captions,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    print("DEBUG encode_prompt type:", type(out))
    if isinstance(out, tuple):
        for i, elem in enumerate(out):
            if isinstance(elem, torch.Tensor):
                print(f"  elem[{i}] shape={elem.shape}, ndim={elem.ndim}, dtype={elem.dtype}")
            else:
                print(f"  elem[{i}] type={type(elem)}")
    else:
        print("  tensor shape", out.shape, "ndim", out.ndim, "dtype", out.dtype)

    prompt_embeds = None
    pooled = None
    if isinstance(out, torch.Tensor):
        prompt_embeds = out

    elif isinstance(out, tuple):
        for elem in out:
            if not isinstance(elem, torch.Tensor):
                continue
            if elem.ndim == 3 and prompt_embeds is None:
                prompt_embeds = elem
            elif elem.ndim == 2 and pooled is None:
                pooled = elem

    if prompt_embeds is None:
        raise RuntimeError("encode_prompt did not return a prompt_embeds tensor")

    if pooled is None:
        with torch.no_grad():
            tok2 = pipe.tokenizer_2(
                captions,
                padding="max_length",
                max_length=pipe.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            tok2 = {k: v.to(device) for k, v in tok2.items()}

            te2_out = pipe.text_encoder_2(
                **tok2,
                output_hidden_states=True,
                return_dict=True,
            )

            if hasattr(te2_out, "pooler_output") and te2_out.pooler_output is not None:
                pooled = te2_out.pooler_output  # [B, D2]
            else:
                last_hidden = getattr(te2_out, "last_hidden_state", None)
                if last_hidden is None:
                    raise RuntimeError(
                        "text_encoder_2 returned neither pooler_output nor last_hidden_state"
                    )
                cls = last_hidden[:, 0, :]
                proj = getattr(pipe.text_encoder_2, "text_projection", None)
                if proj is not None:
                    pooled = proj(cls)
                else:
                    pooled = cls

    if pooled.ndim == 3 and pooled.shape[1] == 1:
        pooled = pooled.squeeze(1)
    elif pooled.ndim == 3 and pooled.shape[1] > 1:
        pooled = pooled.mean(dim=1)

    if pooled.ndim != 2:
        raise RuntimeError(f"Expected pooled_embeds [B, D2]; got {pooled.shape}")

    return prompt_embeds, pooled

def _sdxl_time_ids(pipe: "StableDiffusionXLPipeline", bsz: int, height: int, width: int, device: str, text_encoder_projection_dim: int):
    add_time_ids = pipe._get_add_time_ids(
        (height, width),
        (0, 0),
        (height, width),
        dtype=torch.long,
        text_encoder_projection_dim=int(text_encoder_projection_dim),
    )
    time_ids = torch.as_tensor(add_time_ids, device=device, dtype=torch.long)
    return time_ids.repeat(bsz, 1)

def _vae_encode(pipe: "StableDiffusionXLPipeline", imgs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        imgs = imgs.to(pipe.device, dtype=pipe.vae.dtype)
        posterior = pipe.vae.encode(imgs).latent_dist
        latents = posterior.sample() * pipe.vae.config.scaling_factor
    return latents

def main(argv=None):
    ap = argparse.ArgumentParser(description="LoRA Train (minimal) for SDXL UNet.")
    ap.add_argument("--images_dir", required=True, help="Folder with training images")
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--output_dir", default="outputs/lora/runs/exp01")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--max_train_steps", type=int, default=50)
    ap.add_argument("--checkpoint_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_jsonl", default=None, help="If absent, create in <output_dir>/manifests/captions.jsonl from --images_dir.")
    ap.add_argument("--fallback_caption", default=None)
    args = ap.parse_args(argv)
    print("Entered main(), parsed args:", args, flush=True)

    _assert_reqs()
    if dlogging:
        dlogging.set_verbosity_error()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    print("Requirements OK, seeds set.", flush=True)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = Path(args.train_jsonl) if args.train_jsonl else (manifests_dir / "captions.jsonl")
    if not args.train_jsonl:
        images_dir = Path(args.images_dir).resolve()
        build_manifest(images_dir, train_jsonl, fallback_caption=args.fallback_caption)
        print(f"Building manifest from images_dir={images_dir}", flush=True)

    cfg = LoraTrainConfig(
        model_id=args.model_id,
        output_dir=str(output_dir),
        train_jsonl=str(train_jsonl),
        resolution=int(args.resolution),
        rank=int(args.rank),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        max_train_steps=int(args.max_train_steps),
        checkpoint_steps=int(args.checkpoint_steps),
        seed=int(args.seed),
    )
    _write_config(cfg, output_dir / "lora_config.json")
    print("Training LoRA with config:", cfg, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device={device}, dtype={dtype}", flush=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(cfg.model_id, torch_dtype=dtype)
    pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    print("Loaded model.", flush=True)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    pipe.unet.enable_gradient_checkpointing()
    torch.backends.cuda.matmul.allow_tf32 = True
    pipe.unet.train()
    pipe.text_encoder.train(False)
    pipe.text_encoder_2.train(False)
    pipe.vae.train(False)

    injected = _inject_unet_lora(pipe, rank=cfg.rank)
    print(f"Injected LoRA adapters; trainable_param_flags={injected}")

    lora_params = _collect_lora_params(pipe)
    opt = torch.optim.AdamW(
        lora_params,
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.0,
    )
    ds = JsonlImageDataset(Path(cfg.train_jsonl), resolution=cfg.resolution)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=(device == "cuda"))
    print(f"DataLoader created with {len(ds)} items.", flush=True)

    scheduler = pipe.scheduler
    global_step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)

    while global_step < cfg.max_train_steps:
        for batch in dl:
            print(f"Global step {global_step}, accumulation {accum}", flush=True)
            pixels = batch["pixel_values"].to(device, dtype=dtype)
            captions = batch["caption"]

            latents = _vae_encode(pipe, pixels)
            noise_offset = 0.1
            noise = torch.randn_like(latents)
            if noise_offset > 0:
                noise = noise + noise_offset * torch.randn_like(noise)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            timestep_dtype = next(pipe.unet.parameters()).dtype
            t = timesteps.to(device=device, dtype=timestep_dtype)

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            print("encoding prompts...")
            prompt_embeds, pooled_embeds = _encode_prompts(pipe, captions, device=device)
            print("prompt_embeds:", prompt_embeds.shape, prompt_embeds.dtype)

            proj_dim_target = getattr(pipe.text_encoder_2.config, "projection_dim", None)
            if proj_dim_target is None:
                proj = getattr(pipe.text_encoder_2, "text_projection", None)
                if proj is not None and hasattr(proj, "weight"):
                    proj_dim_target = proj.weight.shape[1]
                else:
                    proj_dim_target = 1280
            pipe.text_encoder_2.config.projection_dim = int(proj_dim_target)

            if pooled_embeds.ndim == 3 and pooled_embeds.shape[1] == 1:
                pooled_embeds = pooled_embeds.squeeze(1)
            if pooled_embeds.shape[-1] != proj_dim_target:
                pooled_embeds = pooled_embeds[..., :proj_dim_target]

            unet_dtype = next(pipe.unet.parameters()).dtype
            prompt_embeds = prompt_embeds.to(device=device, dtype=unet_dtype)
            pooled_embeds = pooled_embeds.to(device=device, dtype=unet_dtype)

            time_ids = _sdxl_time_ids(
                pipe,
                bsz=bsz,
                height=pixels.shape[-2],
                width=pixels.shape[-1],
                device=device,
                text_encoder_projection_dim=proj_dim_target,
            )
            assert prompt_embeds.ndim == 3, f"prompt_embeds ndim={prompt_embeds.ndim}, shape={prompt_embeds.shape}"
            assert pooled_embeds.ndim == 2, f"pooled_embeds ndim={pooled_embeds.ndim}, shape={pooled_embeds.shape}"
            assert time_ids.dtype == torch.long, f"time_ids dtype={time_ids.dtype} must be torch.long"
            assert time_ids.shape[0] == bsz, f"time_ids batch mismatch: {time_ids.shape[0]} vs {bsz}"

            try:
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=unet_dtype):
                        model_pred = pipe.unet(
                            noisy_latents.to(unet_dtype),
                            t,
                            prompt_embeds,
                            added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids},
                        ).sample
                else:
                    model_pred = pipe.unet(
                        noisy_latents.to(unet_dtype),
                        t,
                        prompt_embeds,
                        added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids},
                    ).sample
            except Exception as e:
                print("DEBUG: Fail context:")
                print(f"  latents: {latents.shape} {latents.dtype} on {latents.device}")
                print(f"  noisy_latents: {noisy_latents.shape} {noisy_latents.dtype} on {noisy_latents.device}")
                print(f"  timesteps: {timesteps.shape} {timesteps.dtype} on {timesteps.device}")
                print(f"  prompt_embeds: {prompt_embeds.shape} {prompt_embeds.dtype} on {prompt_embeds.device}")
                print(f"  pooled_embeds: {pooled_embeds.shape} {pooled_embeds.dtype} on {pooled_embeds.device}")
                print(f"  time_ids: {time_ids.shape} {time_ids.dtype} on {time_ids.device}")
                proj_dim = getattr(pipe.text_encoder_2.config, 'projection_dim', None)
                print(f"  projection_dim: {proj_dim}")
                raise

            pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
            target = scheduler.get_velocity(latents, noise, timesteps) if (pred_type == "v_prediction" and hasattr(scheduler, "get_velocity")) else noise

            snr_gamma = 5.0
            with torch.no_grad():
                alphas_cumprod = scheduler.alphas_cumprod.to(device)[timesteps.long().cpu()].to(device)
            snr = alphas_cumprod / (1 - alphas_cumprod)
            weight = (snr_gamma / (snr + 1)).view(-1, *([1] * (model_pred.ndim - 1))).to(model_pred.dtype)

            loss = (weight * (model_pred.float() - target.float())**2).mean()
            if not torch.isfinite(loss):
                print(f"WARNING: non-finite loss detected ({loss.item()}); skipping step.")
                opt.zero_grad(set_to_none=True)
                continue
            (loss / cfg.gradient_accumulation_steps).backward()
            accum += 1

            if accum % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    print(f"step {global_step}/{cfg.max_train_steps} loss {loss.item():.4f}")

                if global_step % 20 == 0:
                    torch.cuda.empty_cache()

                if global_step % max(1, cfg.checkpoint_steps) == 0:
                    ckpt_dir = output_dir / f"ckpt_step_{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    pipe.save_lora_weights(ckpt_dir, unet_lora_layers=pipe.unet, weight_name="pytorch_lora_weights.safetensors")
                    print(f"Saved LoRA checkpoint to {ckpt_dir}")

                if global_step >= cfg.max_train_steps:
                    break

    final_dir = output_dir / "final_lora"
    final_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_lora_weights(final_dir, unet_lora_layers=pipe.unet, weight_name="pytorch_lora_weights.safetensors")
    print(f"Saved final LoRA weights to {final_dir}")
    print("Training complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())