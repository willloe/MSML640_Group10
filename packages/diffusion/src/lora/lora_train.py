import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import numpy as np

    from diffusers import StableDiffusionXLPipeline
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0
    from diffusers.utils import logging as dlogging
except Exception as e:
    # Defer hard failure to a clear check later
    torch = None
    F = None
    Dataset = None
    DataLoader = None
    Image = None
    np = None
    StableDiffusionXLPipeline = None
    LoRAAttnProcessor2_0 = None
    dlogging = None

from .lora_data import build_manifest

@dataclass
class LoraConfig:
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

def _write_config(cfg: LoraConfig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(cfg), indent=2))

def _assert_reqs():
    if torch is None or StableDiffusionXLPipeline is None or LoRAAttnProcessor2_0 is None:
        raise RuntimeError("Required packages not available. Please install: torch, diffusers>=0.24 (LoRAAttnProcessor2_0), transformers, peft, accelerate, safetensors, ")

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
    unet = pipe.unet
    block_out = list(unet.config.block_out_channels)
    cross_dim_cfg = getattr(unet.config, "cross_attention_dim", None)

    procs = {}
    for name in unet.attn_processors.keys():
        is_self_attn = name.endswith("attn1.processor")
        cross_dim = None if is_self_attn else cross_dim_cfg

        if name.startswith("mid_block"):
            hidden_size = block_out[-1]
        elif name.startswith("down_blocks"):
            try:
                i = int(name.split(".")[1])
            except Exception:
                i = 0
            hidden_size = block_out[i]
        elif name.startswith("up_blocks"):
            try:
                i = int(name.split(".")[1])
            except Exception:
                i = 0
            hidden_size = list(reversed(block_out))[i]
        else:
            hidden_size = block_out[-1]

        if cross_dim is None:
            procs[name] = LoRAAttnProcessor2_0(hidden_size, rank=rank)
        else:
            procs[name] = LoRAAttnProcessor2_0(hidden_size, cross_dim, rank=rank)

    unet.set_attn_processor(procs)
    return len(procs)

def _collect_lora_params(pipe: "StableDiffusionXLPipeline"):
    params = []
    for _, m in pipe.unet.attn_processors.items():
        params += [p for p in m.parameters() if p.requires_grad]
    return params

def _encode_prompts(pipe: "StableDiffusionXLPipeline", captions: List[str], device: str):
    enc = pipe.encode_prompt(
        prompt=captions,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )
    return enc[0], enc[1]

def _vae_encode(pipe: "StableDiffusionXLPipeline", imgs: torch.Tensor) -> torch.Tensor:
    imgs = imgs.to(pipe.device, dtype=pipe.vae.dtype)
    posterior = pipe.vae.encode(imgs).latent_dist
    latents = posterior.sample() * pipe.vae.config.scaling_factor
    return latents

def main(argv=None):
    ap = argparse.ArgumentParser(description="LoRA Train (minimal) for SDXL UNet.")
    ap.add_argument("--images_dir", required=True, help="Folder with training images")
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--output_dir", default="outputs/lora/runs/exp01")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--max_train_steps", type=int, default=50)
    ap.add_argument("--checkpoint_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_jsonl", default=None, help="If absent, create in <output_dir>/manifests/captions.jsonl from --images_dir.")
    ap.add_argument("--fallback_caption", default=None)
    args = ap.parse_args(argv)

    _assert_reqs()
    if dlogging:
        dlogging.set_verbosity_error()
    torch.manual_seed(int(args.seed))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = Path(args.train_jsonl) if args.train_jsonl else (manifests_dir / "captions.jsonl")
    if not args.train_jsonl:
        images_dir = Path(args.images_dir).resolve()
        build_manifest(images_dir, train_jsonl, fallback_caption=args.fallback_caption)

    cfg = LoraConfig(
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(cfg.model_id, torch_dtype=dtype)
    pipe.to(device)
    pipe.unet.train()
    pipe.text_encoder.train(False)
    pipe.text_encoder_2.train(False)
    pipe.vae.train(False)

    injected = _inject_unet_lora(pipe, rank=cfg.rank)
    print(f"Injected LoRA into {injected} UNet attention processors (rank={cfg.rank})")

    lora_params = _collect_lora_params(pipe)
    opt = torch.optim.AdamW(lora_params, lr=cfg.lr)
    ds = JsonlImageDataset(Path(cfg.train_jsonl), resolution=cfg.resolution)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)

    scheduler = pipe.scheduler
    global_step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)

    while global_step < cfg.max_train_steps:
        for batch in dl:
            pixels = batch["pixel_values"].to(device, dtype=dtype)
            captions = batch["caption"]

            latents = _vae_encode(pipe, pixels)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            prompt_embeds, pooled_embeds = _encode_prompts(pipe, captions, device=device)

            unet_dtype = next(pipe.unet.parameters()).dtype
            model_pred = pipe.unet(
                noisy_latents.to(unet_dtype),
                timesteps,
                prompt_embeds,
                added_cond_kwargs={"pooled_text_embeds": pooled_embeds},
            ).sample

            pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
            if pred_type == "v_prediction" and hasattr(scheduler, "get_velocity"):
                target = scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            (loss / cfg.gradient_accumulation_steps).backward()
            accum += 1

            if accum % cfg.gradient_accumulation_steps == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    print(f"step {global_step}/{cfg.max_train_steps} loss {loss.item():.4f}")

                if global_step % max(1, cfg.checkpoint_steps) == 0:
                    ckpt_dir = output_dir / f"ckpt_step_{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    pipe.unet.save_attn_procs(ckpt_dir)
                    print(f"Saved LoRA checkpoint to {ckpt_dir}")

                if global_step >= cfg.max_train_steps:
                    break

    final_dir = output_dir / "final_lora"
    final_dir.mkdir(parents=True, exist_ok=True)
    pipe.unet.save_attn_procs(final_dir)
    print(f"Saved final LoRA weights to {final_dir}")
    print("Training complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())