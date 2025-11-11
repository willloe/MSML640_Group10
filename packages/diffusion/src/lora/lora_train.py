from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

try:
    import torch
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import logging as dlogging
except Exception as e:
    torch = None
    StableDiffusionXLPipeline = None
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
    mixed_precision: str = "fp16"
    use_8bit_adam: bool = True
    seed: int = 42

def _write_config(cfg: LoraConfig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(cfg), indent=2))

def _assert_reqs():
    if torch is None or StableDiffusionXLPipeline is None:
        raise RuntimeError("Required packages not available. Please install: torch, diffusers, transformers, peft, accelerate, safetensors.")

def _inject_unet_lora(pipe: "StableDiffusionXLPipeline", rank: int = 8) -> int:
    try:
        attn_modules = []
        for name, module in pipe.unet.named_modules():
            if "attn" in name and hasattr(module, "to_q") and hasattr(module, "to_v"):
                attn_modules.append(module)
        return len(attn_modules)
    except Exception as e:
        raise RuntimeError(f"Failed to scan UNet attention modules: {e}")

def main(argv=None):
    ap = argparse.ArgumentParser(description="LoRA Train Scaffold for SDXL (dry-run by default).")
    ap.add_argument("--images_dir", required=True, help="Folder with training images")
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--output_dir", default="outputs/lora")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--max_train_steps", type=int, default=0, help="0 means dry-run only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_jsonl", default=None, help="Optional existing captions.jsonl; otherwise will be generated")
    ap.add_argument("--fallback_caption", default=None, help="Fallback caption if no sidecar caption found")
    args = ap.parse_args(argv)

    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = Path(args.train_jsonl) if args.train_jsonl else (manifests_dir / "captions.jsonl")
    if not args.train_jsonl:
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
        seed=int(args.seed),
    )
    _write_config(cfg, output_dir / "lora_config.json")

    _assert_reqs()
    if dlogging:
        dlogging.set_verbosity_error()

    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    pipe = StableDiffusionXLPipeline.from_pretrained(cfg.model_id)
    pipe.to(device)

    injected = _inject_unet_lora(pipe, rank=cfg.rank)

    print("LoRA Scaffold:")
    print(f"images_dir: {images_dir}")
    print(f"train_jsonl: {train_jsonl}")
    print(f"output_dir: {output_dir}")
    print(f"device: {device}")
    print(f"unet_attn_modules_detected: {injected}")
    print(f"max_train_steps: {cfg.max_train_steps} (0 means dry-run, no training)")

    if cfg.max_train_steps == 0:
        print("Dry-run complete. Next step: wire actual LoRA layers and optimizer.")
        return 0

    raise SystemExit("Training not implemented yet in scaffold step. Set max_train_steps=0 for dry-run.")

if __name__ == "__main__":
    raise SystemExit(main())
