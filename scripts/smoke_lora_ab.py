import argparse
import json
import sys
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
sys.path.append(str(SRC))

import infer
import generate
import synthetic

def _read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(
        description="A/B smoke: base SDXL vs SDXL+LoRA on same seed/layout/palette."
    )
    ap.add_argument("--lora_dir", required=True,
                    help="Folder containing LoRA weights (e.g., outputs/lora/runs/exp01/final_lora)")
    ap.add_argument("--out_dir", default="outputs/lora_ab")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=5.5)
    ap.add_argument("--control_mode", choices=["element", "safe", "edge", "none"], default="safe")
    ap.add_argument("--layout_json", default=None,
                    help="Optional path to layout JSON; if omitted, use synthetic sampler")
    ap.add_argument("--palette_json", default=None,
                    help="Optional path to palette JSON; if omitted, use synthetic sampler")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    _ensure_outdir(out_dir)

    if args.layout_json:
        layout = _read_json(Path(args.layout_json))
        if args.palette_json:
            palette = _read_json(Path(args.palette_json))
        else:
            palette = synthetic.sample_condition_batch(
                1, canvas_size=(args.height, args.width), seed=args.seed
            )[0]["palette"]
        control_map, safe_zone = generate.create_layout_control_map(layout)
    else:
        sample = synthetic.sample_condition_batch(
            1, canvas_size=(args.height, args.width), seed=args.seed
        )[0]
        layout = sample["layout"]
        palette = sample["palette"]
        control_map = sample.get("control_map", None)
        safe_zone = sample.get("safe_zone", None)
        if control_map is None or safe_zone is None:
            control_map, safe_zone = generate.create_layout_control_map(layout)

    use_controlnet = (args.control_mode != "none")
    control_from = None if not use_controlnet else args.control_mode

    common = dict(
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        layout=layout,
        palette=palette,
        use_controlnet=use_controlnet,
        control_from=control_from,
        control_map=control_map,
        safe_zone=safe_zone,
        save_dir=str(out_dir),
    )

    base_img_path = infer.generate_and_mask(lora_path=None, **common)
    base_img_path = Path(base_img_path)
    base_copy = out_dir / f"ab_seed{args.seed}_base.png"
    copyfile(base_img_path, base_copy)
    print(f" Base image: {base_copy}")

    lora_dir = Path(args.lora_dir).resolve()
    if not lora_dir.exists():
        raise SystemExit(f"LoRA dir not found: {lora_dir}")
    lora_img_path = infer.generate_and_mask(lora_path=str(lora_dir), **common)
    lora_img_path = Path(lora_img_path)
    lora_copy = out_dir / f"ab_seed{args.seed}_lora.png"
    copyfile(lora_img_path, lora_copy)
    print(f" LoRA image: {lora_copy}")

    print("Done. Compare *_base.png vs *_lora.png in:", out_dir)

if __name__ == "__main__":
    main()