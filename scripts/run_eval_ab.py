import argparse
import csv
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import synthetic
import infer
from evaluate import wcag_pass_rate, layout_uniformity_score


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed_start", type=int, default=777)
    ap.add_argument("--text_size", type=str, default="normal", choices=["small", "normal", "large"])
    ap.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--controlnet_id", type=str, default="diffusers/controlnet-canny-sdxl-1.0")
    ap.add_argument("--control_mode", type=str, default="safe", choices=["safe", "element", "none"])
    ap.add_argument("--control_strength", type=float, default=0.8)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    base_dir = out_dir / "base"
    lora_dir_out = out_dir / "lora"
    _ensure_dir(out_dir)
    _ensure_dir(base_dir)
    _ensure_dir(lora_dir_out)

    lora_weights_dir = Path(args.lora_dir).resolve()
    if not lora_weights_dir.exists():
        raise SystemExit(f"LoRA dir not found: {lora_weights_dir}")

    rows = []

    for i in range(args.num_samples):
        seed = args.seed_start + i
        print(f"\n=== Sample {i+1}/{args.num_samples} (seed={seed}) ===", flush=True)

        sample = synthetic.sample_condition_batch(
            1,
            canvas_size=(args.height, args.width),
            seed=seed,
        )[0]
        layout = sample["layout"]
        palette = sample["palette"]
        control_map = sample.get("control_map")
        safe_zone = sample.get("safe_zone")

        common_gen_kwargs = dict(
            layout=layout,
            palette=palette,
            safe_zone=safe_zone,
            control_map=control_map,
            model_id=args.model_id,
            steps=args.steps,
            guidance=args.guidance,
            width=args.width,
            height=args.height,
            seed=seed,
            negative_prompt=(
                "illegible text, cluttered layout, high-frequency noise, "
                "heavy gradients, harsh contrast bands"
            ),
            control_mode=args.control_mode,
            control_strength=args.control_strength,
            controlnet_id=args.controlnet_id,
        )

        base_name = f"seed{seed}_base.png"
        base_path_str = infer.generate_and_mask(
            lora_path=None,
            out_dir=base_dir,
            out_name=base_name,
            **common_gen_kwargs,
        )
        base_path = Path(base_path_str)

        lora_name = f"seed{seed}_lora.png"
        lora_path_str = infer.generate_and_mask(
            lora_path=str(lora_weights_dir),
            out_dir=lora_dir_out,
            out_name=lora_name,
            **common_gen_kwargs,
        )
        lora_img_path = Path(lora_path_str)

        def eval_one(path: Path, variant: str) -> dict:
            img = Image.open(path).convert("RGB")

            wcag_pass, wcag_details = wcag_pass_rate(
                img,
                text_size=args.text_size,
                return_details=True,
            )

            layout_metrics = layout_uniformity_score(img, layout, safe_zone=safe_zone)

            row = {
                "seed": seed,
                "variant": variant,
                "image_path": str(path.relative_to(out_dir)),
                "wcag_pass_rate": float(wcag_pass),
                "wcag_passes": wcag_details.get("passes", 0),
                "wcag_total": wcag_details.get("total", 0),
                "mean_contrast": wcag_details.get("mean_contrast", float("nan")),
                "min_contrast": wcag_details.get("min_contrast", float("nan")),
                "safe_std": layout_metrics.get("safe_std", float("nan")),
                "bg_std": layout_metrics.get("bg_std", float("nan")),
                "uniformity": layout_metrics.get("uniformity", float("nan")),
            }
            return row

        rows.append(eval_one(base_path, "base"))
        rows.append(eval_one(lora_img_path, "lora"))

    csv_path = out_dir / "ab_metrics.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    print(f"\nSaved A/B metrics to: {csv_path}")
    print(f"Images under: {base_dir} and {lora_dir_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
