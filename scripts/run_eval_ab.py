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
    ap = argparse.ArgumentParser(
        description=(
            "Generate a small batch of slides with base SDXL vs SDXL+LoRA, "
            "run WCAG + layout metrics, and save a CSV."
        )
    )
    ap.add_argument(
        "--lora_dir",
        required=True,
        help="Folder containing LoRA weights (e.g., outputs/lora/runs/exp01/final_lora)",
    )
    ap.add_argument(
        "--out_dir",
        default="outputs/lora_ab_eval",
        help="Directory to store generated images and CSV metrics.",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of synthetic layouts / seeds to evaluate.",
    )
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=5.5)
    ap.add_argument("--seed_start", type=int, default=777)
    ap.add_argument(
        "--text_size",
        choices=["normal", "large"],
        default="normal",
        help="WCAG text-size regime for contrast thresholds.",
    )
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
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            steps=args.steps,
            guidance=args.guidance,
            width=args.width,
            height=args.height,
            seed=seed,
            negative_prompt=(
                "illegible text, cluttered layout, high-frequency noise, "
                "heavy gradients, harsh contrast bands"
            ),
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

            wcag = wcag_pass_rate(
                img,
                text_size=args.text_size,
                return_details=True,
            )
            layout_metrics = layout_uniformity_score(img, layout, safe_zone=safe_zone)

            row = {
                "seed": seed,
                "variant": variant,
                "image_path": str(path.relative_to(out_dir)),
                "wcag_pass_rate": wcag.get("pass_rate", float("nan")),
                "wcag_passes": wcag.get("passes", 0),
                "wcag_total": wcag.get("total", 0),
                "mean_contrast": wcag.get("mean_contrast", float("nan")),
                "min_contrast": wcag.get("min_contrast", float("nan")),
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
