import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from synthetic import sample_condition_batch
from infer import generate_and_mask


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Simple SDXL/LoRA smoke test on a synthetic slide layout."
    )
    ap.add_argument("--out_dir", default=str(ROOT / "outputs"),
                    help="Folder to save the generated image.")
    ap.add_argument("--out_name", default="smoke_infer.png",
                    help="Output filename (PNG).")
    ap.add_argument("--width", type=int, default=1024,
                    help="Output width in pixels (slide-like).")
    ap.add_argument("--height", type=int, default=576,
                    help="Output height in pixels (slide-like).")
    ap.add_argument("--steps", type=int, default=28,
                    help="Number of diffusion steps.")
    ap.add_argument("--guidance", type=float, default=4.0,
                    help="Classifier-free guidance scale.")
    ap.add_argument("--seed", type=int, default=777,
                    help="Random seed for layout + generation.")
    ap.add_argument(
        "--control_mode",
        choices=["none", "element", "safe", "edge"],
        default="none"
    )
    ap.add_argument(
        "--lora_dir",
        default=None,
        help="Optional LoRA directory (e.g. outputs/lora/runs/exp01/final_lora). "
             "If omitted or missing, run base SDXL only."
    )
    args = ap.parse_args(argv)

    samples = sample_condition_batch(
        n=1,
        canvas_size=(args.height, args.width),
        seed=args.seed,
    )
    s = samples[0]

    lora_path = None
    if args.lora_dir:
        ld = Path(args.lora_dir).resolve()
        if ld.exists():
            lora_path = str(ld)
            print(f"Using LoRA from: {ld}")
        else:
            print(f"WARNING: LoRA dir {ld} not found; using base model only.")

    use_controlnet = (args.control_mode != "none")
    control_from = None if not use_controlnet else args.control_mode

    out_path = generate_and_mask(
        palette=s["palette"],
        layout=s["layout"],
        safe_zone=s.get("safe_zone"),
        control_map=s.get("control_map"),
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        lora_path=lora_path,
        steps=args.steps,
        guidance=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed,
        out_dir=args.out_dir,
        out_name=args.out_name,
        use_controlnet=use_controlnet,
        control_from=control_from,
    )

    print("Saved image to:", out_path)


if __name__ == "__main__":
    main()
