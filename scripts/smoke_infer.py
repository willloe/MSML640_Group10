import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
sys.path.append(str(SRC))

from synthetic import sample_condition_batch
from infer import generate_and_mask

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scheduler", type=str, default=None, help="ddim or dpmpp")
    p.add_argument("--use_controlnet", type=int, default=0, help="0 or 1")
    p.add_argument("--control_from", type=str, default="element", choices=["element", "safe", "edge"])
    p.add_argument("--controlnet_model_id", type=str, default=None)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--guidance", type=float, default=5.5)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--seed", type=int, default=777)
    return p.parse_args()

def main():
    args = parse_args()
    samples = sample_condition_batch(n=1, canvas_size=(768, 1024), seed=7)
    s = samples[0]

    # Call the end-to-end helper
    out_path = generate_and_mask(
        palette=s["palette"],
        layout=s["layout"],
        safe_zone=s["safe_zone"],
        control_map=s["control_map"],  # not used yet, placeholder for future ControlNet
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        lora_path=None,
        steps=args.steps,
        guidance=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed,
        negative_prompt="busy patterns, high-frequency noise",
        out_dir=ROOT / "outputs",
        out_name="smoke_infer.png",
        use_controlnet=bool(args.use_controlnet),
        controlnet_model_id=args.controlnet_model_id,
        control_strength=0.8,
        control_from=args.control_from,
        scheduler=args.scheduler,
        debug=True,
    )
    print("Saved (or would save) to:", out_path)


if __name__ == "__main__":
    main()
