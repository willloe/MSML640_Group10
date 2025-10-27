import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
sys.path.append(str(SRC))

import torch
from sdxl import load_sdxl_with_lora, prepare_prompt, prompt_from_palette

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scheduler", type=str, default=None, help="try: ddim or dpmpp")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--guidance", type=float, default=5.5)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        pipe = load_sdxl_with_lora(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            lora_path=None,
            device=device,
            dtype=None,         # auto-selects fp16 on GPU, fp32 on CPU
            cpu_offload=True,
            scheduler=args.scheduler
        )
    except Exception as e:
        print("Failed to initialize SDXL pipeline:", e)
        return

    try:
        dev = "cuda" if pipe._execution_device.type == "cuda" else "cpu"
    except Exception:
        dev = device
    print("Device:", dev)
    try:
        print("Scheduler class:", pipe.scheduler.__class__.__name__)
    except Exception:
        pass

    palette = {"primary": "#1E40AF", "secondary": "#60A5FA", "accent": "#F59E0B", "bg_style": "minimalist gradient"}
    prompt = prompt_from_palette(palette)

    print("Device:", device)
    print("Prompt:", prompt)

    if device == "cpu":
        print("Loaded SDXL on CPU")
        return

    try:
        out = pipe(
            prompt=prompt,
            num_inference_steps=6,
            guidance_scale=5.5,
            width=1024,
            height=1024,
        )
        img = out.images[0]
        out_dir = ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "sdxl_smoke.png"
        img.save(str(path))
        print("Saved:", path)
    except Exception as e:
        print("Pipeline call failed:", e)


if __name__ == "__main__":
    main()
