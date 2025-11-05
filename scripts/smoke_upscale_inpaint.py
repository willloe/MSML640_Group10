import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
sys.path.append(str(SRC))

import torch
from synthetic import sample_condition_batch
from infer import generate_and_mask, inpaint_neutral_edges, upscale_image
from PIL import Image

def main():
    s = sample_condition_batch(n=1, canvas_size=(768, 1024), seed=11)[0]
    out_path = generate_and_mask(
        palette=s["palette"],
        layout=s["layout"],
        safe_zone=s["safe_zone"],
        control_map=s["control_map"],
        steps=10,
        guidance=5.5,
        width=1024,
        height=768,
        seed=2025,
        negative_prompt="busy patterns, high-frequency noise",
        out_dir=ROOT / "outputs",
        out_name="smoke_upscale_inpaint_base.png",
        use_controlnet=False,
        debug=True,
    )

    img = Image.open(out_path).convert("RGB")
    tidied = inpaint_neutral_edges(img, s["layout"], mode="blur", pad_px=3, ring_thickness=4, blur_radius=6)
    up = upscale_image(tidied, target_wh=(1280, 720))

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tidied_path = out_dir / "smoke_edge_tidy.png"
    up_path = out_dir / "smoke_edge_tidy_1280x720.png"
    tidied.save(tidied_path)
    up.save(up_path)

    print("Saved:", tidied_path)
    print("Saved:", up_path)

if __name__ == "__main__":
    main()
