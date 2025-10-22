import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
sys.path.append(str(SRC))

from synthetic import sample_condition_batch
from infer import generate_and_mask


def main():
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
        steps=10,
        guidance=5.5,
        width=1024,
        height=768,
        seed=777,
        negative_prompt="busy patterns, high-frequency noise",
        out_dir=ROOT / "outputs",
        out_name="smoke_infer.png",
        use_controlnet=False,
        controlnet_model_id=None,
        control_strength=0.8,
        debug=True,
    )
    print("Saved (or would save) to:", out_path)


if __name__ == "__main__":
    main()
