from pathlib import Path
from typing import Dict, Optional
import torch

try:
    from sdxl import load_sdxl_with_lora, prompt_from_palette
except Exception:
    import sys
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from sdxl import load_sdxl_with_lora, prompt_from_palette

try:
    from generate import apply_safe_zone_mask
except Exception:
    import sys
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from generate import apply_safe_zone_mask


def generate_and_mask(
    palette: Dict,
    layout: Dict,
    safe_zone,
    control_map=None,
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    lora_path: Optional[str] = None,
    device: Optional[str] = None,
    steps: int = 12,
    guidance: float = 5.5,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = 1234,
    negative_prompt: Optional[str] = None,
    out_dir: str | Path = "outputs",
    out_name: str = "gen_masked.png",
    neutral_rgb=(245, 245, 245),
) -> str:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompt = prompt_from_palette(palette)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    if dev == "cpu":
        print("generate_and_mask: running on CPU; skipping heavy generation.")
        print("Prompt:", prompt)
        print("Would save to:", out_path)
        return str(out_path)

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    # Load SDXL (with optional LoRA)
    pipe = load_sdxl_with_lora(
        model_id=model_id,
        lora_path=lora_path,
        device=dev,
        dtype=None,        # default: fp16 on CUDA, fp32 on CPU
        cpu_offload=True,
    )

    # Minimal generate call
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=float(guidance),
        num_inference_steps=int(steps),
        width=int(width),
        height=int(height),
        generator=generator,
    )
    img = result.images[0]

    # Apply safe-zone masking to enforce neutral element regions
    masked = apply_safe_zone_mask(img, safe_zone, neutral_color=tuple(int(x) for x in neutral_rgb))

    masked.save(str(out_path))
    return str(out_path)
