from pathlib import Path
from typing import Dict, Optional
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import numpy as np

try:
    from sdxl import load_sdxl_with_lora, prompt_from_palette, _set_scheduler as _sdxl_set_scheduler
except Exception:
    import sys
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from sdxl import load_sdxl_with_lora, prompt_from_palette
    try:
        from sdxl import _set_scheduler as _sdxl_set_scheduler
    except Exception:
        _sdxl_set_scheduler = None

try:
    from control import control_image_from_map
except Exception:
    control_image_from_map = None

try:
    from generate import apply_safe_zone_mask
except Exception:
    import sys
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from generate import apply_safe_zone_mask

# Delay heavy imports
try:
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
except Exception:
    ControlNetModel = None
    StableDiffusionXLControlNetPipeline = None

def save_np_mask(mask_t, path):
    m = mask_t.squeeze(0).detach().cpu().numpy()
    m8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(m8, mode="L").save(str(path))

def draw_layout_boxes(base_img: Image.Image, layout: Dict, color=(255, 0, 0), width=3):
    img = base_img.copy()
    d = ImageDraw.Draw(img)
    for el in layout.get("elements", []):
        x, y, w, h = el["bbox_xywh"]
        d.rectangle([x, y, x + w, y + h], outline=color, width=width)
    return img

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
    use_controlnet: bool = False,
    controlnet_model_id: Optional[str] = None,
    control_strength: float = 0.8,
    control_from: str = "element",  # "element" | "safe" | "edge"
    scheduler: Optional[str] = None,
    debug: bool = True,
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

    stem = Path(out_name).stem
    img: Image.Image
    if use_controlnet:
        if StableDiffusionXLControlNetPipeline is None or ControlNetModel is None:
            raise ImportError("diffusers ControlNet classes not available.")
        if control_image_from_map is None:
            raise RuntimeError("control.control_image_from_map not importable.")
        if control_map is None:
            raise ValueError("use_controlnet=True requires a control_map tensor [4,H,W].")
        if controlnet_model_id is None:
            controlnet_model_id = "diffusers/controlnet-canny-sdxl-1.0"

        print(f"Using ControlNet: True ({controlnet_model_id}), strength={control_strength}")

        control_image = control_image_from_map(control_map=control_map, safe_zone=safe_zone, size=(int(width), int(height)), mode=str(control_from))
        controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=torch.float16
        )
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe = pipe.to("cuda")

        if scheduler and _sdxl_set_scheduler is not None:
            try:
                _sdxl_set_scheduler(pipe, scheduler)
                print(f"Scheduler active (ControlNet): {pipe.scheduler.__class__.__name__}")
            except Exception as e:
                print(f"Failed to set scheduler on ControlNet pipeline: {e}")

        if lora_path:
            pipe.load_lora_weights(lora_path)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            controlnet_conditioning_scale=float(control_strength),
            image=control_image,
            generator=generator,
            width=int(width),
            height=int(height),
        )
        img = result.images[0]
    else:
        # Base SDXL only
        pipe = load_sdxl_with_lora(
            model_id=model_id, lora_path=lora_path, device=dev, dtype=None, cpu_offload=True, scheduler=scheduler,
        )
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

    if debug:
        img.save(str(out_dir / f"{stem}_raw.png"))

    masked = apply_safe_zone_mask(img, safe_zone, neutral_color=tuple(int(x) for x in neutral_rgb))
    masked_path = out_dir / f"{stem}_masked.png"
    masked.save(str(masked_path))

    if debug:
        save_np_mask(safe_zone, out_dir / f"{stem}_safe_zone.png")
        overlay = draw_layout_boxes(masked, layout, color=(255, 0, 0), width=3)
        overlay.save(str(out_dir / f"{stem}_overlay.png"))

        cov = float((safe_zone > 0).float().mean().item())
        print(f"safe_zone coverage (1=safe): {cov:.3f}")
        print(f"reserved element coverage:   {1.0 - cov:.3f}")

    return str(masked_path)

def upscale_image(img: Image.Image, target_wh: tuple[int, int]) -> Image.Image:
    w, h = map(int, target_wh)
    if (img.width, img.height) == (w, h):
        return img
    return img.resize((w, h), resample=Image.LANCZOS)

def _edge_ring_mask(layout: Dict, canvas_wh: tuple[int, int], pad_px: int = 3, ring_thickness: int = 4) -> Image.Image:
    W, H = canvas_wh
    ring = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(ring)

    for el in layout.get("elements", []):
        x, y, w, h = map(int, el["bbox_xywh"])
        outer = [x - pad_px, y - pad_px, x + w + pad_px, y + h + pad_px]
        inner = [x + pad_px, y + pad_px, x + w - pad_px, y + h - pad_px]
        draw.rectangle(outer, fill=255)
        draw.rectangle(inner, fill=0)

        if ring_thickness > 0:
            f = ring.filter(ImageFilter.MaxFilter(size=max(3, 2 * ring_thickness + 1)))
            ring = ImageChops.lighter(ring, f)

    ring = ring.filter(ImageFilter.GaussianBlur(radius=2))
    return ring

def inpaint_neutral_edges(
    img: Image.Image,
    layout: Dict,
    mode: str = "blur",
    pad_px: int = 3,
    ring_thickness: int = 4,
    neutral_rgb: tuple[int, int, int] = (245, 245, 245),
    blur_radius: int = 6,
) -> Image.Image:
    ring = _edge_ring_mask(layout, (img.width, img.height), pad_px=pad_px, ring_thickness=ring_thickness)

    if mode == "blur":
        blurred = img.filter(ImageFilter.GaussianBlur(radius=int(blur_radius)))
        out = img.copy()
        out.paste(blurred, mask=ring)
        return out

    if mode == "neutral":
        fill = Image.new("RGB", img.size, tuple(int(c) for c in neutral_rgb))
        out = img.copy()
        out.paste(fill, mask=ring)
        return out

    raise ValueError("inpaint_neutral_edges: mode must be 'blur' or 'neutral'")