import os
import torch

try:
    from diffusers import DiffusionPipeline
except Exception as e:
    DiffusionPipeline = None
    _DIFFUSERS_IMPORT_ERR = e
else:
    _DIFFUSERS_IMPORT_ERR = None

try:
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
except Exception:
    DDIMScheduler = None
    DPMSolverMultistepScheduler = None

PROMPT_PRESETS = {
    "default": (
        "single minimal professional presentation slide background, "
        "soft gradients, large clean content area, low contrast, "
        "no text, no logos, no collage, no grid"
    ),
    "academic": (
        "clean academic presentation slide background, white or light canvas, "
        "subtle sectioning for title and bullet points, low contrast, "
        "no photos, no heavy textures, no collage"
    ),
    "noisy": (
        "highly textured abstract geometric background, layered shapes and patterns, "
        "vibrant but not neon colors, noticeable noise and detail, "
        "complex visual structure, no text or logos"
    ),
    "gradient": (
        "smooth multi-tone gradient presentation background, soft transitions, "
        "no hard shapes, no grid, no collage, very low texture, "
        "large empty regions for overlaying text"
    ),
    "photo": (
        "photographic background with shallow depth of field, "
        "soft bokeh and light leaks, muted colors, "
        "subject off-center, large negative space"
    ),
}

SLIDESAFE_TOKEN = "slidesafe"

def add_slidesafe_token(prompt: str) -> str:
    """
    Prefix the special 'slidesafe' token if it's not already present.
    This lets LoRA learn a behavior keyed to this token.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return SLIDESAFE_TOKEN
    if SLIDESAFE_TOKEN in prompt:
        return prompt
    return f"{SLIDESAFE_TOKEN} {prompt}"

def _pick_device(device):
    if device is not None:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pick_dtype(dtype, device):
    if dtype is not None:
        return dtype
    return torch.float16 if device == "cuda" else torch.float32

def _set_scheduler(pipe, name: str):
    if not name or not hasattr(pipe, "scheduler"):
        return

    cfg = getattr(pipe.scheduler, "config", None)
    if cfg is None:
        print("set_scheduler: no scheduler config found on pipeline; skipping")
        return

    key = str(name).strip().lower()
    try:
        if key in ("ddim", "ddimscheduler") and DDIMScheduler is not None:
            pipe.scheduler = DDIMScheduler.from_config(cfg)
            return
        if key in ("dpmpp", "dpm++", "dpmsolver", "dpmsolvermultistep") and DPMSolverMultistepScheduler is not None:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfg)
            return
        print(f"set_scheduler: unknown or unavailable scheduler '{name}', keeping default")
    except Exception as e:
        print(f"set_scheduler: failed to set '{name}': {e}")

def load_sdxl_with_lora(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    lora_path=None,
    device=None,
    dtype=None,
    cpu_offload=True,
    scheduler: str | None = None,
):
    if DiffusionPipeline is None:
        raise ImportError(
            f"Could not import diffusers. Original error: {_DIFFUSERS_IMPORT_ERR}"
        )

    dev = _pick_device(device)
    dt = _pick_dtype(dtype, dev)

    variant = "fp16" if (dev == "cuda" and dt == torch.float16) else None

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dt,
            variant=variant,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load SDXL base model '{model_id}': {e}")

    if dev == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    try:
        pipe.to(dev)
    except Exception:
        dev = "cpu"
        pipe.to("cpu")

    if scheduler:
        _set_scheduler(pipe, scheduler)
        try:
            print(f"Scheduler active: {pipe.scheduler.__class__.__name__}")
        except Exception:
            pass

    if cpu_offload and dev == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    if lora_path:
        if not os.path.exists(lora_path):
            raise RuntimeError(f"LoRA path does not exist: {lora_path}")
        try:
            pipe.load_lora_weights(lora_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA from '{lora_path}': {e}")

    return pipe


def prepare_prompt(prompt, palette=None):
    if not palette:
        return prompt

    parts = [prompt]
    primary = str(palette.get("primary", "")).strip()
    secondary = str(palette.get("secondary", "")).strip()
    accent = str(palette.get("accent", "")).strip()
    style = str(palette.get("bg_style", "")).strip()

    if primary:
        parts.append(f"primary color {primary}")
    if secondary:
        parts.append(f"secondary color {secondary}")
    if accent:
        parts.append(f"accent color {accent}")
    if style:
        parts.append(style)

    return ", ".join([p for p in parts if p])

def prompt_from_palette(palette, preset = "academic"):
    base = PROMPT_PRESETS.get(preset, PROMPT_PRESETS["academic"])
    primary = palette.get("primary", "#75E4E4")
    secondary = palette.get("secondary", "#BC8AAC")
    accent = palette.get("accent", "#14E4C7")
    color_str = (
        f"primary color {primary}, secondary color {secondary}, accent color {accent}, "
        "subtle texture"
    )
    raw = f"{base}, {color_str}"
    return add_slidesafe_token(raw)