import os
import torch

# Delay heavy imports to keep import cost low in other modules.
try:
    from diffusers import DiffusionPipeline
except Exception as e:
    DiffusionPipeline = None
    _DIFFUSERS_IMPORT_ERR = e
else:
    _DIFFUSERS_IMPORT_ERR = None


def _pick_device(device):
    if device is not None:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pick_dtype(dtype, device):
    if dtype is not None:
        return dtype
    return torch.float16 if device == "cuda" else torch.float32


def load_sdxl_with_lora(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    lora_path=None,
    device=None,
    dtype=None,
    cpu_offload=True,
):
    if DiffusionPipeline is None:
        raise ImportError(
            f"Could not import diffusers. Original error: {_DIFFUSERS_IMPORT_ERR}"
        )

    dev = _pick_device(device)
    dt = _pick_dtype(dtype, dev)

    # If running on CUDA with fp16, request fp16 weights variant.
    variant = "fp16" if (dev == "cuda" and dt == torch.float16) else None

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dt,
            variant=variant,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load SDXL base model '{model_id}': {e}")

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # Move to device
    try:
        pipe.to(dev)
    except Exception:
        # Fallback to CPU if GPU move fails
        dev = "cpu"
        pipe.to("cpu")

    # Optional offload when using CUDA
    if cpu_offload and dev == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    # Optional LoRA
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
    style = str(palette.get("bg_style", "")).strip()

    if primary:
        parts.append(f"primary color {primary}")
    if secondary:
        parts.append(f"secondary accent {secondary}")
    if style:
        parts.append(style)

    return ", ".join([p for p in parts if p])