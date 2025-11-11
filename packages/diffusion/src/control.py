from typing import Tuple
import numpy as np
from PIL import Image
from typing import Optional
import torch

def to_uint8(mask):
    return (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)

def erode_3x3_binary(mask01: np.ndarray):
    m = (mask01 > 0.5).astype(np.uint8)
    H, W = m.shape
    p = np.pad(m, ((1, 1), (1, 1)), mode="edge")
    neighbor = [
        p[0:H, 0:W], p[0:H, 1:W+1], p[0:H, 2:W+2],
        p[1:H+1, 0:W], p[1:H+1, 1:W+1], p[1:H+1, 2:W+2],
        p[2:H+2, 0:W], p[2:H+2, 1:W+1], p[2:H+2, 2:W+2],
    ]
    out = neighbor[0]
    for n in neighbor[1:]:
        out = np.minimum(out, n)
    return out.astype(np.float32)

def make_control_image(control_map: torch.Tensor, safe_zone: Optional[torch.Tensor], size: Tuple[int, int], mode: str = "element") -> Image.Image:
    return control_image_from_map(control_map, safe_zone, size, mode)

def control_image_from_map(control_map: torch.Tensor, safe_zone: Optional[torch.Tensor], size: Tuple[int, int],  mode: str = "element") -> Image.Image:
    if not (isinstance(control_map, torch.Tensor) and control_map.ndim == 3 and control_map.shape[0] >= 1):
        raise ValueError("control_map must be a torch.Tensor of shape [C,H,W] with C >= 1")

    element_mask = control_map[0].detach().cpu().numpy()
    img = (np.clip(element_mask, 0.0, 1.0) * 255).astype(np.uint8)

    if mode == "element":
        img = to_uint8(element_mask)
    elif mode == "safe":
        if safe_zone is None:
            raise ValueError("safe_zone tensor is required when mode='safe'")
        sz = safe_zone.squeeze(0).detach().cpu().numpy().astype(np.float32)
        img = to_uint8(sz)
    elif mode == "edge":
        eroded = erode_3x3_binary(element_mask)
        border = np.clip(element_mask - eroded, 0.0, 1.0)
        img = to_uint8(border)
    else:
        raise ValueError("mode must be one of: 'element', 'safe', or 'edge'")

    rgb = np.stack([img, img, img], axis=2)
    pil = Image.fromarray(rgb, mode="RGB")

    w, h = size
    if pil.size != (w, h):
        pil = pil.resize((w, h), resample=Image.BILINEAR)

    return pil
