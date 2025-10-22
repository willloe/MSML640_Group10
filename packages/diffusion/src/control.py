from typing import Tuple
import numpy as np
from PIL import Image
import torch

def control_image_from_map(control_map: torch.Tensor, size: Tuple[int, int]) -> Image.Image:
    if not control_map.ndim == 3 and control_map.shape[0] >= 1:
        raise ValueError("control_map must be [C,H,W]")

    element_mask = control_map[0].detach().cpu().numpy()
    img = (np.clip(element_mask, 0.0, 1.0) * 255).astype(np.uint8)

    rgb = np.stack([img, img, img], axis=2)
    pil = Image.fromarray(rgb, mode="RGB")

    w, h = size
    if pil.size != (w, h):
        pil = pil.resize((w, h), resample=Image.BILINEAR)

    return pil
