import numpy as np
from PIL import Image
from typing import Any, Union, Tuple, List, Dict, Optional
from pathlib import Path


def _rgb_to_relative_luminance(rgb: np.ndarray) -> np.ndarray:

    rgb_norm = rgb.astype(np.float32) / 255.0

    def gamma_correct(channel):
        return np.where(
            channel <= 0.03928,
            channel / 12.92,
            np.power((channel + 0.055) / 1.055, 2.4)
        )

    r_linear = gamma_correct(rgb_norm[..., 0])
    g_linear = gamma_correct(rgb_norm[..., 1])
    b_linear = gamma_correct(rgb_norm[..., 2])


    luminance = 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear
    return luminance


def _contrast_ratio(lum1: np.ndarray, lum2: np.ndarray) -> np.ndarray:

    lighter = np.maximum(lum1, lum2)
    darker = np.minimum(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def wcag_pass_rate(
    image: Union[str, Path, Image.Image, np.ndarray],
    text_size: str = "normal",
    sample_grid: int = 10,
    return_details: bool = False
) -> Union[float, Tuple[float, Dict]]:

    if isinstance(image, (str, Path)):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        img = image.convert('RGB')
    elif isinstance(image, np.ndarray):
        if image.dtype == np.uint8 and len(image.shape) == 3:
            img = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("Array uint8 not with shape (H, W, 3)")
    else:
        raise TypeError(f"Unsupported image: {type(image)}")

    img_array = np.array(img)
    h, w = img_array.shape[:2]

    threshold = 4.5 if text_size == "normal" else 3.0

    x_samples = np.linspace(0, w - 1, sample_grid, dtype=int)
    y_samples = np.linspace(0, h - 1, sample_grid, dtype=int)

    bg_colors = []
    coords = []
    for y in y_samples:
        for x in x_samples:
            bg_colors.append(img_array[y, x])
            coords.append((int(x), int(y)))

    bg_colors = np.stack(bg_colors, axis=0)
    bg_lum = _rgb_to_relative_luminance(bg_colors)

    contrast_white = _contrast_ratio(bg_lum, 1.0)
    contrast_black = _contrast_ratio(bg_lum, 0.0)

    passes_white = contrast_white >= threshold
    passes_black = contrast_black >= threshold

    pass_rate_white = passes_white.mean() if passes_white.size > 0 else 0.0
    pass_rate_black = passes_black.mean() if passes_black.size > 0 else 0.0

    if pass_rate_white >= pass_rate_black:
        chosen_color = "white"
        chosen_passes = passes_white
        chosen_contrast = contrast_white
    else:
        chosen_color = "black"
        chosen_passes = passes_black
        chosen_contrast = contrast_black

    pass_rate = float(chosen_passes.mean()) if chosen_passes.size > 0 else 0.0

    if not return_details:
        return pass_rate

    sample_points = []
    for (x, y), bg, c_w, c_b, pw, pb in zip(
        coords,
        bg_colors,
        contrast_white,
        contrast_black,
        passes_white,
        passes_black,
    ):
        sample_points.append(
            {
                "x": x,
                "y": y,
                "bg_color": bg.tolist(),
                "contrast_white": float(c_w),
                "contrast_black": float(c_b),
                "passes_white": bool(pw),
                "passes_black": bool(pb),
            }
        )

    details = {
        "passes": int(chosen_passes.sum()),
        "total": int(chosen_passes.size),
        "threshold": float(threshold),
        "chosen_text_color": chosen_color,
        "pass_rate_white": float(pass_rate_white),
        "pass_rate_black": float(pass_rate_black),
        "mean_contrast_chosen": float(chosen_contrast.mean()),
        "min_contrast_chosen": float(chosen_contrast.min()),
        "max_contrast_chosen": float(chosen_contrast.max()),
        "sample_points": sample_points,
    }
    return pass_rate, details


def layout_safety(
    control_map: np.ndarray,
    generated_image: Union[str, Path, Image.Image, np.ndarray],
    threshold: float = 0.1
) -> Dict[str, float]:

    if isinstance(generated_image, (str, Path)):
        img = Image.open(generated_image).convert('RGB')
    elif isinstance(generated_image, Image.Image):
        img = generated_image.convert('RGB')
    elif isinstance(generated_image, np.ndarray):
        if generated_image.dtype == np.uint8 and len(generated_image.shape) == 3:
            img = Image.fromarray(generated_image, mode='RGB')
        else:
            raise ValueError("Array uint8 not with shape (H, W, 3)")
    else:
        raise TypeError(f"Unsupported image: {type(generated_image)}")

    img_array = np.array(img)

    if len(control_map.shape) == 3:
        element_mask = control_map[0]  # Shape: (H, W)
    else:
        raise ValueError(f"Control map need shape (4, H, W), instead of {control_map.shape}")

    if element_mask.shape != img_array.shape[:2]:
        element_mask_img = Image.fromarray((element_mask * 255).astype(np.uint8))
        element_mask_resized = element_mask_img.resize(
            (img_array.shape[1], img_array.shape[0]),
            Image.Resampling.NEAREST
        )
        element_mask = np.array(element_mask_resized) / 255.0

    img_gray = np.mean(img_array.astype(np.float32) / 255.0, axis=2)

    neutral_value = 245.0 / 255.0
    content_intensity = np.abs(img_gray - neutral_value)

    reserved_mask = element_mask > 0.5
    safe_zone_mask = element_mask <= 0.5

    total_pixels = element_mask.size
    reserved_pixels = np.sum(reserved_mask)
    safe_zone_pixels = np.sum(safe_zone_mask)

    overlap_mask = reserved_mask & (content_intensity > threshold)
    overlap_pixels = np.sum(overlap_mask)

    reserved_percent = (reserved_pixels / total_pixels) * 100.0
    safe_zone_percent = (safe_zone_pixels / total_pixels) * 100.0

    if reserved_pixels > 0:
        reserved_overlap_percent = (overlap_pixels / reserved_pixels) * 100.0
        mean_overlap = float(np.mean(content_intensity[reserved_mask]))
    else:
        reserved_overlap_percent = 0.0
        mean_overlap = 0.0

    return {
        'reserved_overlap_percent': float(reserved_overlap_percent),
        'mean_overlap': float(mean_overlap),
        'safe_zone_percent': float(safe_zone_percent),
        'reserved_percent': float(reserved_percent),
        'overlap_pixels': int(overlap_pixels),
        'reserved_pixels': int(reserved_pixels),
        'total_pixels': int(total_pixels)
    }

def _layout_masks_from_elements(
    img: Image.Image,
    layout: Dict,
) -> tuple[np.ndarray, np.ndarray]:
    w, h = img.size
    content_mask = np.zeros((h, w), dtype=bool)

    for el in layout.get("elements", []):
        bbox = el.get("bbox_xywh")
        if not bbox or len(bbox) != 4:
            continue
        x, y, bw, bh = bbox
        x0 = max(int(round(x)), 0)
        y0 = max(int(round(y)), 0)
        x1 = min(int(round(x + bw)), w)
        y1 = min(int(round(y + bh)), h)
        if x1 <= x0 or y1 <= y0:
            continue
        content_mask[y0:y1, x0:x1] = True

    background_mask = ~content_mask
    return content_mask, background_mask

def _safe_zone_to_mask(
    safe_zone: Any,
    size: Tuple[int, int],
) -> Optional[np.ndarray]:
    import torch

    W, H = size

    if isinstance(safe_zone, torch.Tensor):
        sz = safe_zone.detach().cpu().float()
        if sz.ndim == 3:
            sz = sz.mean(0)
        elif sz.ndim != 2:
            return None

        if sz.shape != (H, W):
            sz = torch.nn.functional.interpolate(
                sz.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        sz = sz.numpy()
        mask = sz > 0.0
        return mask.astype(bool)

    if isinstance(safe_zone, np.ndarray):
        sz = safe_zone.astype(np.float32)
        if sz.ndim == 3:
            sz = sz.mean(0)
        elif sz.ndim != 2:
            return None

        if sz.shape != (H, W):
            from PIL import Image as _PILImage

            sz_img = _PILImage.fromarray(sz)
            sz_img = sz_img.resize((W, H), _PILImage.NEAREST)
            sz = np.array(sz_img, dtype=np.float32)

        mask = sz > 0.0
        return mask.astype(bool)

    if isinstance(safe_zone, dict) and "mask" in safe_zone:
        return _safe_zone_to_mask(safe_zone["mask"], size)

    return None

def layout_uniformity_score(
    img: Image.Image,
    layout: Dict,
    safe_zone: Optional[Dict] = None,
) -> Dict[str, float]:
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    content_mask = None
    if safe_zone is not None:
        content_mask = _safe_zone_to_mask(safe_zone, img.size)

    if content_mask is None:
        content_mask, background_mask = _layout_masks_from_elements(img, layout)
    else:
        background_mask = ~content_mask

    safe_pixels = gray[content_mask]
    bg_pixels = gray[background_mask]

    if safe_pixels.size == 0:
        safe_std = float("nan")
    else:
        safe_std = float(safe_pixels.std())

    if bg_pixels.size == 0:
        bg_std = float("nan")
    else:
        bg_std = float(bg_pixels.std())

    eps = 1e-6
    if np.isfinite(safe_std) and safe_std > 0 and np.isfinite(bg_std):
        uniformity = float(bg_std / (safe_std + eps))
        uniformity_norm = float(bg_std / (bg_std + safe_std + eps))
    else:
        uniformity = float("nan")
        uniformity_norm = float("nan")

    return {
        "safe_std": safe_std,
        "bg_std": bg_std,
        "uniformity": uniformity,
        "uniformity_norm": uniformity_norm,
    }

if __name__ == "__main__":

    example_img = Image.new('RGB', (100, 100), color=(128, 128, 128))
    pass_rate = wcag_pass_rate(example_img, text_size="normal", sample_grid=5)
    print(f"Example pass rate: {pass_rate:.2%}")

    pass_rate, details = wcag_pass_rate(
        example_img,
        text_size="normal",
        sample_grid=5,
        return_details=True
    )
    print(f"\nresults:")
    print(f"Passes: {details['passes']}/{details['total']}")
    print(f"Mean contrast: {details['mean_contrast']:.2f}")
    print(f"Min contrast: {details['min_contrast']:.2f}")
    print(f"Max contrast: {details['max_contrast']:.2f}")
