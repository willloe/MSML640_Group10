import numpy as np
from PIL import Image
from typing import Union, Tuple, List, Dict
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
    sample_points = []
    contrast_ratios = []

    for y in y_samples:
        for x in x_samples:
            bg_color = img_array[y, x]
            bg_lum = _rgb_to_relative_luminance(bg_color.reshape(1, 3))[0]

            white_contrast = _contrast_ratio(bg_lum, 1.0)
            black_contrast = _contrast_ratio(bg_lum, 0.0)
            best_contrast = max(white_contrast, black_contrast)
            contrast_ratios.append(best_contrast)
            passes = best_contrast >= threshold
            sample_points.append({
                'x': int(x),
                'y': int(y),
                'bg_color': bg_color.tolist(),
                'contrast': float(best_contrast),
                'passes': passes
            })


    passes = sum(1 for p in sample_points if p['passes'])
    total = len(sample_points)
    pass_rate = passes / total if total > 0 else 0.0

    if return_details:
        details = {
            'passes': passes,
            'total': total,
            'contrast_ratios': contrast_ratios,
            'threshold': threshold,
            'sample_points': sample_points,
            'mean_contrast': float(np.mean(contrast_ratios)),
            'min_contrast': float(np.min(contrast_ratios)),
            'max_contrast': float(np.max(contrast_ratios))
        }
        return pass_rate, details

    return pass_rate


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
