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

    raise NotImplementedError("layout_safety will be implemented later")


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
