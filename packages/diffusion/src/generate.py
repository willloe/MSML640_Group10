import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from PIL import Image


CLASS_MAP = {
    'title': 1,
    'body': 2,
    'image': 3,
    'logo': 4,
    'caption': 5,
    'footer': 6,
}


def create_layout_control_map(
    layout_json: Dict,
    canvas_size: Optional[Tuple[int, int]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    if canvas_size is not None:
        canvas_h, canvas_w = canvas_size
    elif 'canvas_size' in layout_json:
        canvas_h, canvas_w = layout_json['canvas_size']
    else:
        raise ValueError("Canvas size must be provided either in layout_json or as canvas_size")

    control_map = torch.zeros(4, canvas_h, canvas_w, dtype=torch.float32)
    elements = layout_json.get('elements', [])

    for elem in elements:
        x, y, w, h = elem['bbox_xywh']
        x = max(0, min(x, canvas_w - 1))
        y = max(0, min(y, canvas_h - 1))
        x_end = max(0, min(x + w, canvas_w))
        y_end = max(0, min(y + h, canvas_h))

        if x_end <= x or y_end <= y:
            continue

        elem_class = elem.get('class', 'body')
        class_id = CLASS_MAP.get(elem_class, 2) 

        z_order = elem.get('z_order', 0)
        reading_order = elem.get('reading_order', 0)

        control_map[0, y:y_end, x:x_end] = 1.0
        control_map[1, y:y_end, x:x_end] = class_id / len(CLASS_MAP)
        control_map[2, y:y_end, x:x_end] = min(z_order / 10.0, 1.0)
        control_map[3, y:y_end, x:x_end] = min(reading_order / 10.0, 1.0)

    safe_zone_mask = 1.0 - control_map[0:1]

    return control_map, safe_zone_mask


def apply_safe_zone_mask(
    generated_image: Image.Image,
    safe_zone_mask: torch.Tensor,
    neutral_color: Tuple[int, int, int] = (240, 240, 240)
) -> Image.Image:

    img_array = np.array(generated_image)

    if safe_zone_mask.shape[1:] != img_array.shape[:2]:
        mask_np = safe_zone_mask.squeeze(0).numpy()
        mask_resized = np.array(
            Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (img_array.shape[1], img_array.shape[0]),
                Image.Resampling.BILINEAR
            )
        ) / 255.0
    else:
        mask_resized = safe_zone_mask.squeeze(0).numpy()

    neutral_bg = np.full_like(img_array, neutral_color)
    mask_3d = np.expand_dims(mask_resized, axis=2)
    result = (img_array * mask_3d + neutral_bg * (1 - mask_3d)).astype(np.uint8)

    return Image.fromarray(result)


def visualize_control_map(
    control_map: torch.Tensor,
    save_path: Optional[str] = None
) -> Image.Image:

    channels = []
    for i in range(4):
        channel = control_map[i].numpy()
        channel_norm = ((channel - channel.min()) / (channel.max() - channel.min() + 1e-8) * 255).astype(np.uint8)
        channels.append(Image.fromarray(channel_norm, mode='L'))

    h, w = control_map.shape[1:]
    vis_image = Image.new('RGB', (w * 4, h))

    channel_names = ['Element Mask', 'Class Importance', 'Z-Order', 'Reading Order']
    for i, (channel_img, name) in enumerate(zip(channels, channel_names)):
        channel_rgb = channel_img.convert('RGB')
        vis_image.paste(channel_rgb, (i * w, 0))

    if save_path:
        vis_image.save(save_path)

    return vis_image


def main():

    print('Layout to Control Map Conversion\n')

    example_layout = {
        'canvas_size': [1024, 768],
        'elements': [
            {
                'class': 'title',
                'bbox_xywh': [100, 50, 800, 100],
                'z_order': 2,
                'reading_order': 1
            },
            {
                'class': 'body',
                'bbox_xywh': [100, 200, 800, 400],
                'z_order': 1,
                'reading_order': 2
            },
            {
                'class': 'logo',
                'bbox_xywh': [850, 650, 120, 80],
                'z_order': 3,
                'reading_order': 3
            }
        ]
    }

    print(f"Input layout: {len(example_layout['elements'])} elements")
    print(f"Canvas size: {example_layout['canvas_size']}\n")

    control_map, safe_zone_mask = create_layout_control_map(example_layout)

    print(f"Generated control map: {control_map.shape}")
    print(f"Generated safe zone mask: {safe_zone_mask.shape}")
    print(f"\nControl map statistics:")
    print(f"  Element coverage: {(control_map[0] > 0).sum().item() / control_map[0].numel() * 100:.2f}%")
    print(f"  Safe zone coverage: {(safe_zone_mask > 0).sum().item() / safe_zone_mask.numel() * 100:.2f}%")

    print("\nGenerating visualization...")
    vis = visualize_control_map(control_map, save_path='control_map_demo.png')
    print("Saved control map visualization to: control_map_demo.png")

    print("\nDemo: Creating placeholder background with safe zone masking...")
    placeholder_bg = Image.new('RGB', (1024, 768), color=(100, 150, 200))
    masked_bg = apply_safe_zone_mask(placeholder_bg, safe_zone_mask, neutral_color=(245, 245, 245))
    masked_bg.save('safe_zone_demo.png')
    print("Saved safe zone demo to: safe_zone_demo.png")

    print("\n=== Demo Complete ===")
    print("Next steps:")
    print("  1. Integrate with SDXL pipeline for actual generation")
    print("  2. Implement ControlNet conditioning")
    print("  3. Add palette-based color conditioning")


if __name__ == '__main__':
    main()
