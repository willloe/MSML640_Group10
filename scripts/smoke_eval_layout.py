import sys
from pathlib import Path
import csv
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
sys.path.append(str(SRC))

from evaluate import layout_safety
from generate import create_layout_control_map


def main():
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sample_layout = {
        'canvas_size': [768, 1024],
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

    control_map, safe_zone_mask = create_layout_control_map(sample_layout)
    print(f"\nControl map shape: {control_map.shape}")
    print(f"Safe zone mask shape: {safe_zone_mask.shape}")

    test_image_path = outputs_dir / "gen_masked.png"

    if not test_image_path.exists():

        h, w = sample_layout['canvas_size']
        sample_img = Image.new('RGB', (w, h), color=(245, 245, 245))

        pixels = sample_img.load()
        for y in range(100, 200):
            for x in range(200, 400):
                pixels[x, y] = (100, 150, 200) 

        test_image_path = outputs_dir / "test_layout.png"
        sample_img.save(test_image_path)


    metrics = layout_safety(
        control_map=control_map.numpy(),
        generated_image=test_image_path,
        threshold=0.1
    )

    print(f"\nLayout safety:")
    print(f"Reserved overlap percent: {metrics['reserved_overlap_percent']:.2f}%")
    print(f"Mean overlap intensity:   {metrics['mean_overlap']:.4f}")
    print(f"Safe zone percent:        {metrics['safe_zone_percent']:.2f}%")
    print(f"Reserved percent:         {metrics['reserved_percent']:.2f}%")

    print(f"\nDetails:")
    print(f"Overlap pixels:           {metrics['overlap_pixels']:,}")
    print(f"Reserved pixels:          {metrics['reserved_pixels']:,}")
    print(f"Total pixels:             {metrics['total_pixels']:,}")

    csv_path = outputs_dir / "metrics.csv"
    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                'metric_type',
                'image',
                'reserved_overlap_percent',
                'mean_overlap',
                'safe_zone_percent',
                'reserved_percent',
                'overlap_pixels',
                'reserved_pixels',
                'total_pixels'
            ])

        writer.writerow([
            'layout_safety',
            test_image_path.name,
            f"{metrics['reserved_overlap_percent']:.4f}",
            f"{metrics['mean_overlap']:.6f}",
            f"{metrics['safe_zone_percent']:.4f}",
            f"{metrics['reserved_percent']:.4f}",
            metrics['overlap_pixels'],
            metrics['reserved_pixels'],
            metrics['total_pixels']
        ])


    print("Results:")
    print(f"Reserved overlap percent: {metrics['reserved_overlap_percent']:.2f}%")
    print(f"Mean overlap:             {metrics['mean_overlap']:.4f}")

    target_overlap = 5.0
    if metrics['reserved_overlap_percent'] <= target_overlap:
        print(f"\nPASS: Reserved overlap {metrics['reserved_overlap_percent']:.2f}% <= {target_overlap}%")
    else:
        print(f"\nFAIL: Reserved overlap {metrics['reserved_overlap_percent']:.2f}% > {target_overlap}%")


if __name__ == "__main__":
    main()
