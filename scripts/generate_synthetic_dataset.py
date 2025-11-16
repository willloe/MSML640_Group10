import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'packages' / 'diffusion' / 'src'))

from synthetic import random_palette, random_layout, save_control_visuals
from generate import create_layout_control_map
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_background(
    palette: dict,
    canvas_size: tuple = (1024, 768)
) -> Image.Image:

    width, height = canvas_size

    bg_style = palette.get('bg_style', 'minimalist gradient')

    primary = palette['primary'].lstrip('#')
    r1, g1, b1 = int(primary[0:2], 16), int(primary[2:4], 16), int(primary[4:6], 16)

    if 'secondary' in palette:
        secondary = palette['secondary'].lstrip('#')
        r2, g2, b2 = int(secondary[0:2], 16), int(secondary[2:4], 16), int(secondary[4:6], 16)
    else:
        r2, g2, b2 = min(255, r1 + 50), min(255, g1 + 50), min(255, b1 + 50)

    gradient = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        ratio = y / height
        r = int(r1 * (1 - ratio) + r2 * ratio)
        g = int(g1 * (1 - ratio) + g2 * ratio)
        b = int(b1 * (1 - ratio) + b2 * ratio)
        gradient[y, :] = [r, g, b]

    return Image.fromarray(gradient)


def generate_synthetic_sample(
    sample_id: str,
    output_dir: Path,
    seed: int = None
) -> dict:

    logger.info(f"Generating {sample_id}")

    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    json_dir = output_dir / 'json'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    palette = random_palette(seed=seed)
    layout = random_layout(canvas_w=1024, canvas_h=768, seed=seed)

    background = generate_synthetic_background(palette, canvas_size=(1024, 768))
    image_path = images_dir / f"{sample_id}.png"
    background.save(image_path)

    layout_path = json_dir / f"{sample_id}.layout.json"
    with open(layout_path, 'w') as f:
        json.dump(layout, f, indent=2)

    control_map, safe_zone_mask = create_layout_control_map(layout)

    safe_zone_array = safe_zone_mask.numpy()[0]
    safe_zone_image = (safe_zone_array * 255).astype(np.uint8)
    safe_zone_path = masks_dir / f"{sample_id}.safe.png"
    Image.fromarray(safe_zone_image).save(safe_zone_path)

    element_mask = control_map.numpy()[0]  
    control_image = (element_mask * 255).astype(np.uint8)
    control_path = masks_dir / f"{sample_id}.control.png"
    Image.fromarray(control_image).save(control_path)

    metadata = {
        'id': sample_id,
        'image_path': f"processed/images/{sample_id}.png",
        'layout_json': f"processed/json/{sample_id}.layout.json",
        'safe_zone_path': f"processed/masks/{sample_id}.safe.png",
        'control_map_path': f"processed/masks/{sample_id}.control.png",
        'palette': palette,
        'width': 1024,
        'height': 768,
        'source': 'synthetic'
    }

    logger.info(f"Generated {len(layout['elements'])} elements")
    logger.info(f"Palette: {palette['primary']}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--start-id', type=int, default=0,
                        help='Starting sample ID number')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.num_samples} samples")

    all_metadata = []

    for i in range(args.num_samples):
        sample_id = f"sample_{args.start_id + i:06d}"
        seed = args.seed + i if args.seed is not None else None

        try:
            metadata = generate_synthetic_sample(sample_id, output_dir, seed)
            all_metadata.append(metadata)
        except Exception as e:
            logger.error(f"Failed to generate {sample_id}: {e}")
            continue

    index_path = output_dir / 'index.jsonl'

    existing_samples = []
    if index_path.exists():
        with open(index_path, 'r') as f:
            for line in f:
                existing_samples.append(json.loads(line))

        logger.info(f"Appending to existing index with {len(existing_samples)} samples")

    with open(index_path, 'w') as f:
        for metadata in existing_samples + all_metadata:
            f.write(json.dumps(metadata) + '\n')

    logger.info(f"\nGenerated {len(all_metadata)} synthetic samples")
    logger.info(f"Total samples in index: {len(existing_samples) + len(all_metadata)}")



if __name__ == '__main__':
    main()
