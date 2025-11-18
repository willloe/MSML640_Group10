import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple
import colorsys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hex_to_hue_family(hex_color: str) -> str:
    hex_color = hex_color.lstrip('#')

    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    if s < 0.1:
        return 'grayscale'

    hue_deg = h * 360

    if hue_deg < 15 or hue_deg >= 345:
        return 'red'
    elif hue_deg < 45:
        return 'orange'
    elif hue_deg < 75:
        return 'yellow'
    elif hue_deg < 165:
        return 'green'
    elif hue_deg < 195:
        return 'cyan'
    elif hue_deg < 255:
        return 'blue'
    elif hue_deg < 285:
        return 'purple'
    else:
        return 'pink'


def get_layout_density(layout_json_path: Path) -> str:

    with open(layout_json_path, 'r') as f:
        layout = json.load(f)

    elements = layout.get('elements', [])
    num_elements = len(elements)

    canvas_h, canvas_w = layout['canvas_size']
    total_area = canvas_h * canvas_w

    covered_area = sum(e['bbox_xywh'][2] * e['bbox_xywh'][3] for e in elements)
    area_ratio = covered_area / total_area

    if num_elements <= 2 or area_ratio < 0.2:
        return 'low'
    elif num_elements <= 4 or area_ratio < 0.4:
        return 'medium'
    else:
        return 'high'


def stratify_samples(
    samples: List[Dict],
    base_dir: Path
) -> Dict[str, List[Dict]]:
    
    strata = {}

    for sample in samples:
        primary_color = sample['palette']['primary']
        hue_family = hex_to_hue_family(primary_color)

        layout_path = base_dir / sample['layout_json']
        density = get_layout_density(layout_path)

        stratum = f"{hue_family}_{density}"

        if stratum not in strata:
            strata[stratum] = []

        strata[stratum].append(sample)

    return strata


def split_stratum(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:

    random.seed(seed)
    samples = samples.copy()
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    return train, val, test


def create_splits(
    index_path: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):

    samples = []
    with open(index_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} samples from {index_path}")

    base_dir = index_path.parent.parent

    strata = stratify_samples(samples, base_dir)

    logger.info(f"Created {len(strata)} strata:")
    for stratum, stratum_samples in sorted(strata.items()):
        logger.info(f"  {stratum}: {len(stratum_samples)} samples")

    all_train = []
    all_val = []
    all_test = []

    for stratum, stratum_samples in strata.items():
        train, val, test = split_stratum(
            stratum_samples,
            train_ratio,
            val_ratio,
            test_ratio,
            seed
        )

        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)

        logger.info(f"  {stratum}: train={len(train)}, val={len(val)}, test={len(test)}")

    random.seed(seed)
    random.shuffle(all_train)
    random.shuffle(all_val)
    random.shuffle(all_test)

    logger.info(f"\nFinal splits:")
    logger.info(f"Train: {len(all_train)}")
    logger.info(f"Val: {len(all_val)}")
    logger.info(f"Test: {len(all_test)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in [
        ('train', all_train),
        ('val', all_val),
        ('test', all_test)
    ]:
        split_file = output_dir / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for sample in split_samples:
                f.write(sample['id'] + '\n')

        logger.info(f"Wrote {split_file}")


    for split_name, split_samples in [
        ('train', all_train),
        ('val', all_val),
        ('test', all_test)
    ]:
        split_file = output_dir / f"{split_name}.jsonl"
        with open(split_file, 'w') as f:
            for sample in split_samples:
                f.write(json.dumps(sample) + '\n')

        logger.info(f"Wrote detailed {split_file}")

    stats = {
        'total_samples': len(samples),
        'train_samples': len(all_train),
        'val_samples': len(all_val),
        'test_samples': len(all_test),
        'strata': {
            stratum: len(stratum_samples)
            for stratum, stratum_samples in sorted(strata.items())
        },
        'seed': seed
    }

    stats_file = output_dir / 'split_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nSplit saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Create stratified train/val/test splits')
    parser.add_argument('--index', type=str, default='data/processed/index.jsonl',
                        help='Path to index.jsonl')
    parser.add_argument('--output', type=str, default='data/splits',
                        help='Output directory for splits')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.error(f"Ratios must sum to 1.0, got {total_ratio}")
        return

    index_path = Path(args.index)
    output_dir = Path(args.output)

    if not index_path.exists():
        logger.error(f"Index file not found: {index_path}")
        return

    create_splits(
        index_path,
        output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )


if __name__ == '__main__':
    main()
