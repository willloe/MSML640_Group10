import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityValidator:
    def __init__(self, base_dir: Path, variance_threshold: float = 0.02):

        self.base_dir = base_dir
        self.variance_threshold = variance_threshold

    def check_file_integrity(self, sample: Dict) -> Dict[str, bool]:

        checks = {}

        file_keys = ['image_path', 'layout_json', 'safe_zone_path', 'control_map_path']

        for key in file_keys:
            if key in sample:
                file_path = self.base_dir / sample[key]
                checks[key] = file_path.exists()
            else:
                checks[key] = False

        return checks

    def check_safe_zone_variance(self, sample: Dict) -> Dict:

        image_path = self.base_dir / sample['image_path']
        safe_zone_path = self.base_dir / sample['safe_zone_path']

        image = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
        safe_zone = np.array(Image.open(safe_zone_path)).astype(np.float32) / 255.0

        safe_pixels = image[safe_zone > 0.5]

        if len(safe_pixels) == 0:
            return {
                'mean_intensity': None,
                'variance': None,
                'pass': False,
                'message': 'No safe zone pixels found'
            }

        mean_intensity = float(np.mean(safe_pixels))
        variance = float(np.var(safe_pixels))

        passes = variance <= self.variance_threshold

        return {
            'mean_intensity': mean_intensity,
            'variance': variance,
            'pass': passes,
            'threshold': self.variance_threshold,
            'message': 'OK' if passes else f'Variance {variance:.4f} exceeds threshold {self.variance_threshold}'
        }

    def check_leakage(self, sample: Dict, edge_threshold: float = 0.1) -> Dict:

        image_path = self.base_dir / sample['image_path']
        control_path = self.base_dir / sample['control_map_path']

        image = np.array(Image.open(image_path).convert('L'))
        control_map = np.array(Image.open(control_path))

        edges = cv2.Canny(image, 50, 150)

        element_mask = (control_map > 127).astype(np.uint8)
        edge_mask = (edges > 0).astype(np.uint8)

        overlap = np.logical_and(element_mask, edge_mask).sum()
        total_element_pixels = element_mask.sum()
        total_edges = edge_mask.sum()

        if total_element_pixels == 0:
            leakage_ratio = 0.0
        else:
            leakage_ratio = float(overlap / total_element_pixels)

        passes = leakage_ratio <= edge_threshold

        return {
            'edge_pixels_in_elements': int(overlap),
            'total_element_pixels': int(total_element_pixels),
            'total_edge_pixels': int(total_edges),
            'leakage_ratio': leakage_ratio,
            'pass': passes,
            'threshold': edge_threshold,
            'message': 'OK' if passes else f'Leakage ratio {leakage_ratio:.3f} exceeds threshold {edge_threshold}'
        }

    def validate_schema(self, sample: Dict) -> Dict:

        checks = {}

        palette = sample.get('palette', {})
        checks['has_primary_color'] = 'primary' in palette

        if 'primary' in palette:
            primary = palette['primary']
            checks['primary_valid_hex'] = (
                isinstance(primary, str) and
                primary.startswith('#') and
                len(primary) == 7
            )

        try:
            layout_path = self.base_dir / sample['layout_json']
            with open(layout_path, 'r') as f:
                layout = json.load(f)

            checks['has_canvas_size'] = 'canvas_size' in layout
            checks['has_elements'] = 'elements' in layout

            if 'elements' in layout:
                elements = layout['elements']
                checks['elements_count'] = len(elements)

                valid_elements = all(
                    'class' in e and 'bbox_xywh' in e
                    for e in elements
                )
                checks['all_elements_valid'] = valid_elements

        except Exception as e:
            checks['layout_error'] = str(e)

        all_pass = all(v for k, v in checks.items() if isinstance(v, bool))

        return {
            'checks': checks,
            'pass': all_pass,
            'message': 'OK' if all_pass else 'Schema validation failed'
        }

    def validate_sample(self, sample: Dict) -> Dict:

        logger.info(f"Validating {sample['id']}")

        report = {
            'id': sample['id'],
            'file_integrity': {},
            'safe_zone_variance': {},
            'leakage': {},
            'schema': {},
            'overall_pass': False
        }

        try:
            report['file_integrity'] = self.check_file_integrity(sample)
            files_ok = all(report['file_integrity'].values())
        except Exception as e:
            logger.error(f"File integrity check failed: {e}")
            report['file_integrity']['error'] = str(e)
            files_ok = False

        if not files_ok:
            logger.warning(f"File integrity check failed for {sample['id']}")
            return report

        try:
            report['safe_zone_variance'] = self.check_safe_zone_variance(sample)
            if not report['safe_zone_variance']['pass']:
                logger.warning(f"  {report['safe_zone_variance']['message']}")
        except Exception as e:
            logger.error(f"Safe zone variance check failed: {e}")
            report['safe_zone_variance']['error'] = str(e)

        try:
            report['leakage'] = self.check_leakage(sample)
            if not report['leakage']['pass']:
                logger.warning(f"  {report['leakage']['message']}")
        except Exception as e:
            logger.error(f"Leakage check failed: {e}")
            report['leakage']['error'] = str(e)

        try:
            report['schema'] = self.validate_schema(sample)
            if not report['schema']['pass']:
                logger.warning(f"  {report['schema']['message']}")
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            report['schema']['error'] = str(e)

        report['overall_pass'] = all([
            files_ok,
            report.get('safe_zone_variance', {}).get('pass', False),
            report.get('leakage', {}).get('pass', False),
            report.get('schema', {}).get('pass', False)
        ])

        if report['overall_pass']:
            logger.info(f"All checks passed")
        else:
            logger.warning(f"Some checks failed")

        return report


def main():
    parser = argparse.ArgumentParser(description='Validate quality of preprocessed samples')
    parser.add_argument('--index', type=str, default='data/processed/index.jsonl',
                        help='Path to index.jsonl')
    parser.add_argument('--output', type=str, default='data/processed/quality.jsonl',
                        help='Output path for quality reports')
    parser.add_argument('--variance-threshold', type=float, default=0.02,
                        help='Max allowed variance in safe zones')
    parser.add_argument('--edge-threshold', type=float, default=0.1,
                        help='Max allowed edge leakage ratio')

    args = parser.parse_args()

    index_path = Path(args.index)
    output_path = Path(args.output)

    if not index_path.exists():
        logger.error(f"Index file not found: {index_path}")
        return

    samples = []
    with open(index_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} samples from {index_path}")

    base_dir = index_path.parent.parent

    validator = QualityValidator(base_dir, args.variance_threshold)

    reports = []
    pass_count = 0

    for sample in samples:
        try:
            report = validator.validate_sample(sample)
            reports.append(report)

            if report['overall_pass']:
                pass_count += 1

        except Exception as e:
            logger.error(f"Failed to validate {sample['id']}: {e}")
            continue

    with open(output_path, 'w') as f:
        for report in reports:
            f.write(json.dumps(report) + '\n')

    logger.info(f"\nQuality validation complete:")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Passed: {pass_count}")
    logger.info(f"Failed: {len(samples) - pass_count}")
    logger.info(f"Pass rate: {pass_count / len(samples) * 100:.1f}%")


if __name__ == '__main__':
    main()
