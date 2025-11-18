import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2

try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False
    try:
        import pytesseract
        HAS_TESSERACT = True
    except ImportError:
        HAS_TESSERACT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlidePreprocessor:

    TARGET_SIZE = (1024, 768)

    def __init__(self, use_paddle: bool = True):
        self.ocr = None
        if use_paddle and HAS_PADDLE:
            logger.info("Initializing")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.ocr_type = 'paddle'
        elif HAS_TESSERACT:
            logger.info("Using Tesseract OCR")
            self.ocr_type = 'tesseract'
        else:
            logger.warning("No OCR library available")
            self.ocr_type = None

    def normalize_image(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = TARGET_SIZE
    ) -> Tuple[Image.Image, Dict]:

        orig_width, orig_height = image.size
        target_width, target_height = target_size

        if image.mode != 'RGB':
            image = image.convert('RGB')

        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        canvas = Image.new('RGB', target_size, (0, 0, 0))

        pad_left = (target_width - new_width) // 2
        pad_top = (target_height - new_height) // 2

        canvas.paste(image, (pad_left, pad_top))

        metadata = {
            'original_size': [orig_width, orig_height],
            'resized_size': [new_width, new_height],
            'padding': [pad_left, pad_top, target_width - new_width - pad_left, target_height - new_height - pad_top],
            'scale': scale
        }

        return canvas, metadata

    def extract_layout_ocr(self, image: Image.Image) -> List[Dict]:
        if self.ocr_type is None:
            logger.warning("No OCR available")
            return []

        img_array = np.array(image)

        elements = []

        if self.ocr_type == 'paddle':
            result = self.ocr.ocr(img_array, cls=True)

            if result is None or len(result) == 0 or result[0] is None:
                return elements

            for line in result[0]:
                box = line[0]
                text = line[1][0]
                conf = line[1][1]

                if conf < 0.5:
                    continue

                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                x = int(min(x_coords))
                y = int(min(y_coords))
                w = int(max(x_coords) - x)
                h = int(max(y_coords) - y)

                elements.append({
                    'bbox_xywh': [x, y, w, h],
                    'text': text,
                    'confidence': float(conf)
                })

        elif self.ocr_type == 'tesseract':
            data = pytesseract.image_to_data(img_array, output_type=pytesseract.Output.DICT)

            for i in range(len(data['text'])):
                if int(data['conf'][i]) < 50:
                    continue

                text = data['text'][i].strip()
                if not text:
                    continue

                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                elements.append({
                    'bbox_xywh': [x, y, w, h],
                    'text': text,
                    'confidence': int(data['conf'][i]) / 100.0
                })

        return elements

    def merge_nearby_boxes(
        self,
        elements: List[Dict],
        horizontal_threshold: int = 20,
        vertical_threshold: int = 10
    ) -> List[Dict]:
        if not elements:
            return []
        sorted_elements = sorted(elements, key=lambda e: e['bbox_xywh'][1])

        lines = []
        current_line = [sorted_elements[0]]

        for elem in sorted_elements[1:]:
            prev_y = current_line[-1]['bbox_xywh'][1]
            prev_h = current_line[-1]['bbox_xywh'][3]
            curr_y = elem['bbox_xywh'][1]

            if abs(curr_y - prev_y) < max(prev_h, elem['bbox_xywh'][3]) * 0.5:
                current_line.append(elem)
            else:
                lines.append(current_line)
                current_line = [elem]

        if current_line:
            lines.append(current_line)

        merged = []

        for line_idx, line in enumerate(lines):
            line = sorted(line, key=lambda e: e['bbox_xywh'][0])

            x_min = min(e['bbox_xywh'][0] for e in line)
            y_min = min(e['bbox_xywh'][1] for e in line)
            x_max = max(e['bbox_xywh'][0] + e['bbox_xywh'][2] for e in line)
            y_max = max(e['bbox_xywh'][1] + e['bbox_xywh'][3] for e in line)

            if line_idx == 0:
                elem_type = 'title'
            elif y_max > 700:
                elem_type = 'footer'
            else:
                elem_type = 'body'

            merged.append({
                'class': elem_type,
                'bbox_xywh': [x_min, y_min, x_max - x_min, y_max - y_min],
                'z_order': line_idx,
                'reading_order': line_idx
            })

        return merged

    def create_safe_zone_mask(
        self,
        elements: List[Dict],
        canvas_size: Tuple[int, int] = TARGET_SIZE,
        blur_radius: int = 3
    ) -> np.ndarray:

        width, height = canvas_size
        mask = np.ones((height, width), dtype=np.float32)

        for elem in elements:
            x, y, w, h = elem['bbox_xywh']
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            w = min(w, width - x)
            h = min(h, height - y)

            mask[y:y+h, x:x+w] = 0.0

        if blur_radius > 0:
            mask = cv2.GaussianBlur(mask, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)

        return mask

    def create_control_map(
        self,
        elements: List[Dict],
        canvas_size: Tuple[int, int] = TARGET_SIZE
    ) -> np.ndarray:

        width, height = canvas_size
        mask = np.zeros((height, width), dtype=np.uint8)

        for elem in elements:
            x, y, w, h = elem['bbox_xywh']
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            w = min(w, width - x)
            h = min(h, height - y)

            mask[y:y+h, x:x+w] = 255

        return mask

    def extract_palette(self, image: Image.Image, n_colors: int = 3) -> Dict:
        img_array = np.array(image)

        pixels = img_array.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        centers = centers.astype(int)
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in centers]

        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)

        palette = {
            'primary': hex_colors[sorted_indices[0]] if len(sorted_indices) > 0 else '#000000',
            'bg_style': 'extracted from slide'
        }

        if len(sorted_indices) > 1:
            palette['secondary'] = hex_colors[sorted_indices[1]]

        if len(sorted_indices) > 2:
            palette['accent'] = hex_colors[sorted_indices[2]]

        return palette

    def process_single_image(
        self,
        input_path: Path,
        output_dir: Path,
        sample_id: str
    ) -> Dict:

        logger.info(f"Processing {sample_id}: {input_path.name}")

        images_dir = output_dir / 'images'
        masks_dir = output_dir / 'masks'
        json_dir = output_dir / 'json'
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(input_path)
        normalized_image, norm_metadata = self.normalize_image(image)

        image_path = images_dir / f"{sample_id}.png"
        normalized_image.save(image_path)

        ocr_elements = self.extract_layout_ocr(normalized_image)
        merged_elements = self.merge_nearby_boxes(ocr_elements)

        layout = {
            'canvas_size': [self.TARGET_SIZE[1], self.TARGET_SIZE[0]],  # [height, width]
            'elements': merged_elements
        }

        layout_path = json_dir / f"{sample_id}.layout.json"
        with open(layout_path, 'w') as f:
            json.dump(layout, f, indent=2)

        safe_zone_mask = self.create_safe_zone_mask(merged_elements)
        safe_zone_path = masks_dir / f"{sample_id}.safe.png"
        Image.fromarray((safe_zone_mask * 255).astype(np.uint8)).save(safe_zone_path)

        control_map = self.create_control_map(merged_elements)
        control_path = masks_dir / f"{sample_id}.control.png"
        Image.fromarray(control_map).save(control_path)

        palette = self.extract_palette(normalized_image)

        metadata = {
            'id': sample_id,
            'image_path': f"processed/images/{sample_id}.png",
            'layout_json': f"processed/json/{sample_id}.layout.json",
            'safe_zone_path': f"processed/masks/{sample_id}.safe.png",
            'control_map_path': f"processed/masks/{sample_id}.control.png",
            'palette': palette,
            'width': self.TARGET_SIZE[0],
            'height': self.TARGET_SIZE[1],
            'normalization': norm_metadata
        }

        logger.info(f"  -> Extracted {len(merged_elements)} layout elements")
        logger.info(f"  -> Palette: {palette['primary']}")

        return metadata


def main():
    parser = argparse.ArgumentParser(description='Preprocess slide images')
    parser.add_argument('--input', type=str, default='data/raw',
                        help='Input directory with raw images')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory')
    parser.add_argument('--use-tesseract', action='store_true',
                        help='Use Tesseract instead of PaddleOCR')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    preprocessor = SlidePreprocessor(use_paddle=not args.use_tesseract)

    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = [f for f in input_dir.iterdir()
                   if f.is_file() and f.suffix in image_extensions]

    if not image_files:
        logger.warning(f"No images found: {input_dir}")
        return

    logger.info(f"Found {len(image_files)} images to process")

    all_metadata = []

    for idx, image_file in enumerate(image_files):
        sample_id = f"sample_{idx:06d}"

        try:
            metadata = preprocessor.process_single_image(
                image_file,
                output_dir,
                sample_id
            )
            all_metadata.append(metadata)
        except Exception as e:
            logger.error(f"Failed to process {image_file.name}: {e}")
            continue

    index_path = output_dir / 'index.jsonl'
    with open(index_path, 'w') as f:
        for metadata in all_metadata:
            f.write(json.dumps(metadata) + '\n')

    logger.info(f"Processed {len(all_metadata)} images")


if __name__ == '__main__':
    main()
