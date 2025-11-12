import easyocr
from pathlib import Path
import json
import re
from PIL import Image
from text_type_predictor import TextTypePredictor


def extract_text_from_image(image_path, reader=None, predictor=None):
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)
    
    # Load image
    image = Image.open(image_path)
    
    results = reader.readtext(str(image_path))
    
    json_output = {
        'filename': Path(image_path).name,
        'num_texts': len(results),
        'extracted_texts': []
    }
    
    # Collect all bboxes for batch prediction
    bboxes = []
    for bbox, text, confidence in results:
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        width = int(max(x_coords) - x_min)
        height = int(max(y_coords) - y_min)
        
        bboxes.append([x_min, y_min, width, height])
    
    # Predict text types for all bboxes
    if predictor and bboxes:
        text_types = predictor.predict_batch(image, bboxes)
    else:
        text_types = [('unknown', 0.0)] * len(bboxes)
    
    # Build output
    for i, (bbox, text, confidence) in enumerate(results):
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        width = int(max(x_coords) - x_min)
        height = int(max(y_coords) - y_min)
        
        text_type, type_confidence = text_types[i]
        
        json_output['extracted_texts'].append({
            'text': text,
            'bbox': [x_min, y_min, width, height],
            'confidence': round(float(confidence), 3),
            'type': text_type,
            'type_confidence': round(float(type_confidence), 3)
        })
    
    return json_output


def save_extraction_to_json(extraction_result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Serialize to string
    json_string = json.dumps(extraction_result, indent=2)
    
    # Compact bbox arrays to single line
    json_string = re.sub(
        r'"bbox":\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]',
        r'"bbox": [\1, \2, \3, \4]',
        json_string
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_string)
    
    print(f"Json Saved to: {output_path}")


def main():
    script_dir = Path(__file__).parent
    image_dir = script_dir / "sample_data" / "raw_images"
    output_dir = script_dir / "output_with_types"
    model_dir = script_dir / "trained_models"
    
    # Initialize OCR reader
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Initialize classifier (if model exists)
    predictor = None
    if (model_dir / 'classifier_model.pkl').exists():
        print("Loading trained classifier...")
        predictor = TextTypePredictor(model_dir)
        print("Classifier loaded successfully!")
    else:
        print("No trained model found. Run train_classifier.py first.")
        print("Proceeding without text type classification...")
    
    images = list(image_dir.glob("*.jpg"))
    if not images:
        print("No images found.....")
        return
    
    # Process all images
    for image in images:
        print(f"Processing image: {image.name}")
        result = extract_text_from_image(image, reader, predictor)
        
        print(f"  Extracted {result['num_texts']} text elements")
        output_path = output_dir / f"{image.stem}_extracted.json"
        save_extraction_to_json(result, output_path)
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print('='*60)


if __name__ == "__main__":
    main()