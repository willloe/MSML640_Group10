import easyocr
from pathlib import Path
import json


def extract_text_from_image(image_path, reader=None):
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)
    
    results = reader.readtext(str(image_path))
    
    json_output = {
        'filename': Path(image_path).name,
        'num_texts': len(results),
        'extracted_texts': []
    }
    
    for bbox, text, confidence in results:
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        width = int(max(x_coords) - x_min)
        height = int(max(y_coords) - y_min)
        
        json_output['extracted_texts'].append({
            'text': text,
            'bbox': [x_min, y_min, width, height],
            'confidence': round(float(confidence), 3)
        })
    
    return json_output


def save_extraction_to_json(extraction_result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump(extraction_result, f, indent=2)
    
    print(f"Json Saved to: {output_path}")


def main():
    script_dir = Path(__file__).parent
    image_dir = script_dir / "sample_data" / "raw_images"
    output_dir = script_dir / "output"
    
    images = list(image_dir.glob("*.jpg"))
    if not images:
        print("No images found.....")
        return
    
    # testing for only 1 image
    # test_image = images[0]
    reader = easyocr.Reader(['en'], gpu=False)

    #looping through all images
    for image in images:
        print(f"Processing image: {image.name}")
        result = extract_text_from_image(image, reader)

        print(f"\nExtracted {result['num_texts']} text elements")
        output_path = output_dir / f"{image.stem}_extracted.json"
        save_extraction_to_json(result, output_path)

if __name__ == "__main__":
    main()
