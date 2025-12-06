import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from text_type_predictor import TextTypePredictor
import easyocr

COLORS = {
    'Title': '#00FF00',
    'Caption': '#FF00FF',
    'Obj-text': '#FFFF00',
    'Other-text': '#00FFFF',
    'Page-text': '#FFA500'
}

def visualize_predictions(image_path, output_path, predictor, reader):

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    # Run OCR to get bboxes
    results = reader.readtext(str(image_path))
    
    # Process each detected text region
    for bbox, text, ocr_confidence in results:
        # Convert bbox to [x, y, width, height]
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        width = int(max(x_coords) - x_min)
        height = int(max(y_coords) - y_min)
        
        # Predict text type
        text_type, type_confidence = predictor.predict(image, [x_min, y_min, width, height])
        
        # Get color for this type
        color = COLORS.get(text_type, '#FFFFFF')
        
        # Draw bounding box
        draw.rectangle([x_min, y_min, x_min + width, y_min + height], 
                      outline=color, width=3)
        
        # Draw label background
        label = f"{text_type} ({type_confidence:.2f})"
        bbox_text = draw.textbbox((x_min, y_min - 25), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        
        # Draw label text
        draw.text((x_min, y_min - 25), label, fill='black', font=font)
    
    # Save annotated image
    output_path.parent.mkdir(exist_ok=True, parents=True)
    image.save(output_path)
    print(f"Saved visualization to: {output_path}")


def main():
    script_dir = Path(__file__).parent
    image_dir = script_dir / "sample_data" / "raw_images"
    output_dir = script_dir / "visualizations"
    model_dir = script_dir / "trained_models"
    
    # Initialize predictor and OCR
    predictor = TextTypePredictor(model_dir)
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Get all images (or just a few for testing)
    images = list(image_dir.glob("*.jpg"))[:5]  
    
    
    for image_path in images:
        output_path = output_dir / f"{image_path.stem}_predicted.jpg"
        print(f"\nProcessing: {image_path.name}")
        visualize_predictions(image_path, output_path, predictor, reader)
    

    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()