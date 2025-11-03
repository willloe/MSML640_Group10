
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import pickle

class TrainingDataPreparer: 
    def __init__(self, metadata_path, images_dir):
        self.metadata_path = Path(metadata_path)
        self.images_dir = Path(images_dir)
        self.label_mapping = {
            'Title': 0,
            'Caption': 1,
            'Text': 2,
            'Paragraph': 3,
            'TextBox': 4,
            'Heading': 5,
            'Label': 6,
            'PageNumber': 7
        }
        
    def load_metadata(self):
        with open(self.metadata_path, 'r') as f:
            return json.load(f)
    
    def extract_features_from_bbox(self, image, bbox):
        x, y, w, h = bbox
        img_width, img_height = image.size
        
        # Crop the region
        cropped = image.crop((x, y, x+w, y+h))
        cropped_array = np.array(cropped)
        
        gray = cv2.cvtColor(cropped_array, cv2.COLOR_RGB2GRAY)
        
        relative_y = y / img_height
        relative_x = x / img_width
        relative_center_x = (x + w/2) / img_width
        relative_center_y = (y + h/2) / img_height
        
        relative_width = w / img_width
        relative_height = h / img_height
        relative_area = (w * h) / (img_width * img_height)
        aspect_ratio = w / h if h > 0 else 0
        
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (w * h) if (w * h) > 0 else 0
        
   
        text_density = np.sum(gray < 240) / (w * h) if (w * h) > 0 else 0
        

        is_top_third = 1 if relative_y < 0.33 else 0
        is_middle_third = 1 if 0.33 <= relative_y < 0.67 else 0
        is_bottom_third = 1 if relative_y >= 0.67 else 0
        is_horizontally_centered = 1 if abs(relative_center_x - 0.5) < 0.15 else 0
  
        is_large = 1 if relative_area > 0.15 else 0
        is_wide = 1 if relative_width > 0.6 else 0
        is_tall = 1 if relative_height > 0.3 else 0
        
        features = {
            'relative_y': relative_y,
            'relative_x': relative_x,
            'relative_center_x': relative_center_x,
            'relative_center_y': relative_center_y,
            
            'relative_width': relative_width,
            'relative_height': relative_height,
            'relative_area': relative_area,
            'aspect_ratio': aspect_ratio,
 
            'mean_brightness': mean_brightness / 255.0,  
            'std_brightness': std_brightness / 255.0,
            'edge_density': edge_density,
            'text_density': text_density,
            

            'is_top_third': is_top_third,
            'is_middle_third': is_middle_third,
            'is_bottom_third': is_bottom_third,
            'is_horizontally_centered': is_horizontally_centered,
            'is_large': is_large,
            'is_wide': is_wide,
            'is_tall': is_tall,
        }
        
        return features
    
    def prepare_dataset(self):

        metadata = self.load_metadata()
        
        X = []
        y = []
        skipped = 0
        
        print(f"Processing {len(metadata)} slides...")
        
        for slide_data in metadata:
            filename = slide_data['filename']
            image_path = self.images_dir / filename
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                skipped += 1
                continue
            
            try:
                image = Image.open(image_path)
                
                for bbox_data in slide_data['bboxes']:
                    bbox = bbox_data['bbox']
                    class_name = bbox_data['class']
                    
                    if class_name not in self.label_mapping:
                        continue
                    
                    # Extract features
                    features = self.extract_features_from_bbox(image, bbox)
                    feature_vector = list(features.values())
                    
                    X.append(feature_vector)
                    y.append(self.label_mapping[class_name])
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                skipped += 1
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nDataset prepared:")
        print(f"  Total samples: {len(y)}")
        print(f"  Features per sample: {X.shape[1]}")
        print(f"\nClass distribution:")
        for class_name, class_id in self.label_mapping.items():
            count = np.sum(y == class_id)
            print(f"  {class_name}: {count}")
        
        feature_names = list(self.extract_features_from_bbox(
            Image.new('RGB', (100, 100)), [0, 0, 10, 10]
        ).keys())
        
        return X, y, feature_names
    
    def save_prepared_data(self, X, y, feature_names, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        np.save(output_dir / 'X.npy', X)
        np.save(output_dir / 'y.npy', y)
        
        with open(output_dir / 'feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
        
        with open(output_dir / 'label_mapping.json', 'w') as f:
            json.dump(self.label_mapping, f, indent=2)
        
        print(f"\nData saved to {output_dir}")


def main():
    script_dir = Path(__file__).parent
    metadata_path = script_dir / 'sample_data' / 'sample_metadata.json'
    images_dir = script_dir / 'sample_data' / 'raw_images'
    output_dir = script_dir / 'training_data'
    
    preparer = TrainingDataPreparer(metadata_path, images_dir)
    X, y, feature_names = preparer.prepare_dataset()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Test samples: {len(y_test)}")
    
    preparer.save_prepared_data(X, y, feature_names, output_dir)
    
    # Save split indices
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_test.npy', y_test)


if __name__ == '__main__':
    main()