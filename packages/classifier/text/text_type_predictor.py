# File 3: packages/classifier/text/text_type_predictor.py
import numpy as np
import json
import pickle
from pathlib import Path
from PIL import Image
import cv2
import joblib


class TextTypePredictor:
    """Predict text type using trained classifier"""
    
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / 'classifier_model.pkl')
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        
        with open(self.model_dir.parent / 'training_data' / 'label_mapping.json', 'r') as f:
            self.label_mapping = json.load(f)
        
        # Reverse mapping
        self.id_to_label = {v: k for k, v in self.label_mapping.items()}
    
    def extract_features_from_bbox(self, image, bbox):
        """
        Extract features from a bounding box region
        (Same as in TrainingDataPreparer)
        """
        x, y, w, h = bbox
        img_width, img_height = image.size
        
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
        
        features = np.array([
            relative_y, relative_x, relative_center_x, relative_center_y,
            relative_width, relative_height, relative_area, aspect_ratio,
            mean_brightness / 255.0, std_brightness / 255.0,
            edge_density, text_density,
            is_top_third, is_middle_third, is_bottom_third, 
            is_horizontally_centered, is_large, is_wide, is_tall
        ])
        
        return features
    
    def predict(self, image, bbox):
        """
        Predict text type for a bounding box
        
        Args:
            image: PIL Image
            bbox: [x, y, width, height]
            
        Returns:
            text_type: Predicted class name
            confidence: Prediction confidence
        """
        features = self.extract_features_from_bbox(image, bbox)
        features_scaled = self.scaler.transform([features])
        
        pred_id = self.model.predict(features_scaled)[0]
        pred_proba = self.model.predict_proba(features_scaled)[0]
        
        text_type = self.id_to_label[pred_id]
        confidence = pred_proba[pred_id]
        
        return text_type, confidence
    
    def predict_batch(self, image, bboxes):
        """
        Predict text types for multiple bounding boxes
        
        Args:
            image: PIL Image
            bboxes: List of [x, y, width, height]
            
        Returns:
            List of (text_type, confidence) tuples
        """
        features_list = [self.extract_features_from_bbox(image, bbox) for bbox in bboxes]
        features_scaled = self.scaler.transform(features_list)
        
        pred_ids = self.model.predict(features_scaled)
        pred_probas = self.model.predict_proba(features_scaled)
        
        results = []
        for pred_id, pred_proba in zip(pred_ids, pred_probas):
            text_type = self.id_to_label[pred_id]
            confidence = pred_proba[pred_id]
            results.append((text_type, confidence))
        
        return results