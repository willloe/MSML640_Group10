## Anirud's Text Classifier

### Overview
The text classifier identifies and categorizes different types of text elements in presentation slides using a Random Forest machine learning model. The system extracts text using EasyOCR and classifies it into five categories based on visual and positional features.

### Text Categories
1. **Title**: Main slide headings
2. **Caption**: Image/diagram captions
3. **Obj-text**: Text within objects (charts, diagrams)
4. **Other-text**: Body text, paragraphs
5. **Page-text**: Page numbers, footer text

### Dataset
- **Source**: [SlideVQA Dataset](https://github.com/nttmdlab-nlp/SlideVQA)
- **Format**: JSONL annotation files with bounding box coordinates and text labels
- **Training Data**: Extracted from annotated presentation slides
- **Annotation Structure**: Each text region includes position, dimensions, and category label

### Pipeline Architecture
1. **Data Collection**: Download and annotate slides from SlideVQA dataset
2. **Feature Engineering**: Extract 19 visual and positional features from text bounding boxes
3. **Model Training**: Train Random Forest classifier with 200 estimators
4. **Text Extraction**: Use EasyOCR to detect text regions
5. **Classification**: Predict text type for each detected region
6. **Visualization**: Generate annotated output images with classified text

### Feature Set (19 Features)
The classifier uses the following features:

**Position Features (4):**
- `relative_y`: Vertical position (0-1)
- `relative_x`: Horizontal position (0-1)
- `relative_center_x`: Horizontal center position
- `relative_center_y`: Vertical center position

**Size Features (4):**
- `relative_width`: Width relative to slide
- `relative_height`: Height relative to slide
- `relative_area`: Area coverage
- `aspect_ratio`: Width/height ratio

**Visual Features (4):**
- `mean_brightness`: Average pixel brightness
- `std_brightness`: Brightness variation
- `edge_density`: Edge detection metric
- `text_density`: Dark pixel density

**Location Flags (7):**
- `is_top_third`: In upper third of slide
- `is_middle_third`: In middle third
- `is_bottom_third`: In lower third
- `is_horizontally_centered`: Near horizontal center
- `is_large`: Large area coverage (>15%)
- `is_wide`: Wide element (>60% width)
- `is_tall`: Tall element (>30% height)

---

## Setup Instructions

### Prerequisites
```bash
# Create and activate virtual environment (if not already done)
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows
```

### Dependencies
```bash
# Install required packages
pip install easyocr pillow opencv-python scikit-learn numpy matplotlib seaborn tqdm requests joblib
```

**Required Package Versions:**
- Python >= 3.8
- easyocr >= 1.7.0
- opencv-python >= 4.8.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- pillow >= 10.0.0

### Data Setup
```bash
# Navigate to the text folder
cd packages/classifier/text

# Create directory for annotations
mkdir -p temp_slidevqa/annotations/bbox

# Download all JSONL annotation files from:
# https://github.com/nttmdlab-nlp/SlideVQA/tree/main/annotations/bbox
# Place them in: temp_slidevqa/annotations/bbox/
```

---

## Usage Instructions

### Step 1: Data Collection and Annotation
```bash
# This script downloads slide images and creates annotated versions with bounding boxes
python extract_and_annotate_text.py
```

**Output:**
- Downloaded slide images in `sample_data/raw_images/`
- Annotated images with bounding boxes in `sample_data/annotated/`
- Metadata JSON file with annotation details

### Step 2: Feature Extraction
```bash
# Extract features from annotated data and create train/test split
python prepare_training_data.py
```

**Output:**
- `training_data/X_train.npy` - Training features
- `training_data/X_test.npy` - Test features
- `training_data/y_train.npy` - Training labels
- `training_data/y_test.npy` - Test labels
- `training_data/label_mapping.json` - Label encoding

### Step 3: Model Training
```bash
# Train Random Forest model and generate evaluation metrics
python train_classifier.py
```

**Output:**
- `trained_models/classifier_model.pkl` - Trained Random Forest model
- `trained_models/scaler.pkl` - Feature scaler
- `plots/confusion_matrix.png` - Confusion matrix visualization
- `plots/feature_importance.png` - Feature importance chart
- Console output with accuracy, precision, recall, F1-score

### Step 4: Text Extraction and Classification
```bash
# Run OCR and classification on slide images
python text_extractor.py
```

**Output Format:**
```json
{
  "filename": "slide_001.jpg",
  "num_texts": 5,
  "extracted_texts": [
    {
      "text": "Introduction to Machine Learning",
      "bbox": [120, 45, 580, 72],
      "confidence": 0.987,
      "type": "Title",
      "type_confidence": 0.923
    }
  ]
}
```

### Step 5: Visualization
```bash
# Create visual overlays of classified text regions
python visualize_output.py
```

**Output:**
- Annotated images with color-coded bounding boxes
- Each text type displayed in different colors
- Confidence scores overlaid on boxes

---

## Project Structure
```
packages/classifier/text/
├── extract_and_annotate_text.py   # Step 1: Download & annotate data
├── prepare_training_data.py       # Step 2: Feature engineering
├── train_classifier.py            # Step 3: Model training
├── text_extractor.py              # Step 4: OCR + classification
├── text_type_predictor.py         # Classifier inference class
├── visualize_output.py            # Step 5: Result visualization
├── sample_data/                   # Downloaded slides
│   ├── raw_images/
│   ├── annotated/
│   └── sample_metadata.json
├── training_data/                 # Prepared features
│   ├── X_train.npy, X_test.npy
│   ├── y_train.npy, y_test.npy
│   └── label_mapping.json
├── trained_models/                # Saved models
│   ├── classifier_model.pkl
│   └── scaler.pkl
├── output_with_types/             # Extraction results
├── visualizations/                # Visual outputs
└── plots/                         # Training metrics
```

## Troubleshooting

### Common Issues

**Issue: EasyOCR fails to initialize**
```bash
# Solution: Install with GPU support or use CPU mode
pip install easyocr torch torchvision
```

**Issue: Out of memory during OCR**
```python
# Solution: Process images in smaller batches or reduce image size
# In text_extractor.py, resize images before processing
```

**Issue: Low classification accuracy**
```bash
# Solution: Retrain with more data or adjust hyperparameters
# Edit train_classifier.py: n_estimators, max_depth, min_samples_split
```

**Issue: Missing annotation files**
```bash
# Solution: Manually download from SlideVQA repository
# Ensure all JSONL files are in temp_slidevqa/annotations/bbox/
```

---

## Configuration

### Model Hyperparameters
Located in `train_classifier.py`:
- **n_estimators**: 200 (number of trees)
- **max_depth**: None (unlimited)
- **min_samples_split**: 2
- **min_samples_leaf**: 1
- **random_state**: 42

### OCR Settings
Located in `text_extractor.py`:
- **Languages**: ['en'] (English)
- **GPU**: True/False (auto-detected)
- **Confidence threshold**: 0.1 (minimum OCR confidence)

### Feature Engineering
Located in `prepare_training_data.py`:
- Adjustable thresholds for location flags
- Normalization strategy (StandardScaler)

---

## Future Improvements

1. **Deep Learning Integration**
   - Explore CNN-based text classifiers
   - Fine-tune vision transformers (ViT)

2. **Multi-language Support**
   - Extend EasyOCR to support more languages
   - Language-specific feature engineering

3. **Advanced Features**
   - Font style detection
   - Color-based features
   - Proximity to other elements

4. **Model Ensemble**
   - Combine Random Forest with Gradient Boosting
   - Implement voting classifier


---

## References

- [SlideVQA Dataset](https://github.com/nttmdlab-nlp/SlideVQA)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)

---

## License

This project is part of MSML640 coursework.

## Contact

For questions or issues related to the text classifier:
- **Anirud Mohan** - Text Classification Component
- **Project Repository**: [MSML640_Group10](https://github.com/willloe/MSML640_Group10)