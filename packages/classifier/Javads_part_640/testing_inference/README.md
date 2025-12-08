# Slide Visual Element Detection Pipeline

Detect and classify visual elements in slides and diagrams using YOLO models with text extraction.

## Environment Setup

```bash
conda create -n finalproject python=3.10
conda activate finalproject
pip install ultralytics opencv-python pillow numpy tqdm easyocr scikit-learn joblib
```

---

## Running Inference

### Slides Pipeline
Detects charts, images, logos, and text in presentation slides.

```bash
python run_inference.py --pipeline slides --conf 0.25
```

**Models used:**
- `graphs/best.pt` (charts & shapes)
- `images_logos/best.pt` (images & logos)
- Text extraction (OCR + classifier)

### Diagrams Pipeline
Detects charts, diagram nodes, and text in technical diagrams.

```bash
python run_inference.py --pipeline diagrams --conf 0.25
```

**Models used:**
- `graphs/best.pt` (charts & shapes)
- `nodes/best.pt` (diagram nodes)
- Text extraction (OCR + classifier)

### Bonus Task 5 Output

```bash
# Vision-only detection
python run_inference.py --pipeline slides --bonus-5

# Multimodal detection (vision + text)
python run_inference.py --pipeline slides --text --bonus-5
```

Output structure:
```
Bonus_task-5_output/
├── vision_only/
│   └── results.json
└── multimodal/
    └── results.json
```

---

## Detected Elements

**Charts & Shapes** (23 classes): column, line, pie, bar, area, scatter, histogram, waterfall, flowchart, hierarchy, shapes, tables

**Images & Logos** (2 classes): logo, images

**Diagram Nodes** (1 class): diagram_node

**Text** (5 types): Title, Caption, Obj-text, Other-text, Page-text

---

## Output Format

### JSON Output (`results.json`)

```json
{
  "DETECTION_STATISTICS": {
    "mode": "multimodal",
    "total_slides_processed": 6,
    "total_detections": 117,
    "breakdown": {
      "graphs": 18,
      "images_logos": 15,
      "text": 84
    }
  }
}
```
