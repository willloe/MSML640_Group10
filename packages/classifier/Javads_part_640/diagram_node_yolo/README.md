# Synthetic Diagram Generation

Generate synthetic training data for diagram node detection using YOLO.

## Environment Setup

```bash
conda create -n finalproject python=3.10
conda activate finalproject
pip install ultralytics opencv-python pillow numpy tqdm easyocr scikit-learn joblib
```

## Usage

```bash
cd diagram_node_yolo/
python generate_synthetic_diagrams.py
```

**Parameters:**
- `--output_path`: Output directory (default: `./synthetic-diagrams`)
- `--num_images`: Number of diagrams to generate (default: 1000)
- `--img_width`: Image width in pixels (default: 800)
- `--img_height`: Image height in pixels (default: 600)
- `--min_shapes`: Minimum shapes per diagram (default: 3)
- `--max_shapes`: Maximum shapes per diagram (default: 12)
- `--seed`: Random seed for reproducibility (optional)

## Output Structure

```
synthetic-diagrams/
├── images/
│   ├── synthetic_00000.png
│   ├── synthetic_00001.png
│   └── ...
├── labels/
│   ├── synthetic_00000.txt
│   ├── synthetic_00001.txt
│   └── ...
└── dataset.yaml
```

## Generated Classes

- **Class 0**: diagram_node (rectangles, circles, diamonds, etc.)
- **Class 1**: arrow (straight and curved connections)
- **Class 2**: text_label (text boxes)
- **Class 3**: image_region (placeholder icons)
