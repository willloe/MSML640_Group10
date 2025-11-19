# Custom YOLO Classifier for Charts, Tables, Graphs, and Shapes

## A. Direct Usage of the Custom Model for this Project

#### The `trained_model.pt` model can be directly used to make inference of an image for this project:
```
model   = YOLO("trained_model.pt")
results = model.predict("image.png") # see inference_pipeline.py for more details
```

#### The `results.json` provides structural information about the positions and bounding boxes of an object.

## B. Run Inference Using Pre-trained Custom Model

#### This is the least time-consuming option.

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run the inference script
```
python inference_pipeline.py
```

#### Note:
- Annotated results are saved to: `runs/detect/predict*/`
- Structured detections are saved to: `results.json`
- Modify `image_path` variable to test different images
- Ensure `trained_model.pt` exists in the folder

## C. Train your own custom YOLO model

#### This option can be time-consuming depending on hardware.

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Unzip the dataset into the dataset folder
```
unzip dataset.zip
```

#### Note:
- The folder must contain:
    - `dataset/`
    - `chart_data.yaml`
    - `yolov8s.pt`, will automatically download if not exist

### 3. Train the custom YOLO model
```
python train_yolo_model.py
```

#### Note:
- Training duration depends on:
    - number of epochs (default: 30 epochs)
    - GPU speed, can increase `batch` to 16 or 32 if your GPU has ≥ 8GB VRAM

### 4. Run the inference script
```
python inference_pipeline.py
```

#### Note:
- Annotated results are saved to: `runs/detect/predict*/`
- Structured detections are saved to: `results.json`
- Modify `image_path` variable to test different images

## D. Generate a Synthetic Dataset + Train Model

#### This is the most time-consuming option.

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Generate a synthetic dataset
```
python generate_dataset.py
```

#### This will create the following directory structure:
```
dataset/
  ├── images/
  │     ├── train/
  │     └── val/
  └── labels/
        ├── train/
        └── val/
```

### 3. Train the custom YOLO model
```
python train_yolo_model.py
```

#### Note:
- Training duration depends on:
    - number of epochs (default: 30 epochs)
    - GPU speed, can increase `batch` to 16 or 32 if your GPU has ≥ 8GB VRAM

### 4. Run the inference script
```
python inference_pipeline.py
```

#### Note:
- Annotated results are saved to: `runs/detect/predict*/`
- Structured detections are saved to: `results.json`
- Modify `image_path` variable to test different images
