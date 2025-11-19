## A. Direct Usage of the Custom Model for this Project

#### The `trained_model.pt` model can be directly used to make inference of an image for this project:
```
model   = YOLO("trained_model.pt")
results = model.predict("image.png") # see inference_pipeline.py for more details
```

#### The `results.json` provides structural information about the positions and bounding boxes of an object.

## B. Generate Synthetic Dataset + Train Model

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Generate a synthetic dataset
```
python generate_dataset.py
```

### 3. Train the custom YOLO model
```
python train_yolo_model.py
```

### 4. Run the inference script
```
python inference_pipeline.py
```