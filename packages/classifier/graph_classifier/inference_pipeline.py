import json
import os
from ultralytics import YOLO

# load trained yolo model
def load_model(weights_path="best.pt"):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model file not found: {weights_path}")
    print(f"Loading model: {weights_path}")
    return YOLO(weights_path)

# save results to custom json format
def save_json(results, json_path):
    output = []

    for result in results:
        boxes = result.boxes
        names = result.names  # mapping class_id -> class name

        if boxes is None:
            continue

        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            class_name = names[cls_id]

            # pixel bbox (xywh format)
            xywh = box.xywh[0].tolist()
            x_center, y_center, w, h = xywh

            # convert center-based to top-left format
            left = x_center - w / 2
            top  = y_center - h / 2

            # normalized bbox (relative values)
            rel = box.xywhn[0].tolist()

            comp = \
            {
                "id": idx,
                "type": class_name,
                "bbox_emus": [left, top, w, h],
                "bbox_rel": rel,
                "z": idx,             # simple depth ordering
                "group_id": None,     # placeholder for grouping logic if needed
                "debug": {"tag": class_name}
            }
            output.append(comp)

    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved JSON results to: {json_path}")

# run inference and return results objects
def run_inference(model, source):
    print(f"Running inference on: {source}")
    results = model.predict\
    (
        source=source,
        save=True,         # saves annotated image
        save_txt=False,    # disable YOLO txt saving
        conf=0.25,
        iou=0.5,
        verbose=False
    )
    return results

# text summary in terminal
def print_results(results):
    print("\nDetections:")
    for result in results:
        if result.boxes is None:
            print("No detections.")
            continue

        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            xyxy   = box.xyxy[0].tolist()
            label  = names[cls_id]
            print(f" - {label} | conf={conf:.2f} | bbox={xyxy}")

if __name__ == "__main__":
    model_path = "trained_model.pt"
    image_path = "test.png"

    model   = load_model(model_path)
    results = run_inference(model, image_path)
    print_results(results)

    # save to json output
    save_json(results, "results.json")

    print("\nAnnotated image saved inside: runs/detect/predict*/")
