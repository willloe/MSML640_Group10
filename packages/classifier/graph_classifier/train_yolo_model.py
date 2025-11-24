from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # load model
    # yolov8s is better than yolov8n for this task
    # yolo11m is even better but requires more GPU memory, only for HPC
    model = YOLO("yolov8s.pt")

    # train model
    model.train\
    (
        data="chart_data.yaml",
        epochs=30,                                            # number of training epochs (default: 30)
        imgsz=768,                                            # input image size (default: 640 pixels)
        batch=8,                                              # batch size (default: 8, GPU memory dependent)
        workers=2,                                            # number of data loading workers
        device="0" if torch.cuda.is_available() else "cpu",   # use GPU if available
        lr0=0.003,                                            # initial learning rate
        optimizer="Adam",                                     # optimizer type
        patience=20,                                          # early stopping patience
        task="detect"                                         # detection task
    )

    # save the trained model
    model_path = "trained_model.pt"
    model.save(model_path)
    print(f"Model saved to {model_path}")
