import json
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

def train_model(model, loader, optimizer, device, scaler=None):
    model.train()

    loss    = 0.0
    correct = 0
    total   = 0

    # training loop
    for images, labels in tqdm(loader, desc="Train Model", leave=False):

        # for CUDA acceleration
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # NOTE: torch.cuda.amp.autocast is deprecated, use torch.amp.autocast
        # forward pass
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            logits = model(images)
            loss   = F.cross_entropy(logits, labels)

        # backward pass and optimization step
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # accumulate stats
        loss    += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += images.size(0)

    return loss / total, correct / total

def evaluate_model(model, loader, device, classes):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluate Model", leave=False):

            # for CUDA acceleration
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            preds   = model(images).argmax(1)
            y_true += labels.cpu().tolist()
            y_pred += preds.cpu().tolist()

    # compute classification report and confusion matrix
    my_classification_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    my_confusion_matrix      = confusion_matrix(y_true, y_pred).tolist()
    accuracy                 = my_classification_report["accuracy"]

    return accuracy, my_classification_report, my_confusion_matrix

if __name__ == "__main__":

    # for CUDA acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    save_path = "results"
    os.makedirs(save_path, exist_ok=True)

    # hyperparameters for tuning
    dataset_path       = Path("by_class")
    batch_size         = 32
    image_size         = 384
    num_of_epochs      = 1          # DEBUG: set to larger number for better results
    model_choice       = "resnet18" # options are "resnet18", "resnet50", "efficientnet_b0"
    learning_rate      = 3e-4
    is_mixed_precision = True

    # data transforms and loaders
    train_transform = transforms.Compose\
    ([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # validation transforms
    valid_transform = transforms.Compose\
    ([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # prepare datasets and dataloaders
    full_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)

    # split into train and validation sets
    dataset_len = len(full_dataset)
    valid_len   = int(0.2 * dataset_len)
    train_len   = dataset_len - valid_len

    # use torch random_split for better reproducibility
    train_dataset, valid_dataset    = torch.utils.data.random_split(full_dataset, [train_len, valid_len])
    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = valid_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # TODO: modify classes to get more types of graphs
    # Current classes are: ['Diagram', 'Figure', 'Image']
    classes = full_dataset.classes
    print("Classes:", classes)
    print("Train Samples:", len(train_dataset), "Validate Samples:", len(valid_dataset))

    # model selection
    if model_choice == "resnet18":
        model    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif model_choice == "resnet50":
        model    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    else:
        model               = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))

    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # NOTE: torch.cuda.amp.GradScaler is deprecated, use torch.amp.GradScaler
    scaler = torch.amp.GradScaler('cuda') if (is_mixed_precision and device.type == "cuda") else None

    best_accuracy = 0.0
    for epoch in range(1, num_of_epochs + 1):
        train_loss, train_accuracy                  = train_model(model, train_loader, optimizer, device, scaler)
        valid_accuracy, result, my_confusion_matrix = evaluate_model(model, valid_loader, device, classes)

        print(f"Epoch {epoch:02d} | train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.3f}, validate_accuracy = {valid_accuracy:.3f}")

        # save best model
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy

            # save model and result
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pt"))

            # TODO: save other info if needed, and see if .json format is preferred. Talk to Javad
            with open(os.path.join(save_path, "result.json"), "w") as f:
                json.dump({"accuracy": valid_accuracy, "result": result, "confusion_matrix": my_confusion_matrix}, f, indent=2)

    print(f"Training complete. Best validate accuracy is: {best_accuracy:.3f}")

# Command line to run file:
# TODO: fixed command line path for saving the model correctly

# run from graph_classifier folder (incorrect, but for reference):
# ..\packages\classifier\graph_classifier>python graph_classifier.py

# run from dataset folder (correct, but model is saved in dataset?)
# ..\packages\classifier\images\SlideVQA>python ../../graph_classifier/graph_classifier.py
