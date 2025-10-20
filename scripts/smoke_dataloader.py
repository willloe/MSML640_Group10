# Run: python scripts/smoke_dataloader.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
sys.path.append(str(SRC))

import torch
from data import get_dataloaders

def main():
    cfg = {
        "source": "fake",
        "image_size": 64,
        "batch_size": 8,
        "num_workers": 0,
        "val_ratio": 0.25,
        "subset": 32,
    }
    train_loader, val_loader = get_dataloaders(cfg)

    xb, yb = next(iter(train_loader))
    print("train batch shape:", tuple(xb.shape))
    print("labels shape:", tuple(yb.shape))

    # iterate through the loader once to ensure no crashes
    count = 0
    for xb, _ in train_loader:
        assert xb.dtype == torch.float32
        count += xb.size(0)
    print("samples iterated:", count)


if __name__ == "__main__":
    main()
