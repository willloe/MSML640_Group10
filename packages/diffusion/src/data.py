import torch
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

class BackgroundsFolderDataset(Dataset):

    def __init__(self, folder_path, manifest_path=None, transform=None):

        self.folder_path = Path(folder_path)
        self.transform = transform
        self.images = []
        self.tags_map = {}

        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    for entry in manifest.get('images', []):
                        self.tags_map[entry['path']] = entry.get('tags', [])

        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
        for ext in image_extensions:
            self.images.extend(self.folder_path.glob(f'*{ext}'))
            self.images.extend(self.folder_path.glob(f'*{ext.upper()}'))

        self.images = sorted(self.images)

        if len(self.images) == 0:
            raise ValueError(f"No images found in {folder_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        tags = self.tags_map.get(img_path.name, [])

        return image, tags


def _build_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((int(image_size), int(image_size))),
        transforms.ToTensor(),
    ])

def _make_fake_dataset(image_size, subset):
    tr = _build_transforms(image_size)
    size = int(subset) if subset is not None else 512
    return datasets.FakeData(
        size=size,
        image_size=(3, int(image_size), int(image_size)),
        transform=tr
    )

def _make_dtd_dataset(data_root, image_size, download):
    tr = _build_transforms(image_size)
    # DTD exposes "train" / "val" / "test" but the splits are folds.
    # For now, load "train" and do a random split.
    return datasets.DTD(
        root=str(data_root),
        split="train",
        download=bool(download),
        transform=tr,
    )

def _make_backgrounds_dataset(folder_path, manifest_path, image_size):
    tr = _build_transforms(image_size)
    return BackgroundsFolderDataset(
        folder_path=folder_path,
        manifest_path=manifest_path,
        transform=tr,
    )

def _split_dataset(ds, val_ratio, seed=42):
    n = len(ds)
    v = max(1, int(float(val_ratio) * n))
    t = n - v
    gen = torch.Generator().manual_seed(int(seed))
    return random_split(ds, [t, v], generator=gen)

def get_dataloaders(config):
    source = config.get("source", "fake")
    data_root = config.get("data_root", "./data")
    image_size = config.get("image_size", 256)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 2)
    val_ratio = config.get("val_ratio", 0.1)
    download = config.get("download", True)
    subset = config.get("subset", None)

    if source == "fake":
        base = _make_fake_dataset(image_size, subset)
    elif source == "dtd":
        base = _make_dtd_dataset(data_root, image_size, download)
        if subset is not None:
            k = min(int(subset), len(base))
            indices = list(range(k))
            base = torch.utils.data.Subset(base, indices)
    elif source == "backgrounds":
        folder_path = config.get("folder_path", "./data/backgrounds")
        manifest_path = config.get("manifest_path", None)
        base = _make_backgrounds_dataset(folder_path, manifest_path, image_size)
        if subset is not None:
            k = min(int(subset), len(base))
            indices = list(range(k))
            base = torch.utils.data.Subset(base, indices)
    else:
        raise ValueError(f"Unsupported dataset source: {source}")

    train_ds, val_ds = _split_dataset(base, val_ratio)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
    )
    return train_loader, val_loader