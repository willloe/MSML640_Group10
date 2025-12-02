import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from PIL import Image
import cv2
import random
import numpy as np
import torch

from synthetic import _smooth_regions_inplace

@dataclass
class ImageSample:
    path: Path
    caption: str

def _read_sidecar_caption(img: Path) -> Optional[str]:
    txt = img.with_suffix(".txt")
    if txt.exists():
        return txt.read_text().strip()
    js = img.with_suffix(".json")
    if js.exists():
        try:
            data = json.loads(js.read_text())
            for k in ("caption", "prompt", "text"):
                if k in data and isinstance(data[k], str):
                    return data[k]
        except Exception:
            pass
    return None

def default_style_caption() -> str:
    return "professional slide background, soft low-frequency texture, high readability, minimal clutter"

def build_manifest(images_dir: Path, out_path: Path, fallback_caption: Optional[str] = None) -> List[ImageSample]:
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(sorted(images_dir.glob(ext)))
    samples: List[ImageSample] = []
    for img in images:
        cap = _read_sidecar_caption(img)
        if not cap:
            cap = fallback_caption or default_style_caption()
        samples.append(ImageSample(path=img, caption=cap))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            rec = {"image": str(s.path), "caption": s.caption}
            f.write(json.dumps(rec) + "\n")
    return samples

class AbstractWithLayoutDataset(torch.utils.data.Dataset):
    def __init__(self, bg_dir: Path, layout_dir: Path, resolution: int = 512, alpha: float = 0.6):
        self.bg_paths = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg") for p in bg_dir.glob(ext)])
        self.mask_paths = sorted(list(layout_dir.glob("*.safe.png")))
        if not self.bg_paths:
            raise RuntimeError(f"No images in {bg_dir}")
        if not self.mask_paths:
            raise RuntimeError(f"No .safe.png masks in {layout_dir}")
        self.resolution = resolution
        self.alpha = alpha

    def __len__(self):
        return max(len(self.bg_paths), len(self.mask_paths))

    def __getitem__(self, idx):
        bg_path = self.bg_paths[idx % len(self.bg_paths)]
        mask_path = random.choice(self.mask_paths)

        bg = Image.open(bg_path).convert("RGB").resize((self.resolution, self.resolution), Image.BICUBIC)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        if mask.ndim == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif mask.ndim == 3 and mask.shape[2] == 4:
            mask = mask[:, :, :3]

        mask = cv2.resize(mask, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)

        boxes = parse_layout_mask(mask)

        arr = np.array(bg).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr)

        _smooth_regions_inplace(tensor, boxes, alpha=self.alpha)
        return {"pixel_values": tensor, "caption": "abstract layout background"}


def parse_layout_mask(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    if mask.ndim == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()

    _, mask_bin = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 200:
            continue
        boxes.append((x, y, w, h))

    return boxes