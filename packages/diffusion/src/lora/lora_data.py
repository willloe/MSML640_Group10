from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
