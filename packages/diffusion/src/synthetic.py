import random
from typing import Dict, Tuple, List
from pathlib import Path
import torch
from generate import create_layout_control_map, visualize_control_map
from validation import validate_layout, validate_palette

IMAGE_PROBABILITY = 0.7
LOGO_PROBABILITY = 0.4

def _rand_hex():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"#{r:02X}{g:02X}{b:02X}"


def random_palette(seed: int | None = None) -> Dict[str, str]:
    if seed is not None:
        random.seed(int(seed))

    primary = _rand_hex()
    secondary = _rand_hex()
    accent = _rand_hex()
    styles = ["minimalist gradient", "soft blur", "subtle texture", "solid fill"]
    style = random.choice(styles)

    return {"primary": primary, "secondary": secondary, "accent": accent, "bg_style": style}


def _clip_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x, W - 1))
    y0 = max(0, min(y, H - 1))
    x1 = max(0, min(x0 + max(0, w), W))
    y1 = max(0, min(y0 + max(0, h), H))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def random_layout(canvas_w: int, canvas_h: int, n_boxes: int = 3, seed: int | None = None) -> Dict:
    if seed is not None:
        random.seed(int(seed))

    elements: List[Dict] = []

    # Title
    title_w = int(canvas_w * random.uniform(0.5, 0.8))
    title_h = int(canvas_h * random.uniform(0.06, 0.12))
    title_x = int((canvas_w - title_w) * random.uniform(0.1, 0.2))
    title_y = int(canvas_h * random.uniform(0.05, 0.12))
    x, y, w, h = _clip_box(title_x, title_y, title_w, title_h, canvas_w, canvas_h)
    if w > 0 and h > 0:
        elements.append({
            "class": "title",
            "bbox_xywh": [x, y, w, h],
            "z_order": 2,
            "reading_order": 1,
        })

    # Body
    body_w = int(canvas_w * random.uniform(0.6, 0.85))
    body_h = int(canvas_h * random.uniform(0.35, 0.55))
    body_x = int((canvas_w - body_w) * random.uniform(0.1, 0.2))
    body_y = int(canvas_h * random.uniform(0.2, 0.35))
    x, y, w, h = _clip_box(body_x, body_y, body_w, body_h, canvas_w, canvas_h)
    if w > 0 and h > 0:
        elements.append({
            "class": "body",
            "bbox_xywh": [x, y, w, h],
            "z_order": 1,
            "reading_order": 2,
        })

    # Image
    if n_boxes >= 3 and random.random() < IMAGE_PROBABILITY:
        img_w = int(canvas_w * random.uniform(0.18, 0.3))
        img_h = int(canvas_h * random.uniform(0.18, 0.3))
        corners = [
            (int(canvas_w * 0.05), int(canvas_h * 0.65)),
            (int(canvas_w * 0.70), int(canvas_h * 0.65)),
            (int(canvas_w * 0.70), int(canvas_h * 0.15)),
        ]
        img_x, img_y = random.choice(corners)
        x, y, w, h = _clip_box(img_x, img_y, img_w, img_h, canvas_w, canvas_h)
        if w > 0 and h > 0:
            elements.append({
                "class": "image",
                "bbox_xywh": [x, y, w, h],
                "z_order": 0,
                "reading_order": 3,
            })

    if n_boxes >= 4 and random.random() < LOGO_PROBABILITY:
        logo_w = int(canvas_w * random.uniform(0.08, 0.12))
        logo_h = int(canvas_h * random.uniform(0.05, 0.08))
        logo_x = int(canvas_w * random.uniform(0.82, 0.9))
        logo_y = int(canvas_h * random.uniform(0.85, 0.92))
        x, y, w, h = _clip_box(logo_x, logo_y, logo_w, logo_h, canvas_w, canvas_h)
        if w > 0 and h > 0:
            elements.append({
                "class": "logo",
                "bbox_xywh": [x, y, w, h],
                "z_order": 3,
                "reading_order": 4,
            })

    layout = {
        "canvas_size": [int(canvas_h), int(canvas_w)],
        "elements": elements,
    }
    return layout


def make_control_from_layout(layout: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    control_map, safe_zone = create_layout_control_map(layout)
    return control_map, safe_zone


def _prompt_for_layout(palette: Dict[str, str]) -> str:
    base = "professional slide background"
    p = palette.get("primary", "")
    s = palette.get("secondary", "")
    a = palette.get("accent", "")
    style = palette.get("bg_style", "")
    parts = [base]
    if p:
        parts.append(f"primary color {p}")
    if s:
        parts.append(f"secondary color {s}")
    if a:
        parts.append(f"accent color {a}")
    if style:
        parts.append(style)
    return ", ".join(parts)


def sample_condition_batch(
    n: int,
    canvas_size: Tuple[int, int] = (1024, 768),
    seed: int | None = None,
):
    if seed is not None:
        random.seed(int(seed))

    H, W = int(canvas_size[0]), int(canvas_size[1])
    samples = []
    for _ in range(int(n)):
        pal = random_palette()
        ok_p, errs_p = validate_palette(pal)
        if not ok_p:
            raise ValueError(f"Generated palette failed schema: {errs_p}")

        layout = random_layout(W, H, n_boxes=3 + random.randint(0, 2))
        ok_l, errs_l = validate_layout(layout)
        if not ok_l:
            raise ValueError(f"Generated layout failed schema: {errs_l}")

        control_map, safe_zone = make_control_from_layout(layout)
        prompt = _prompt_for_layout(pal)

        samples.append({
            "prompt": prompt,
            "palette": pal,
            "layout": layout,
            "control_map": control_map,
            "safe_zone": safe_zone,
        })
    return samples


def save_control_visuals(samples: List[Dict], out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(samples):
        vis = visualize_control_map(s["control_map"])
        vis.save(str(out_path / f"control_map_{i:02d}.png"))
