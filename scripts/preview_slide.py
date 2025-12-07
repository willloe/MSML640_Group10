from pathlib import Path
from typing import Dict, List
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int = 32):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def render_slide_with_text(
    bg_path: str | Path,
    layout: Dict,
    out_path: str | Path,
    title_text: str = "Quarterly Overview",
    body_lines: List[str] | None = None,
    panel_alpha: int = 180,
):
    if body_lines is None:
        body_lines = [
            "• Key highlights for this quarter",
            "• Performance vs. last quarter",
            "• Upcoming priorities"
        ]

    bg_path = Path(bg_path)
    out_path = Path(out_path)

    img = Image.open(bg_path).convert("RGB")
    W, H = img.size

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_text = ImageDraw.Draw(overlay)

    title_font = _load_font(size=128)
    body_font  = _load_font(size=64)

    elements = layout.get("elements", [])

    for i, el in enumerate(elements):
        x, y, w, h = el["bbox_xywh"]
        x0, y0 = int(x), int(y)
        x1, y1 = int(x + w), int(y + h)

        draw_overlay.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=int(min(w, h) * 0.08),
            fill=(255, 255, 255, panel_alpha),
        )

    title_el = None
    body_el = None
    for el in elements:
        cls = el.get("class")
        if cls == "title" and title_el is None:
            title_el = el
        elif cls == "body" and body_el is None:
            body_el = el

    # Draw title text
    if title_el is not None:
        tx, ty, tw, th = title_el["bbox_xywh"]
        tx0, ty0 = int(tx), int(ty)

        pad_x = int(tw * 0.05)
        pad_y = int(th * 0.15)
        text_pos = (tx0 + pad_x, ty0 + pad_y)

        draw_text.text(
            text_pos,
            title_text,
            font=title_font,
            fill=(30, 30, 30, 255),
        )

    if body_el is not None:
        bx, by, bw, bh = body_el["bbox_xywh"]
        bx0, by0 = int(bx), int(by)
        pad_x = int(bw * 0.05)
        pad_y = int(bh * 0.10)

        x_cursor = bx0 + pad_x
        y_cursor = by0 + pad_y

        for line in body_lines:
            draw_text.text(
                (x_cursor, y_cursor),
                line,
                font=body_font,
                fill=(40, 40, 40, 255),
            )
            y_cursor += int(body_font.size * 1.4)

    composite = Image.alpha_composite(img.convert("RGBA"), overlay)
    composite.save(out_path)
    print("Slide with text saved to:", out_path)
