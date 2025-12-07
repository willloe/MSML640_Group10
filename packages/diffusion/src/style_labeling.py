import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


STYLE_PRESETS = {
    "default": (
        "single minimal professional presentation slide background, "
        "soft gradients, large clean content area, low contrast, "
        "no text, no logos, no collage, no grid"
    ),
    "academic": (
        "clean academic presentation slide background, white or light canvas, "
        "subtle sectioning for title and bullet points, low contrast, "
        "no photos, no heavy textures, no collage"
    ),
    "noisy": (
        "highly textured abstract geometric background, layered shapes and patterns, "
        "vibrant but not neon colors, noticeable noise and detail, "
        "complex visual structure, no text or logos"
    ),
    "gradient": (
        "smooth multi-tone gradient presentation background, soft transitions, "
        "no hard shapes, no grid, no collage, very low texture, "
        "large empty regions for overlaying text"
    ),
    "photo": (
        "photographic background with shallow depth of field, "
        "soft bokeh and light leaks, muted colors, "
        "subject off-center, large negative space"
    ),
}

def build_style_manifest(
    images_dir: Path,
    out_jsonl: Path,
    overwrite: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Path:
    images_dir = Path(images_dir)
    out_jsonl = Path(out_jsonl)

    if out_jsonl.exists() and not overwrite:
        print(f"[style_labeling] Using existing manifest: {out_jsonl}")
        return out_jsonl

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print("[style_labeling] Loading CLIP...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    preset_names = list(STYLE_PRESETS.keys())
    preset_prompts = [STYLE_PRESETS[k] for k in preset_names]

    img_paths = sorted(
        [p for p in images_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    print(f"[style_labeling] Found {len(img_paths)} images in {images_dir}")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, img_path in enumerate(img_paths, start=1):
            img = Image.open(img_path).convert("RGB")

            inputs = processor(
                text=preset_prompts,
                images=img,
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image[0]  # (num_presets,)
                best_idx = int(torch.argmax(logits_per_image).item())

            best_caption = preset_prompts[best_idx]
            best_name = preset_names[best_idx]

            f.write(
                json.dumps(
                    {
                        "image": str(img_path),
                        "style": best_name,
                        "caption": best_caption,
                    }
                )
                + "\n"
            )

            if idx % 50 == 0 or idx == len(img_paths):
                print(
                    f"[style_labeling] Labeled {idx}/{len(img_paths)} "
                    f"({best_name} for {img_path.name})"
                )

    print(f"[style_labeling] Wrote manifest: {out_jsonl}")
    return out_jsonl