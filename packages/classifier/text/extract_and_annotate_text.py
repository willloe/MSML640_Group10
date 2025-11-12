import json
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib
from tqdm import tqdm
import random

# Include text-related classes
TEXT_CLASSES = {
    'Title', 'Caption', 'Obj-text', 'Other-text', 'Page-text'
}


COLORS = {
    'Text': '#FF0000',
    'Title': '#00FF00', 
    'Paragraph': '#0000FF',
    'TextBox': '#FFFF00',
    'Caption': '#FF00FF',
    'Label': '#00FFFF',
    'Heading': '#FFA500',
    'PageNumber': '#800080'
}

def parse_jsonl(fp):
    stuff = []
    with open(fp, 'r') as f:
        for ln in f:
            if ln.strip():
                stuff.append(json.loads(ln))
    return stuff

def filter_slides_with_text_classes(rec):
    res = []
    urls = rec['image_urls']
    
    for idx, boxes in rec['bboxes']:
        filt = [b for b in boxes if b['class'] in TEXT_CLASSES]
        if filt:
            if 0 <= idx - 1 < len(urls):
                res.append((idx, urls[idx - 1], filt))
    
    return res

def download_image(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    from io import BytesIO
    return Image.open(BytesIO(r.content))

def draw_bboxes(img, boxes):
    d = ImageDraw.Draw(img)
    fnt = ImageFont.load_default()

    for b in boxes:
        cls = b['class']
        x, y, w, h = b['bbox']
        col = COLORS.get(cls, '#FFFFFF')
        
        d.rectangle([x, y, x + w, y + h], outline=col, width=3)
        
        lbl = f"{cls}"
        bb = d.textbbox((x, y), lbl, font=fnt)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        d.rectangle([x, y - th - 4, x + tw + 4, y], fill=col)
        d.text((x + 2, y - th - 2), lbl, fill='white', font=fnt)
    
    return img

def create_filename(dn, si, url):
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    safe = dn.replace('/', '_').replace('\\', '_')
    return f"{safe}_slide{si:03d}_{h}.jpg"

def process_dataset_sample(ann_dir, out_dir, max_imgs=None, seed=42):
    random.seed(seed)

    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / 'annotated').mkdir(exist_ok=True)
    (out_dir / 'raw_images').mkdir(exist_ok=True)

    files = list(ann_dir.glob('*.jsonl'))
    print(f"Found {len(files)} JSONL files: {[f.name for f in files]}")
    
    all_slides = []

    for jf in files:
        print(f"Processing {jf.name}...")
        recs = parse_jsonl(jf)
        
        for r in recs:
            slides = filter_slides_with_text_classes(r)
            for si, url, boxes in slides:
                all_slides.append({
                    'deck_name': r['deck_name'],
                    'slide_idx': si,
                    'image_url': url,
                    'bboxes': boxes,
                    'category': r.get('category', 'unknown')
                })

    print(f"\nTotal slides with text annotations: {len(all_slides)}")

    if max_imgs and len(all_slides) > max_imgs:
        all_slides = random.sample(all_slides, max_imgs)
        print(f"Randomly sampled {max_imgs} slides")

    ok = 0
    fail = 0
    meta_path = out_dir / 'sample_metadata.json'
    # Load existing metadata if present
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        existing_filenames = set(m['filename'] for m in meta)
    else:
        meta = []
        existing_filenames = set()

    for sd in tqdm(all_slides):
        fn = create_filename(sd['deck_name'], sd['slide_idx'], sd['image_url'])
        if fn in existing_filenames:
            continue  # Skip if already processed

        try:
            raw_img_path = out_dir / 'raw_images' / fn
            ann_img_path = out_dir / 'annotated' / fn

            # Download only if not present
            if not raw_img_path.exists():
                img = download_image(sd['image_url'])
                img.save(raw_img_path)
            else:
                img = Image.open(raw_img_path)

            # Annotate only if not present
            if not ann_img_path.exists():
                ann = draw_bboxes(img.copy(), sd['bboxes'])
                ann.save(ann_img_path)

            meta.append({
                'filename': fn,
                'deck_name': sd['deck_name'],
                'slide_idx': sd['slide_idx'],
                'category': sd['category'],
                'image_url': sd['image_url'],
                'bboxes': sd['bboxes'],
                'num_bboxes': len(sd['bboxes'])
            })
            ok += 1

            # Save metadata incrementally after each slide
            if ok % 1 == 0:
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)

        except Exception as e:
            print(f"\nFailed to process {sd['image_url']}: {e}")
            fail += 1

    with open(out_dir / 'sample_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print('='*60)
    print(f"✓ Successfully downloaded: {ok} images")
    print(f"✗ Failed downloads: {fail} images")
    print(f"\nOutput saved to:")
    print(f"  - Raw images: {out_dir / 'raw_images'}")
    print(f"  - Annotated images: {out_dir / 'annotated'}")
    print(f"  - Metadata: {out_dir / 'sample_metadata.json'}")
    print('='*60)

def main():
    p = Path(__file__).parent
    ann_dir = p / 'temp_slidevqa' / 'annotations' / 'bbox'
    out_dir = p / 'sample_data'  
    
    # Processing only 50 images for quick testing
    process_dataset_sample(ann_dir, out_dir, seed=42)

if __name__ == '__main__':
    main()