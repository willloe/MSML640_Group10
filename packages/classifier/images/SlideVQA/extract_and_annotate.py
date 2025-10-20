import json
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib
from tqdm import tqdm

CLASSES = {'Image', 'Figure', 'Diagram'}
COLORS = {'Image': '#FF0000', 'Figure': '#00FF00', 'Diagram': '#0000FF'}

def parse_jsonl(fp):
    stuff = []
    with open(fp, 'r') as f:
        for ln in f:
            if ln.strip():
                stuff.append(json.loads(ln))
    return stuff

def filter_slides_with_target_classes(rec):
    res = []
    urls = rec['image_urls']
    for idx, boxes in rec['bboxes']:
        filt = [b for b in boxes if b['class'] in CLASSES]
        if filt:
            if 0 <= idx - 1 < len(urls):
                res.append((idx, urls[idx - 1], filt))
    return res

def download_image(url):
    r = requests.get(url, timeout=30)
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

def process_dataset(ann_dir, out_dir, max_imgs=None):
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'annotated').mkdir(exist_ok=True)
    (out_dir / 'original').mkdir(exist_ok=True)

    files = list(ann_dir.glob('*.jsonl'))
    all_s = []

    for jf in files:
        recs = parse_jsonl(jf)
        for r in recs:
            slides = filter_slides_with_target_classes(r)
            for si, url, boxes in slides:
                all_s.append({
                    'deck_name': r['deck_name'],
                    'slide_idx': si,
                    'image_url': url,
                    'bboxes': boxes,
                    'category': r.get('category', 'unknown')
                })

    if max_imgs and max_imgs < len(all_s):
        all_s = all_s[:max_imgs]

    ok = 0
    fail = 0
    meta = []

    for sd in tqdm(all_s):
        try:
            img = download_image(sd['image_url'])
            fn = create_filename(sd['deck_name'], sd['slide_idx'], sd['image_url'])
            img.save(out_dir / 'original' / fn)
            ann = draw_bboxes(img.copy(), sd['bboxes'])
            ann.save(out_dir / 'annotated' / fn)
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
        except:
            fail += 1

    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"{fail} failed")

def main():
    p = Path(__file__).parent
    ann = p / 'temp_slidevqa' / 'annotations' / 'bbox'
    out = p / 'full_dataset'
    process_dataset(ann, out, max_imgs=None)

if __name__ == '__main__':
    main()
