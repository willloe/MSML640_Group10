import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from extract_and_annotate import parse_jsonl, filter_slides_with_target_classes, create_filename

def generate_metadata_for_full_dataset():
    p = Path(__file__).parent
    ann_dir = p / 'temp_slidevqa' / 'annotations' / 'bbox'
    ds_dir = p / 'full_dataset'
    ann = ds_dir / 'annotated'

    if not ann_dir.exists() or not ann.exists():
        return

    imgs = set(f.name for f in ann.iterdir() if f.is_file() and f.suffix == '.jpg')
    files = list(ann_dir.glob('*.jsonl'))

    meta = []
    cnt = 0

    for jf in files:
        recs = parse_jsonl(jf)
        for r in recs:
            slides = filter_slides_with_target_classes(r)
            for si, url, boxes in slides:
                fn = create_filename(r['deck_name'], si, url)
                if fn in imgs:
                    meta.append({
                        'filename': fn,
                        'deck_name': r['deck_name'],
                        'slide_idx': si,
                        'category': r.get('category', 'unknown'),
                        'image_url': url,
                        'bboxes': boxes,
                        'num_bboxes': len(boxes)
                    })
                    cnt += 1

    with open(ds_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"{cnt} images")

if __name__ == '__main__':
    generate_metadata_for_full_dataset()
