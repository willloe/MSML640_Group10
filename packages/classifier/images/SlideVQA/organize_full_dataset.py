import json
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def organize_full_dataset_by_class():
    p = Path(__file__).parent
    ds = p / 'full_dataset'
    meta_p = ds / 'metadata.json'
    ann = ds / 'annotated'
    out = p / 'by_class'

    if not meta_p.exists():
        return

    with open(meta_p, 'r') as f:
        meta = json.load(f)

    dirs = {
        'Image': out / 'Image',
        'Figure': out / 'Figure',
        'Diagram': out / 'Diagram'
    }

    for nm, d in dirs.items():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    skip = 0

    for item in tqdm(meta):
        fn = item['filename']
        src = ann / fn

        if not src.exists():
            skip += 1
            continue

        cls = set(b['class'] for b in item['bboxes'])

        for c in cls:
            if c in dirs:
                dst = dirs[c] / fn
                shutil.copy2(src, dst)
                stats[c] += 1

    print(f"{len(meta)} images")

if __name__ == '__main__':
    organize_full_dataset_by_class()
