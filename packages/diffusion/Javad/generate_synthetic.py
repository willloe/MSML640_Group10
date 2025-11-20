import os
import random
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple
import hashlib

# colors
PAL = {
    "title": (255, 0, 0),
    "subtitle": (255, 128, 0),
    "body": (0, 255, 0),
    "caption": (0, 128, 255),
    "footer": (128, 0, 255)
}

DIM = (1024, 768)

class Gen:
    def __init__(self, w=1024, h=768):
        self.w = w
        self.h = h
        self.pad = 50
    
    def make(self) -> Dict:
        # types
        opts = [
            "single_column", "two_column", "three_column",
            "title_only", "title_subtitle", "complex_grid",
            "sidebar_left", "sidebar_right", "hero_image_layout",
            "split_horizontal"
        ]
        kind = random.choice(opts)
        
        items = []
        
        do_ti = random.random() > 0.05
        cy = self.pad
        
        # title
        if do_ti:
            th = random.randint(60, 120)
            tw = random.randint(int(self.w * 0.4), int(self.w * 0.9))
            if random.random() > 0.5:
                tx = (self.w - tw) // 2
            else:
                tx = self.pad
                
            items.append({
                "class": "title",
                "bbox_xywh": [tx, cy, tw, th]
            })
            cy += th + random.randint(10, 30)

        # sub
        if random.random() > 0.3 and cy < self.h - 100:
            sh = random.randint(40, 80)
            sw = random.randint(int(self.w * 0.3), int(self.w * 0.8))
            sx = self.pad if random.random() > 0.5 else (self.w - sw) // 2
            
            items.append({
                "class": "subtitle",
                "bbox_xywh": [sx, cy, sw, sh]
            })
            cy += sh + random.randint(20, 40)

        rem = self.h - cy - self.pad
        
        if kind == "title_only":
            pass
            
        elif kind == "single_column":
            if rem > 100:
                bw = self.w - 2 * self.pad
                n = random.randint(1, 3)
                ph = (rem - (n-1)*20) // n
                for _ in range(n):
                    items.append({
                        "class": "body",
                        "bbox_xywh": [self.pad, cy, bw, ph]
                    })
                    cy += ph + 20

        elif kind == "two_column":
            if rem > 100:
                gap = 40
                cw = (self.w - 2 * self.pad - gap) // 2
                
                items.append({
                    "class": "body",
                    "bbox_xywh": [self.pad, cy, cw, rem]
                })
                
                items.append({
                    "class": "body",
                    "bbox_xywh": [self.pad + cw + gap, cy, cw, rem]
                })

        elif kind == "three_column":
            if rem > 100:
                gap = 30
                cw = (self.w - 2 * self.pad - 2 * gap) // 3
                
                for i in range(3):
                    items.append({
                        "class": "body",
                        "bbox_xywh": [self.pad + i * (cw + gap), cy, cw, rem]
                    })

        elif kind == "sidebar_left":
            if rem > 100:
                sw = int((self.w - 2 * self.pad) * 0.3)
                mw = self.w - 2 * self.pad - sw - 40
                
                items.append({
                    "class": "body",
                    "bbox_xywh": [self.pad, cy, sw, rem]
                })
                
                items.append({
                    "class": "body",
                    "bbox_xywh": [self.pad + sw + 40, cy, mw, rem]
                })

        elif kind == "sidebar_right":
            if rem > 100:
                sw = int((self.w - 2 * self.pad) * 0.3)
                mw = self.w - 2 * self.pad - sw - 40
                
                items.append({
                    "class": "body",
                    "bbox_xywh": [self.pad, cy, mw, rem]
                })
                
                items.append({
                    "class": "body",
                    "bbox_xywh": [self.pad + mw + 40, cy, sw, rem]
                })

        elif kind == "split_horizontal":
            if rem > 200:
                th = rem // 2 - 20
                bh = rem // 2 - 20
                
                items.append({
                    "class": "body",
                    "bbox_xywh": [self.pad, cy, self.w - 2 * self.pad, th]
                })
                
                items.append({
                    "class": "body",
                    "bbox_xywh": [self.pad, cy + th + 40, self.w - 2 * self.pad, bh]
                })

        elif kind == "hero_image_layout":
            if rem > 100:
                hh = int(rem * 0.7)
                items.append({
                    "class": "body",
                    "bbox_xywh": [self.pad, cy, self.w - 2 * self.pad, hh]
                })
                
                th = rem - hh - 30
                if th > 30:
                    items.append({
                        "class": "body",
                        "bbox_xywh": [self.pad, cy + hh + 30, self.w - 2 * self.pad, th]
                    })

        elif kind == "complex_grid":
            if rem > 100:
                nb = random.randint(2, 6)
                for _ in range(nb):
                    w = random.randint(200, 400)
                    h = random.randint(100, 300)
                    x = random.randint(self.pad, self.w - self.pad - w)
                    y = random.randint(cy, self.h - self.pad - h)
                    
                    bad = False
                    for el in items:
                        ex, ey, ew, eh = el['bbox_xywh']
                        # check
                        if (x < ex + ew and x + w > ex and
                            y < ey + eh and y + h > ey):
                            bad = True
                            break
                    
                    if not bad:
                        items.append({
                            "class": "body",
                            "bbox_xywh": [x, y, w, h]
                        })

        # xtra
        if random.random() > 0.5:
            cw = random.randint(100, 300)
            ch = 30
            cx = self.w - self.pad - cw
            cy_cap = self.h - self.pad - ch
            items.append({
                "class": "caption",
                "bbox_xywh": [cx, cy_cap, cw, ch]
            })
            
        if random.random() > 0.7:
            fw = random.randint(200, 500)
            fh = 20
            fx = (self.w - fw) // 2
            fy = self.h - 30
            items.append({
                "class": "footer",
                "bbox_xywh": [fx, fy, fw, fh]
            })

        return {
            "canvas_size": [self.w, self.h],
            "elements": items
        }

def render(l) -> Tuple[np.ndarray, np.ndarray]:
    w, h = l['canvas_size']
    res = np.zeros((h, w, 3), dtype=np.uint8)
    safe = np.ones((h, w), dtype=np.uint8) * 255
    
    for it in l['elements']:
        x, y, w1, h1 = it['bbox_xywh']
        cls = it['class']
        c = PAL.get(cls, (255, 255, 255))
        
        # clip
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        w1 = max(0, min(w1, w - x))
        h1 = max(0, min(h1, h - y))
        
        if w1 > 0 and h1 > 0:
            res[y:y+h1, x:x+w1] = c
            safe[y:y+h1, x:x+w1] = 0
    
    return res, safe

def run():
    p = argparse.ArgumentParser(description="Generate synthetic ControlNet data")
    p.add_argument("--output_dir", type=str, default="Dataset/data_controlnet/synthetic_data", help="Output directory")
    p.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    
    g = Gen(DIM[0], DIM[1])
    
    print(f"Generating {a.num_samples} samples in {a.output_dir}...")
    
    seen = set()
    
    bar = tqdm(total=a.num_samples)
    cnt = 0
    
    while cnt < a.num_samples:
        lay = g.make()
        mp, sf = render(lay)
        
        # hash
        h_val = hashlib.md5(mp.tobytes()).hexdigest()
        
        if h_val in seen:
            continue
            
        seen.add(h_val)
        
        sid = f"synth_{cnt:06d}"
        
        p1 = os.path.join(a.output_dir, f"{sid}.control.png")
        p2 = os.path.join(a.output_dir, f"{sid}.safe.png")
        
        Image.fromarray(mp).save(p1)
        Image.fromarray(sf).save(p2)
        
        cnt += 1
        bar.update(1)

    bar.close()
    print("Done!")

if __name__ == "__main__":
    run()
