#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
import colorsys
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse


class Fixer:
    def __init__(self, h_th=50, v_th=30, min_conf=0.3):
        self.ht = h_th
        self.vt = v_th
        self.mc = min_conf
    
    def convert(self, poly: List[List[int]]) -> Tuple[int, int, int, int]:
        xs = []
        ys = []
        for p in poly:
            xs.append(p[0])
            ys.append(p[1])
        x = min(xs)
        y = min(ys)
        w = max(xs) - x
        h = max(ys) - y
        return x, y, w, h
    
    def check_near(self, b1: Dict, b2: Dict) -> bool:
        x1, y1, w1, h1 = b1['xywh']
        x2, y2, w2, h2 = b2['xywh']
        
        # v-overlap
        v_ov = not (y1 + h1 < y2 - self.vt or 
                               y2 + h2 < y1 - self.vt)
        
        # h-dist
        dist = min(abs(x1 + w1 - x2), abs(x2 + w2 - x1))
        h_ok = dist < self.ht
        
        return v_ov and h_ok
    
    def do_merge(self, grp: List[Dict]) -> Dict:
        xs = []
        ys = []
        for b in grp:
            x, y, w, h = b['xywh']
            xs.append(x)
            xs.append(x + w)
            ys.append(y)
            ys.append(y + h)
        
        x = min(xs)
        y = min(ys)
        w = max(xs) - x
        h = max(ys) - y
        
        sc = sum(b['score'] for b in grp) / len(grp)
        txt = ' '.join(b['text'] for b in grp)
        
        return {
            'xywh': (x, y, w, h),
            'text': txt,
            'score': sc,
            'merged_count': len(grp)
        }
    
    def norm(self, d):
        if isinstance(d, dict) and 'boxes' in d:
            bl = d['boxes']
            ret = []
            for i in bl:
                ret.append({
                    'bbox': i.get('box', i.get('bbox')),
                    'text': i['text'],
                    'score': i.get('confidence', i.get('score', 0))
                })
            return ret
        elif isinstance(d, list):
            ret = []
            for i in d:
                ret.append({
                    'bbox': i.get('bbox', i.get('box')),
                    'text': i['text'],
                    'score': i.get('score', i.get('confidence', 0))
                })
            return ret
        else:
            return []
    
    def run(self, data) -> List[Dict]:
        data = self.norm(data)
        
        tmp = []
        for it in data:
            if it.get('score', 0) >= self.mc:
                r = self.convert(it['bbox'])
                tmp.append({
                    'xywh': r,
                    'text': it['text'],
                    'score': it['score'],
                    'merged_count': 1
                })
        
        # sort
        tmp.sort(key=lambda b: b['xywh'][1])
        
        final = []
        skip = set()
        
        for i in range(len(tmp)):
            if i in skip:
                continue
            
            bx = tmp[i]
            grp = [bx]
            
            for j in range(i+1, len(tmp)):
                if j in skip:
                    continue
                if self.check_near(bx, tmp[j]):
                    grp.append(tmp[j])
                    skip.add(j)
            
            skip.add(i)
            
            if len(grp) > 1:
                final.append(self.do_merge(grp))
            else:
                final.append(bx)
        
        return final


class LayoutMaker:
    def __init__(self, w=1024, h=768):
        self.w = w
        self.h = h
    
    def guess(self, b: Dict, idx: int, tot: int) -> str:
        x, y, w, h = b['xywh']
        sz = w * h
        
        yr = y / self.h
        xr = x / self.w
        wr = w / self.w
        
        # rules
        if yr < 0.25 and wr > 0.5:
            return "title"
        
        if yr > 0.85:
            return "footer"
        
        if sz > (self.w * self.h * 0.1):
            return "body"
        
        if wr > 0.6:
            return "subtitle"
        
        if sz < (self.w * self.h * 0.02):
            return "caption"
        
        return "body"
    
    def make(self, boxes: List[Dict]) -> Dict:
        els = []
        
        for i, b in enumerate(boxes):
            x, y, w, h = b['xywh']
            
            k = self.guess(b, i, len(boxes))
            
            zo = 1
            if k == "title": zo = 3
            elif k == "subtitle": zo = 2
            
            els.append({
                "class": k,
                "bbox_xywh": [int(x), int(y), int(w), int(h)],
                "z_order": zo,
                "reading_order": i + 1
            })
        
        return {
            "canvas_size": [self.w, self.h],
            "elements": els
        }


class MapGen:
    @staticmethod
    def create(lay: Dict) -> Tuple[np.ndarray, np.ndarray]:
        w, h = lay['canvas_size']
        m = np.zeros((h, w, 3), dtype=np.uint8)
        s = np.ones((h, w), dtype=np.uint8) * 255
        
        cols = {
            "title": (255, 0, 0),
            "subtitle": (255, 128, 0),
            "body": (0, 255, 0),
            "caption": (0, 128, 255),
            "footer": (128, 0, 255)
        }
        
        for e in lay['elements']:
            x, y, w, h = e['bbox_xywh']
            k = e['class']
            c = cols.get(k, (255, 255, 255))
            
            m[y:y+h, x:x+w] = c
            s[y:y+h, x:x+w] = 0
        
        return m, s


class Colors:
    @staticmethod
    def get(p: Path, n=5) -> Dict:
        im = Image.open(p).convert('RGB')
        arr = np.array(im)
        
        px = arr.reshape(-1, 3)
        
        # subsample
        lim = min(len(px), 10000)
        idx = np.random.choice(len(px), lim, replace=False)
        sub = px[idx]
        
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        km.fit(sub)
        
        ctrs = km.cluster_centers_.astype(int)
        lbs = km.predict(sub)
        
        unq, cnts = np.unique(lbs, return_counts=True)
        d = dict(zip(unq, cnts))
        
        srt = sorted(d.items(), key=lambda x: x[1], reverse=True)
        
        p_idx = srt[0][0]
        pri = ctrs[p_idx]
        
        if len(srt) > 1:
            s_idx = srt[1][0]
        else:
            s_idx = p_idx
        sec = ctrs[s_idx]
        
        # accent
        a_idx = p_idx
        mx_sat = 0
        for i, c in enumerate(ctrs):
            r, g, b = c / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            if s > mx_sat:
                mx_sat = s
                a_idx = i
        acc = ctrs[a_idx]
        
        def hx(rgb):
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        
        return {
            "primary": hx(pri),
            "secondary": hx(sec),
            "accent": hx(acc),
            "bg_style": "auto"
        }


def doit(odir, idir, outdir, lim=None):
    jd = outdir / "json"
    md = outdir / "masks"
    pd = outdir / "palettes"
    
    # setup
    jd.mkdir(parents=True, exist_ok=True)
    md.mkdir(parents=True, exist_ok=True)
    pd.mkdir(parents=True, exist_ok=True)
    
    fix = Fixer()
    mak = LayoutMaker()
    
    fls = sorted(odir.glob("sample_*.ocr.json"))
    if lim:
        fls = fls[:lim]
    
    print(f"Processing {len(fls)} slides...")
    
    pf = pd / "palettes.jsonl"
    pw = open(pf, 'w')
    
    res = []
    
    for f in tqdm(fls, desc="Processing slides"):
        sid = f.stem.replace('.ocr', '')
        
        fp = open(f, 'r')
        dat = json.load(fp)
        fp.close()
        
        # step 1
        m_box = fix.run(dat)
        
        if len(m_box) == 0:
            print(f"Warning: No valid boxes for {sid}")
            continue
        
        # step 2
        lay = mak.make(m_box)
        
        lp = jd / f"{sid}.layout.json"
        fp2 = open(lp, 'w')
        json.dump(lay, fp2, indent=2)
        fp2.close()
        
        # step 3
        mp, sf = MapGen.create(lay)
        
        cp = md / f"{sid}.control.png"
        sp = md / f"{sid}.safe.png"
        
        Image.fromarray(mp).save(cp)
        Image.fromarray(sf).save(sp)
        
        # step 4
        ip = idir / f"{sid}.png"
        if ip.exists():
            pal = Colors.get(ip)
            pal['id'] = sid
            pw.write(json.dumps(pal) + '\n')
        else:
            pal = None
            print(f"Warning: Image not found for {sid}")
        
        # step 5
        entry = {
            "id": sid,
            "image_path": f"processed/images/{sid}.png",
            "layout_json": f"processed/json/{sid}.layout.json",
            "safe_zone_path": f"processed/masks/{sid}.safe.png",
            "control_map_path": f"processed/masks/{sid}.control.png",
            "palette": pal,
            "tags": ["real"],
            "width": 1024,
            "height": 768
        }
        res.append(entry)
    
    pw.close()
    
    # step 6
    mp = outdir / "index.jsonl"
    f3 = open(mp, 'w')
    for e in res:
        f3.write(json.dumps(e) + '\n')
    f3.close()
    
    print(f"\nProcessed {len(res)} slides")
    print(f"Manifest: {mp}")
    print(f"Palettes: {pf}")
    
    return res


def make_splits(data, d, tr=0.8, vr=0.1):
    sd = d / "splits"
    sd.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    idx = np.random.permutation(len(data))
    
    nt = int(len(idx) * tr)
    nv = int(len(idx) * vr)
    
    ti = idx[:nt]
    vi = idx[nt:nt + nv]
    tei = idx[nt + nv:]
    
    # write
    f1 = open(sd / "train.txt", 'w')
    for i in ti:
        f1.write(data[i]['id'] + '\n')
    f1.close()
    
    f2 = open(sd / "val.txt", 'w')
    for i in vi:
        f2.write(data[i]['id'] + '\n')
    f2.close()
    
    f3 = open(sd / "test.txt", 'w')
    for i in tei:
        f3.write(data[i]['id'] + '\n')
    f3.close()
    
    print(f"\nSplits created:")
    print(f"  Train: {len(ti)} samples")
    print(f"  Val: {len(vi)} samples")
    print(f"  Test: {len(tei)} samples")


def main():
    p = argparse.ArgumentParser(description="slide dataset from OCR")
    p.add_argument("--ocr-dir", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    
    a = p.parse_args()
    
    # run
    res = doit(
        odir=a.ocr_dir,
        idir=a.images_dir,
        outdir=a.output_dir,
        lim=a.limit
    )
    
    # split
    make_splits(res, a.output_dir)
    

if __name__ == "__main__":
    main()
