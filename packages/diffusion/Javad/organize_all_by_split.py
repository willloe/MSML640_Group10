#!/usr/bin/env python3

import shutil
from pathlib import Path
from tqdm import tqdm

def sort_stuff(b, s):
    imd = b / "images"
    jsd = b / "json"
    
    # dirs
    for n in ['train', 'test', 'val']:
        p1 = imd / n
        p1.mkdir(exist_ok=True)
        p2 = jsd / n
        p2.mkdir(exist_ok=True)
    
    lookup = {
        'train': {'i': imd / 'train', 'j': jsd / 'train'},
        'test': {'i': imd / 'test', 'j': jsd / 'test'},
        'val': {'i': imd / 'val', 'j': jsd / 'val'}
    }
    
    for k, v in lookup.items():
        f_txt = s / f"{k}.txt"
        
        print(f"\nProcessing {k} split...")
        
        lst = []
        fh = open(f_txt, 'r')
        for ln in fh:
            if len(ln.strip()) > 0:
                lst.append(ln.strip())
        fh.close()
        
        # move
        for i in tqdm(lst, desc=f"Moving {k} files"):
            s1 = imd / f"{i}.png"
            d1 = v['i'] / f"{i}.png"
            
            if s1.exists():
                shutil.move(str(s1), str(d1))
            
            s2 = jsd / f"{i}.layout.json"
            d2 = v['j'] / f"{i}.layout.json"
            
            if s2.exists():
                shutil.move(str(s2), str(d2))
        
        c1 = len(list(v['i'].glob('*.png')))
        c2 = len(list(v['j'].glob('*.json')))
        print(f"  Moved {c1} images and {c2} JSON files")
    
    print("\n✅ Organization complete!")
    
    print(f"\nFinal structure:")
    # info
    for n in ['train', 'val', 'test']:
        c_img = len(list((imd / n).glob('*.png')))
        c_json = len(list((jsd / n).glob('*.json')))
        print(f"\n  {n}/")
        print(f"    images: {c_img} files")
        print(f"    json: {c_json} files")


if __name__ == "__main__":
    d1 = Path("/home/jbaghiro/MSML640/Dataset/data_controlnet/processed")
    d2 = Path("/home/jbaghiro/MSML640/Dataset/data_controlnet/processed/splits")
    
    sort_stuff(d1, d2)
