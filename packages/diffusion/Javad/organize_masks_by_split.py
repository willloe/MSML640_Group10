#!/usr/bin/env python3

import shutil
from pathlib import Path
from tqdm import tqdm

def go(d, s):
    t = d / "train"
    te = d / "test"
    v = d / "val"
    
    # dirs
    t.mkdir(exist_ok=True)
    te.mkdir(exist_ok=True)
    v.mkdir(exist_ok=True)
    
    m = {
        'train': t,
        'test': te,
        'val': v
    }
    
    for k, dest in m.items():
        f_path = s / f"{k}.txt"
        
        print(f"\nProcessing {k} split...")
        
        lines = []
        fp = open(f_path, 'r')
        for x in fp:
            if x.strip():
                lines.append(x.strip())
        fp.close()
        
        # loop
        for i in tqdm(lines, desc=f"Moving {k} files"):
            src1 = d / f"{i}.control.png"
            dst1 = dest / f"{i}.control.png"
            
            if src1.exists():
                shutil.move(str(src1), str(dst1))
            
            src2 = d / f"{i}.safe.png"
            dst2 = dest / f"{i}.safe.png"
            
            if src2.exists():
                shutil.move(str(src2), str(dst2))
        
        print(f"  Moved {len(lines) * 2} files to {dest}")
    
    print("\n✅ Organization complete!")
    print(f"\nDirectory structure:")
    # stats
    l1 = list(t.glob('*.png'))
    print(f"  {t}: {len(l1)} files")
    l2 = list(v.glob('*.png'))
    print(f"  {v}: {len(l2)} files")
    l3 = list(te.glob('*.png'))
    print(f"  {te}: {len(l3)} files")


if __name__ == "__main__":
    p1 = Path("/home/jbaghiro/MSML640/Dataset/data_controlnet/processed/masks")
    p2 = Path("/home/jbaghiro/MSML640/Dataset/data_controlnet/processed/splits")
    
    go(p1, p2)
