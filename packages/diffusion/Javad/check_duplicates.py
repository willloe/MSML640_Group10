import os
import hashlib
import glob
from collections import defaultdict

def calc(p):
    f = open(p, "rb")
    d = f.read()
    f.close()
    return hashlib.md5(d).hexdigest()

def work(d):
    # list
    lst = glob.glob(os.path.join(d, "*.control.png"))
    print(f"Found {len(lst)} control files.")
    
    m = defaultdict(list)
    for x in lst:
        h = calc(x)
        m[h].append(x)
    
    dups = {}
    for k, v in m.items():
        if len(v) > 1:
            dups[k] = v
    
    if not dups:
        print("All control images are unique.")
    else:
        print(f"Found {len(dups)} sets of duplicates.")
        t = 0
        for h, fls in dups.items():
            print(f"Hash {h}: {len(fls)} files")
            t += len(fls) - 1
        print(f"Total duplicate images: {t}")

if __name__ == "__main__":
    work("Dataset/data_controlnet/synthetic_data")
