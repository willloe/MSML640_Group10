import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

colors = {
    0: "blue",
    1: "green", 
    2: "red",
    3: "orange",
    4: "purple",
}


def drawbox(draw, coords, col, lbl, imgw, imgh):
    # bbox
    pts = []
    pts.append((coords[0] * imgw, coords[1] * imgh))
    pts.append((coords[2] * imgw, coords[3] * imgh))
    pts.append((coords[4] * imgw, coords[5] * imgh))
    pts.append((coords[6] * imgw, coords[7] * imgh))
    draw.polygon(pts, outline=col, width=3)
    draw.text((pts[0][0], pts[0][1] - 18), lbl, fill=col)


def preview(dspath, nsamples=5, outdir=None):
    dspath = Path(dspath)
    
    if outdir == None:
        savedir = Path(f"{dspath.name}_preview")
    else:
        savedir = Path(outdir)
    savedir.mkdir(exist_ok=True)
    
    # classes
    yamlfile = dspath / "dataset.yaml"
    clsnames = {}
    if yamlfile.exists():
        f = open(yamlfile)
        lines = f.readlines()
        f.close()
        for ln in lines:
            ln = ln.strip()
            if ln.startswith("0:") or ln.startswith("1:") or ln.startswith("2:"):
                parts = ln.split(":")
                cid = int(parts[0].strip())
                cname = parts[1].strip()
                clsnames[cid] = cname
    
    print(f"Classes: {clsnames}")
    
    # images
    imgdir = dspath / "images" / "train"
    imgs = list(imgdir.glob("*.png"))
    if len(imgs) == 0:
        imgs = list(imgdir.glob("*.jpg"))
    
    random.seed(42)
    picks = random.sample(imgs, min(nsamples, len(imgs)))
    
    for i in range(len(picks)):
        imgpath = picks[i]
        im = Image.open(imgpath).convert("RGB")
        dr = ImageDraw.Draw(im)
        w = im.size[0]
        h = im.size[1]
        
        # label
        lblpath = dspath / "labels" / "train" / f"{imgpath.stem}.txt"
        
        if lblpath.exists():
            lf = open(lblpath)
            for line in lf:
                parts = line.strip().split()
                classid = int(parts[0])
                coords = []
                for p in parts[1:9]:
                    coords.append(float(p))
                
                c = colors.get(classid, "white")
                name = clsnames.get(classid, f"class_{classid}")
                drawbox(dr, coords, c, name, w, h)
            lf.close()
        
        outpath = savedir / f"sample_{i+1}.png"
        im.save(outpath)
        print(f"Saved: {outpath}")
    
    print(f"\nDone! previews in {savedir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    preview(args.dataset, args.samples, args.output)


if __name__ == "__main__":
    main()
