import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
from pathlib import Path
from io import BytesIO
import shutil

basedir = Path(".")
logopath = basedir / "source_data" / "logos"
outdir = basedir / "dataset_logos_images"

W = 1280
H = 720
num_slides = 2500
max_ovlp = 0.10

classnames = {0: "logo", 1: "images"}

titles_list = [
    "Q4 Results Overview", "Marketing Strategy 2024", "Product Roadmap",
    "Team Performance Updates", "Budget Analysis Report", "Key Metrics Dashboard",
    "Executive Summary", "Quarterly Business Review", "Growth Targets FY24",
]

bullets = [
    "• Increased revenue by 15%", "• New customer acquisition up 23%", 
    "• User engagement improved 40%", "• Reduced churn rate to 2.1%",
    "• Expanded to 12 new markets", "• Launched 3 major features",
]


def make_bg(width, height):
    color_opts = [
        ((240, 240, 245), (200, 210, 230)),
        ((255, 250, 245), (245, 230, 220)),
        ((245, 245, 250), (230, 230, 245)),
        ((45, 50, 60), (65, 75, 95)),
        ((255, 255, 255), (248, 248, 252)),
    ]
    col1, col2 = random.choice(color_opts)
    im = Image.new('RGB', (width, height))
    for yy in range(height):
        rr = int(col1[0] + (col2[0] - col1[0]) * yy / height)
        gg = int(col1[1] + (col2[1] - col1[1]) * yy / height)
        bb = int(col1[2] + (col2[2] - col1[2]) * yy / height)
        for xx in range(width):
            im.putpixel((xx, yy), (rr, gg, bb))
    return im


def get_overlap(b1, b2):
    xx1, yy1, ww1, hh1 = b1
    xx2, yy2, ww2, hh2 = b2
    
    ix1 = max(xx1, xx2)
    iy1 = max(yy1, yy2)
    ix2 = min(xx1 + ww1, xx2 + ww2)
    iy2 = min(yy1 + hh1, yy2 + hh2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    smaller = min(ww1 * hh1, ww2 * hh2)
    if smaller > 0:
        return intersection / smaller
    return 0


def get_pos(boxes, objw, objh, marg=40):
    # placement
    attempts = 0
    while attempts < 100:
        xx = random.randint(marg, max(marg, W - objw - marg))
        yy = random.randint(marg + 40, max(marg + 40, H - objh - marg))
        newbox = (xx, yy, objw, objh)
        
        worst = 0
        for bx in boxes:
            ov = get_overlap(newbox, bx)
            if ov > worst:
                worst = ov
        
        if worst <= max_ovlp:
            return xx, yy
        attempts += 1
    return None, None


def resize_img(imgpath, maxscale=0.35, minsize=80):
    img = Image.open(imgpath).convert('RGBA')
    sc = random.uniform(0.15, maxscale)
    maxw = int(W * sc)
    maxh = int(H * sc)
    rat = min(maxw / img.width, maxh / img.height)
    neww = max(minsize, int(img.width * rat))
    newh = max(minsize, int(img.height * rat))
    resized = img.resize((neww, newh), Image.Resampling.LANCZOS)
    return resized, neww, newh


def bbox_to_obb(x, y, w, h):
    # convert
    x1 = x / W
    y1 = y / H
    x2 = (x + w) / W
    y2 = y / H
    x3 = (x + w) / W
    y3 = (y + h) / H
    x4 = x / W
    y4 = (y + h) / H
    return f"{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"


def make_slide(n, splitname, all_logos):
    random.seed(n * 1000)
    
    bg = make_bg(W, H)
    drawer = ImageDraw.Draw(bg)
    
    # title
    pix = bg.getpixel((100, 100))
    bright = (pix[0] + pix[1] + pix[2]) / 3
    if bright > 128:
        txtcol = (50, 50, 50)
    else:
        txtcol = (240, 240, 240)
    drawer.text((40, 20), random.choice(titles_list), fill=txtcol)
    
    placed = []
    lbls = []
    
    # objects
    howmany = random.randint(2, 6)
    i = 0
    while i < howmany:
        lpath = random.choice(all_logos)
        try:
            limg, ww, hh = resize_img(lpath)
            px, py = get_pos(placed, ww, hh)
            if px is not None:
                bg.paste(limg.convert('RGB'), (px, py))
                placed.append((px, py, ww, hh))
                cls = random.choice([0, 1])
                lbls.append((cls, px, py, ww, hh))
        except:
            pass
        i = i + 1
    
    # save
    fname = f"slide_{splitname}_{n:05d}"
    bg.save(outdir / "images" / splitname / f"{fname}.png")
    
    # labels
    f = open(outdir / "labels" / splitname / f"{fname}.txt", "w")
    for c, xx, yy, ww, hh in lbls:
        f.write(f"{c} {bbox_to_obb(xx, yy, ww, hh)}\n")
    f.close()
    
    return len(lbls)


def run():
    print("="*50)
    print("Generating Logos + Images Dataset")
    print("="*50)
    
    # setup
    if outdir.exists():
        shutil.rmtree(outdir)
    
    for sp in ["train", "val", "test"]:
        (outdir / "images" / sp).mkdir(parents=True)
        (outdir / "labels" / sp).mkdir(parents=True)
    
    # load
    logos = []
    for ext in ["*.jpg", "*.png"]:
        logos = logos + list(logopath.glob(ext))
    print(f"Found {len(logos)} source images")
    
    # splits
    train_count = int(num_slides * 0.7)
    val_count = int(num_slides * 0.15)
    test_count = num_slides - train_count - val_count
    
    splits = [
        ("train", train_count, 0), 
        ("val", val_count, train_count), 
        ("test", test_count, train_count + val_count)
    ]
    
    # generate
    for sname, cnt, startidx in splits:
        print(f"\nGenerating {sname}: {cnt} slides...")
        for j in range(cnt):
            make_slide(startidx + j, sname, logos)
            if (j + 1) % 100 == 0:
                print(f"  {j+1}/{cnt}")
        print(f"  {sname}: {cnt} slides")
    
    # yaml
    yamlcontent = """path: .
        train: images/train
        val: images/val
        test: images/test

        names:
        0: logo
        1: images

        nc: 2
    """
    with open(outdir / "dataset.yaml", "w") as yf:
        yf.write(yamlcontent)
    
    print(f"\nDone! saved to {outdir}/")


if __name__ == "__main__":
    run()
