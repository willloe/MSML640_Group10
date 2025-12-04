import random
from pathlib import Path
from PIL import Image, ImageDraw
import shutil

basedir = Path(".")
ai2d_path = basedir / "ai2d" / "images"
logo_path = basedir / "source_data" / "logos"
outdir = basedir / "dataset_diagrams_ai2d"

W = 1280
H = 720
total = 3000

title_options = [
    "System Overview", "Analysis Results", "Technical Diagram", "Research Findings",
    "Process Flow", "Data Structure", "Component Architecture", "Study Results",
    "Methodology", "Framework Design", "Concept Map", "Scientific Model"
]


def gradient(width, height, seedval):
    random.seed(seedval)
    cols = [
        ((245, 248, 252), (220, 230, 245)),
        ((255, 252, 248), (245, 235, 225)),
        ((248, 248, 255), (235, 235, 250)),
        ((45, 50, 60), (65, 75, 95)),
        ((255, 255, 255), (250, 250, 252)),
        ((240, 248, 255), (200, 220, 240)),
    ]
    c1, c2 = random.choice(cols)
    result = Image.new('RGB', (width, height))
    y = 0
    while y < height:
        r = int(c1[0] + (c2[0] - c1[0]) * y / height)
        g = int(c1[1] + (c2[1] - c1[1]) * y / height)
        b = int(c1[2] + (c2[2] - c1[2]) * y / height)
        x = 0
        while x < width:
            result.putpixel((x, y), (r, g, b))
            x += 1
        y += 1
    return result


def check_overlap(box1, box2, gap=15):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    no_overlap = (x1 + w1 + gap < x2) or (x2 + w2 + gap < x1) or (y1 + h1 + gap < y2) or (y2 + h2 + gap < y1)
    return not no_overlap


def findspot(existing, ow, oh, sd, margin=40):
    random.seed(sd)
    tries = 0
    while tries < 100:
        xpos = random.randint(margin, max(margin, W - ow - margin))
        ypos = random.randint(margin + 40, max(margin + 40, H - oh - margin))
        candidate = (xpos, ypos, ow, oh)
        
        ok = True
        for b in existing:
            if check_overlap(candidate, b):
                ok = False
                break
        
        if ok:
            return xpos, ypos
        tries = tries + 1
        sd = sd + 1
    return None, None


def loadimg(pth, maxsc, minsz, sd):
    random.seed(sd)
    im = Image.open(pth).convert('RGBA')
    s = random.uniform(0.15, maxsc)
    mxw = int(W * s)
    mxh = int(H * s)
    r = min(mxw / im.width, mxh / im.height)
    nw = max(minsz, int(im.width * r))
    nh = max(minsz, int(im.height * r))
    return im.resize((nw, nh), Image.Resampling.LANCZOS), nw, nh


def to_obb(x, y, w, h):
    # format
    p1x = x / W
    p1y = y / H
    p2x = (x + w) / W
    p2y = y / H
    p3x = (x + w) / W
    p3y = (y + h) / H
    p4x = x / W
    p4y = (y + h) / H
    return f"{p1x:.6f} {p1y:.6f} {p2x:.6f} {p2y:.6f} {p3x:.6f} {p3y:.6f} {p4x:.6f} {p4y:.6f}"


def createslide(idx, split, diags, lgos):
    random.seed(idx * 1000)
    
    canvas = gradient(W, H, idx)
    dr = ImageDraw.Draw(canvas)
    
    # title
    pixel = canvas.getpixel((100, 100))
    brightness = (pixel[0] + pixel[1] + pixel[2]) / 3
    if brightness > 128:
        tc = (50, 50, 50)
    else:
        tc = (240, 240, 240)
    dr.text((40, 20), random.choice(title_options), fill=tc)
    
    all_boxes = []
    diag_boxes = []
    
    # diagrams
    ndiags = random.randint(1, 2)
    for i in range(ndiags):
        dp = random.choice(diags)
        try:
            dimg, dw, dh = loadimg(dp, 0.5, 150, idx * 100 + i)
            dx, dy = findspot(all_boxes, dw, dh, idx * 100 + i + 50)
            if dx != None:
                if dimg.mode == 'RGBA':
                    canvas.paste(dimg, (dx, dy), dimg)
                else:
                    canvas.paste(dimg, (dx, dy))
                all_boxes.append((dx, dy, dw, dh))
                diag_boxes.append((dx, dy, dw, dh))
        except:
            continue
    
    # distractors
    nlogos = random.randint(2, 5)
    for i in range(nlogos):
        lp = random.choice(lgos)
        try:
            limg, lw, lh = loadimg(lp, 0.25, 60, idx * 100 + i + 200)
            lx, ly = findspot(all_boxes, lw, lh, idx * 100 + i + 250)
            if lx != None:
                canvas.paste(limg.convert('RGB'), (lx, ly))
                all_boxes.append((lx, ly, lw, lh))
        except:
            continue
    
    # save
    outname = f"slide_{split}_{idx:05d}"
    canvas.convert('RGB').save(outdir / "images" / split / f"{outname}.png")
    
    # labels
    labelfile = open(outdir / "labels" / split / f"{outname}.txt", "w")
    for bx, by, bw, bh in diag_boxes:
        labelfile.write(f"0 {to_obb(bx, by, bw, bh)}\n")
    labelfile.close()
    
    return len(diag_boxes)


def main():
    
    # cleanup
    if outdir.exists():
        shutil.rmtree(outdir)
    
    for s in ["train", "val", "test"]:
        (outdir / "images" / s).mkdir(parents=True)
        (outdir / "labels" / s).mkdir(parents=True)
    
    # load
    diagramlist = list(ai2d_path.glob("*.png"))
    logolist = list(logo_path.glob("*.jpg"))
    if len(logolist) > 50000:
        logolist = logolist[:50000]
    print(f"Diagrams: {len(diagramlist)}, Logos: {len(logolist)}")
    
    # split
    train_n = int(total * 0.7)
    val_n = int(total * 0.15)
    test_n = total - train_n - val_n
    
    configs = [
        ("train", train_n, 0), 
        ("val", val_n, train_n), 
        ("test", test_n, train_n + val_n)
    ]
    
    # generate
    for splitname, num, offset in configs:
        print(f"\nGenerating {splitname}: {num} slides...")
        diagcount = 0
        for j in range(num):
            cnt = createslide(offset + j, splitname, diagramlist, logolist)
            diagcount = diagcount + cnt
            if (j + 1) % 100 == 0:
                print(f"  {j+1}/{num}")
        print(f"  {splitname}: {num} slides, {diagcount} diagram labels")
    
    # yaml
    yamltxt = """path: .
        train: images/train
        val: images/val
        test: images/test

        names:
        0: diagram

        nc: 1
    """
    f = open(outdir / "dataset.yaml", "w")
    f.write(yamltxt)
    f.close()
    
    print(f"\nDone! saved to {outdir}/")


if __name__ == "__main__":
    main()
