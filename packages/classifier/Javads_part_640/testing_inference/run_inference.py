import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import os

# models
classnames_logos = {0: "logo", 1: "images"}
classnames_nodes = {0: "diagram_node"}

clrs = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
}

def getimgs(dir):
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    imgs = []
    for e in exts:
        imgs.extend(Path(dir).glob(f'*{e}'))
        imgs.extend(Path(dir).glob(f'*{e.upper()}'))
    imgs = sorted(imgs)
    return imgs

def runinfer(modelpath, indir, outdir, confthresh=0.25, savetxt=False):
    # load
    print(f"Loading model: {modelpath}")
    mdl = YOLO(modelpath)
    
    isobb = 'obb' in str(modelpath).lower()
    if hasattr(mdl.model, 'task'):
        if mdl.model.task == 'obb':
            isobb = True
    print(f"Model type: {'OBB' if isobb else 'Detection'}")
    
    # pick classnames
    modelstr = str(modelpath).lower()
    if 'node' in modelstr or 'diagram' in modelstr:
        classnames = classnames_nodes
        print("Using: diagram_node classes")
    else:
        classnames = classnames_logos
        print("Using: logo/images classes")
    
    # images
    inp = Path(indir)
    imglist = getimgs(inp)
    
    if len(imglist) == 0:
        print(f"No images found in {indir}")
        return
    
    print(f"Found {len(imglist)} images")
    
    # output
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)
    
    if savetxt:
        lbldir = outp / "labels"
        lbldir.mkdir(exist_ok=True)
    
    # run
    summary = []
    
    i = 0
    while i < len(imglist):
        imgpath = imglist[i]
        print(f"\n[{i+1}/{len(imglist)}] Processing: {imgpath.name}")
        
        res = mdl(imgpath, conf=confthresh, verbose=False)
        r = res[0]
        
        # detections
        if r.obb != None:
            dets = r.obb
            numdet = len(dets.cls) if dets.cls is not None else 0
        elif r.boxes != None:
            dets = r.boxes
            numdet = len(dets.cls) if dets.cls is not None else 0
        else:
            numdet = 0
            dets = None
        
        if numdet == 0:
            print(f"  No detections")
            summary.append({'image': imgpath.name, 'detections': 0})
            im = cv2.imread(str(imgpath))
            outimgpath = outp / f"detected_{imgpath.name}"
            cv2.imwrite(str(outimgpath), im)
            i = i + 1
            continue
        
        # count
        clscounts = {}
        for cid in dets.cls:
            cid = int(cid)
            cname = classnames.get(cid, f"class_{cid}")
            if cname in clscounts:
                clscounts[cname] = clscounts[cname] + 1
            else:
                clscounts[cname] = 1
        
        print(f"  Detections: {clscounts}")
        summary.append({'image': imgpath.name, 'detections': numdet, 'classes': clscounts})
        
        # draw
        im = cv2.imread(str(imgpath))
        
        if r.obb != None:
            # obb
            polys = dets.xyxyxyxy.cpu().numpy()
            j = 0
            while j < len(polys):
                poly = polys[j]
                cid = int(dets.cls[j])
                conf = float(dets.conf[j])
                col = clrs.get(cid, (255, 255, 255))
                
                pts = poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(im, [pts], isClosed=True, color=col, thickness=3)
                
                # label
                lbl = f"{classnames.get(cid, cid)} {conf:.2f}"
                x = int(poly[0][0])
                y = int(poly[0][1])
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(im, (x, y - th - 10), (x + tw + 5, y), col, -1)
                cv2.putText(im, lbl, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                j = j + 1
        else:
            # regular
            for box in dets:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                col = clrs.get(cid, (255, 255, 255))
                cv2.rectangle(im, (x1, y1), (x2, y2), col, 3)
                
                lbl = f"{classnames.get(cid, cid)} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(im, (x1, y1 - th - 10), (x1 + tw + 5, y1), col, -1)
                cv2.putText(im, lbl, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        outimgpath = outp / f"detected_{imgpath.name}"
        cv2.imwrite(str(outimgpath), im)
        
        # savetxt
        if savetxt:
            lblfile = lbldir / f"{imgpath.stem}.txt"
            f = open(lblfile, 'w')
            if r.obb != None:
                polys = dets.xyxyxyxy.cpu().numpy()
                imgh, imgw = r.orig_shape
                for poly, cid, conf in zip(polys, dets.cls, dets.conf):
                    cid = int(cid)
                    conf = float(conf)
                    pnorm = poly.copy()
                    pnorm[:, 0] = pnorm[:, 0] / imgw
                    pnorm[:, 1] = pnorm[:, 1] / imgh
                    coords = ' '.join(f'{c:.6f}' for pt in pnorm for c in pt)
                    f.write(f"{cid} {coords} {conf:.4f}\n")
            else:
                for box in dets:
                    cid = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    imgh, imgw = r.orig_shape
                    xctr = ((x1 + x2) / 2) / imgw
                    yctr = ((y1 + y2) / 2) / imgh
                    w = (x2 - x1) / imgw
                    h = (y2 - y1) / imgh
                    f.write(f"{cid} {xctr:.6f} {yctr:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")
            f.close()
        
        i = i + 1
    
    # summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    totaldet = 0
    for s in summary:
        totaldet = totaldet + s['detections']
    imgswithdet = 0
    for s in summary:
        if s['detections'] > 0:
            imgswithdet = imgswithdet + 1
    print(f"Total images: {len(imglist)}")
    print(f"Images with detections: {imgswithdet}")
    print(f"Total detections: {totaldet}")
    print(f"Results saved to: {outp}")
    
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, default='.')
    parser.add_argument('--output', type=str, default='./results')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--save-txt', action='store_true')
    
    args = parser.parse_args()
    
    scriptdir = Path(__file__).parent
    if args.input == '.':
        indir = scriptdir
    else:
        indir = Path(args.input)
    if args.output == './results':
        outdir = scriptdir / 'results'
    else:
        outdir = Path(args.output)
    
    runinfer(args.model, indir, outdir, args.conf, args.save_txt)

if __name__ == "__main__":
    main()
