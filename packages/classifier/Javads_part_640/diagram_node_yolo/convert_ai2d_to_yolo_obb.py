import os
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

clsnames = [
    "diagram_node",
    "arrow",
    "text_label",
    "image_region",
]

ai2d_mapping = {
    "blob": 0,
    "regionBoundary": 0,
    "arrow": 1,
    "arrowHead": 1,
    "text": 2,
    "textRegion": 2,
    "imageRegion": 3,
    "image": 3,
}


def poly_to_obb(poly, imgw, imgh):
    if len(poly) < 3:
        return None
    
    pts = np.array(poly, dtype=np.float32)
    
    try:
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
    except:
        return None
    
    # norm
    norm = []
    for p in box:
        xn = p[0] / imgw
        yn = p[1] / imgh
        xn = max(0, min(1, xn))
        yn = max(0, min(1, yn))
        norm.extend([xn, yn])
    
    return norm


def bbox_to_obb(bbox, imgw, imgh):
    try:
        if 'min' in bbox and 'max' in bbox:
            x1, y1 = bbox['min']
            x2, y2 = bbox['max']
        elif 'x' in bbox:
            x1 = bbox['x']
            y1 = bbox['y']
            x2 = x1 + bbox['width']
            y2 = y1 + bbox['height']
        else:
            return None
        
        # corners
        corners = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]
        
        return poly_to_obb(corners, imgw, imgh)
    except:
        return None


def parse_annotation(annpath, imgw, imgh):
    f = open(annpath, 'r')
    data = json.load(f)
    f.close()
    
    anns = []
    
    # blobs
    if 'blobs' in data:
        for blobid, blobdata in data['blobs'].items():
            if 'polygon' in blobdata:
                obb = poly_to_obb(blobdata['polygon'], imgw, imgh)
                if obb:
                    anns.append((0, obb))
            elif 'rectangle' in blobdata:
                obb = bbox_to_obb(blobdata['rectangle'], imgw, imgh)
                if obb:
                    anns.append((0, obb))
    
    # arrows
    if 'arrows' in data:
        for arrowid, arrowdata in data['arrows'].items():
            if 'polygon' in arrowdata:
                obb = poly_to_obb(arrowdata['polygon'], imgw, imgh)
                if obb:
                    anns.append((1, obb))
            elif 'head' in arrowdata and 'tail' in arrowdata:
                head = arrowdata['head']
                tail = arrowdata['tail']
                
                dx = head[0] - tail[0]
                dy = head[1] - tail[1]
                length = np.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    width = max(10, length * 0.1)
                    
                    px = -dy / length * width / 2
                    py = dx / length * width / 2
                    
                    polygon = [
                        [tail[0] + px, tail[1] + py],
                        [tail[0] - px, tail[1] - py],
                        [head[0] - px, head[1] - py],
                        [head[0] + px, head[1] + py],
                    ]
                    
                    obb = poly_to_obb(polygon, imgw, imgh)
                    if obb:
                        anns.append((1, obb))
    
    # text
    if 'text' in data:
        for txtid, txtdata in data['text'].items():
            if 'rectangle' in txtdata:
                obb = bbox_to_obb(txtdata['rectangle'], imgw, imgh)
                if obb:
                    anns.append((2, obb))
            elif 'polygon' in txtdata:
                obb = poly_to_obb(txtdata['polygon'], imgw, imgh)
                if obb:
                    anns.append((2, obb))
    
    # imageRegions
    if 'imageRegions' in data:
        for regid, regdata in data['imageRegions'].items():
            if 'rectangle' in regdata:
                obb = bbox_to_obb(regdata['rectangle'], imgw, imgh)
                if obb:
                    anns.append((3, obb))
            elif 'polygon' in regdata:
                obb = poly_to_obb(regdata['polygon'], imgw, imgh)
                if obb:
                    anns.append((3, obb))
    
    # containers
    if 'containers' in data:
        for contid, contdata in data['containers'].items():
            if 'rectangle' in contdata:
                obb = bbox_to_obb(contdata['rectangle'], imgw, imgh)
                if obb:
                    anns.append((0, obb))
            elif 'polygon' in contdata:
                obb = poly_to_obb(contdata['polygon'], imgw, imgh)
                if obb:
                    anns.append((0, obb))
    
    return anns


def makeyaml(outpath, classnames):
    yamlcontent = f"""path: {os.path.abspath(outpath)}
        train: images
        val: images

        names:
        """
    for i, name in enumerate(classnames):
        yamlcontent = yamlcontent + f"  {i}: {name}\n"
    
    yamlpath = os.path.join(outpath, 'dataset.yaml')
    f = open(yamlpath, 'w')
    f.write(yamlcontent)
    f.close()
    
    print(f"Created dataset.yaml at {yamlpath}")


def convert(ai2dpath, outpath, trainsplit=0.8):
    ai2dpath = Path(ai2dpath)
    outpath = Path(outpath)
    
    # check
    imgdir = ai2dpath / 'images'
    anndir = ai2dpath / 'annotations'
    
    if not imgdir.exists():
        raise ValueError(f"Images directory not found: {imgdir}")
    if not anndir.exists():
        raise ValueError(f"Annotations directory not found: {anndir}")
    
    # output
    outimgs = outpath / 'images'
    outlbls = outpath / 'labels'
    outimgs.mkdir(parents=True, exist_ok=True)
    outlbls.mkdir(parents=True, exist_ok=True)
    
    # images
    imgfiles = list(imgdir.glob('*.png'))
    imgfiles = imgfiles + list(imgdir.glob('*.jpg'))
    
    print(f"Found {len(imgfiles)} images in AI2D dataset")
    
    stats = {
        'processed': 0,
        'skipped': 0,
        'total_annotations': 0,
        'class_counts': {}
    }
    for n in clsnames:
        stats['class_counts'][n] = 0
    
    for imgpath in tqdm(imgfiles, desc="Converting AI2D"):
        imgid = imgpath.stem
        annpath = anndir / f"{imgid}.json"
        
        if not annpath.exists():
            stats['skipped'] = stats['skipped'] + 1
            continue
        
        # dimensions
        try:
            im = Image.open(imgpath)
            imgw, imgh = im.size
            im.close()
        except Exception as e:
            print(f"Error loading image {imgpath}: {e}")
            stats['skipped'] = stats['skipped'] + 1
            continue
        
        # parse
        try:
            anns = parse_annotation(str(annpath), imgw, imgh)
        except Exception as e:
            print(f"Error parsing annotation {annpath}: {e}")
            stats['skipped'] = stats['skipped'] + 1
            continue
        
        if len(anns) == 0:
            stats['skipped'] = stats['skipped'] + 1
            continue
        
        # copy
        outimgpath = outimgs / imgpath.name
        shutil.copy2(imgpath, outimgpath)
        
        # labels
        lblpath = outlbls / f"{imgid}.txt"
        f = open(lblpath, 'w')
        for classid, obbcoords in anns:
            coordstr = ' '.join(f'{c:.6f}' for c in obbcoords)
            f.write(f"{classid} {coordstr}\n")
            stats['class_counts'][clsnames[classid]] = stats['class_counts'][clsnames[classid]] + 1
            stats['total_annotations'] = stats['total_annotations'] + 1
        f.close()
        
        stats['processed'] = stats['processed'] + 1
    
    # yaml
    makeyaml(str(outpath), clsnames)
    
    # stats
    print(f"processed: {stats['processed']}")
    print(f"skipped: {stats['skipped']}")
    print(f"total: {stats['total_annotations']}")
    for name, cnt in stats['class_counts'].items():
        print(f"  {name}: {cnt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ai2d_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='./ai2d-yolo-obb')
    parser.add_argument('--train_split', type=float, default=0.8)
    
    args = parser.parse_args()
    
    convert(args.ai2d_path, args.output_path, args.train_split)


if __name__ == '__main__':
    main()
