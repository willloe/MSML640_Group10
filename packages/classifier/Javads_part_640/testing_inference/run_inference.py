import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import json

logo_classes = {0: "logo", 1: "images"}
node_cls = {0: "diagram_node"}
chart_stuff = {
    0: "column_chart", 1: "line_chart", 2: "pie_chart", 3: "bar_chart",
    4: "area_chart", 5: "scatter_chart", 6: "histogram_chart", 7: "waterfall_chart",
    8: "flowchart", 9: "hierarchy_vertical_diagram", 10: "hierarchy_horizontal_diagram",
    11: "table", 12: "circle", 13: "oval", 14: "triangle_right", 15: "triangle_isosceles",
    16: "diamond", 17: "parallelogram", 18: "trapezoid", 19: "pentagon",
    20: "hexagon", 21: "octagon", 22: "rectangle"
}

logo_colors = {0: (0, 255, 0), 1: (255, 128, 0)}
node_colors = {0: (255, 0, 255)}
chart_colors = {
    0: (0, 255, 255), 1: (255, 255, 0), 2: (0, 128, 255), 3: (255, 0, 128),
    4: (128, 255, 0), 5: (255, 128, 128), 6: (128, 128, 255), 7: (128, 255, 128),
    8: (200, 100, 50), 9: (50, 100, 200), 10: (100, 200, 50), 11: (200, 50, 100),
    12: (192, 192, 0), 13: (192, 0, 192), 14: (0, 192, 192), 15: (128, 128, 0),
    16: (128, 0, 128), 17: (0, 128, 128), 18: (64, 64, 255), 19: (64, 255, 64),
    20: (255, 64, 64), 21: (100, 150, 200), 22: (200, 150, 100)
}

txt_colors = {
    'Title': (50, 205, 50),
    'Caption': (255, 165, 0),
    'Obj-text': (147, 112, 219),
    'Other-text': (70, 130, 180),
    'Page-text': (220, 20, 60),
    'unknown': (128, 128, 128)
}

def grab_images(folder):
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    found = []
    for ext in extensions:
        found.extend(Path(folder).glob(f'*{ext}'))
        found.extend(Path(folder).glob(f'*{ext.upper()}'))
    return sorted(found)

def setup_text_stuff(base_dir):
    # ocr
    try:
        import easyocr
        ocr = easyocr.Reader(['en'], gpu=True)
    except:
        import easyocr
        ocr = easyocr.Reader(['en'], gpu=False)
    
    clf = None
    txt_folder = base_dir / 'text'
    mdl_folder = txt_folder / 'trained_models'
    
    if (mdl_folder / 'classifier_model.pkl').exists():
        sys.path.insert(0, str(txt_folder))
        from text_type_predictor import TextTypePredictor
        clf = TextTypePredictor(mdl_folder)
        print("Text classifier loaded")
    else:
        print("No text classifier, using OCR only")
    
    return ocr, clf

def get_text_from_img(img_file, ocr, clf):
    # extraction
    from PIL import Image
    
    raw = ocr.readtext(str(img_file))
    pil_img = Image.open(img_file)
    
    text_list = []
    boxes = []
    
    for bb, txt, c in raw:
        x_vals = [p[0] for p in bb]
        y_vals = [p[1] for p in bb]
        x = int(min(x_vals))
        y = int(min(y_vals))
        w = int(max(x_vals) - x)
        h = int(max(y_vals) - y)
        boxes.append([x, y, w, h])
        text_list.append({'text': txt, 'conf': c, 'bbox': [x, y, w, h]})
    
    if clf and boxes:
        preds = clf.predict_batch(pil_img, boxes)
        for idx, (ttype, tconf) in enumerate(preds):
            text_list[idx]['type'] = ttype
            text_list[idx]['type_conf'] = float(tconf)
    else:
        for item in text_list:
            item['type'] = 'unknown'
            item['type_conf'] = 0.0
    
    return text_list

def draw_text_stuff(img, text_data):
    for t in text_data:
        x, y, w, h = t['bbox']
        typ = t.get('type', 'unknown')
        c = txt_colors.get(typ, (128, 128, 128))
        
        # dashed
        for i in range(0, w, 10):
            x1 = x + i
            x2 = min(x + i + 5, x + w)
            cv2.line(img, (x1, y), (x2, y), c, 2)
            cv2.line(img, (x1, y + h), (x2, y + h), c, 2)
        for i in range(0, h, 10):
            y1 = y + i
            y2 = min(y + i + 5, y + h)
            cv2.line(img, (x, y1), (x, y2), c, 2)
            cv2.line(img, (x + w, y1), (x + w, y2), c, 2)
        
        label = f"{typ}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y + h), (x + tw + 4, y + h + th + 6), c, -1)
        cv2.putText(img, label, (x + 2, y + h + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

def do_detection(mdl, img_path, thresh, cls_names, clr_map, name):
    # inference
    out = mdl(img_path, conf=thresh, verbose=False)
    r = out[0]
    
    results = []
    
    if r.obb is not None and len(r.obb.cls) > 0:
        polys = r.obb.xyxyxyxy.cpu().numpy()
        for j in range(len(polys)):
            cid = int(r.obb.cls[j])
            conf = float(r.obb.conf[j])
            poly = polys[j]
            results.append({
                'model': name,
                'class_id': cid,
                'class_name': cls_names.get(cid, f"class_{cid}"),
                'confidence': conf,
                'type': 'obb',
                'polygon': poly.tolist(),
                'color': clr_map.get(cid, (255, 255, 255))
            })
    elif r.boxes is not None and len(r.boxes.cls) > 0:
        for box in r.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            results.append({
                'model': name,
                'class_id': cid,
                'class_name': cls_names.get(cid, f"class_{cid}"),
                'confidence': conf,
                'type': 'box',
                'bbox': [x1, y1, x2, y2],
                'color': clr_map.get(cid, (255, 255, 255))
            })
    
    return results

def render_boxes(img, detections):
    # drawing
    for d in detections:
        col = d['color']
        lbl = f"{d['class_name']} {d['confidence']:.2f}"
        
        if d['type'] == 'obb':
            pts = np.array(d['polygon']).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=col, thickness=3)
            x = int(d['polygon'][0][0])
            y = int(d['polygon'][0][1])
        else:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            x, y = x1, y1
        
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x, y - th - 10), (x + tw + 5, y), col, -1)
        cv2.putText(img, lbl, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

def run_everything(model_dict, in_folder, out_folder, conf=0.25, do_text=False, ocr=None, clf=None):
    # main
    imgs = grab_images(Path(in_folder))
    if len(imgs) == 0:
        print(f"No images in {in_folder}")
        return
    
    print(f"Found {len(imgs)} images")
    print(f"Models: {list(model_dict.keys())}")
    if do_text:
        print("Text: ENABLED")
    
    out_path = Path(out_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    
    loaded = {}
    for n, cfg in model_dict.items():
        print(f"Loading {n}: {cfg['path']}")
        loaded[n] = YOLO(cfg['path'])
    
    all_results = []
    
    for i, img_file in enumerate(imgs):
        print(f"\n[{i+1}/{len(imgs)}] {img_file.name}")
        
        dets = []
        img_data = {'image': img_file.name, 'detections': {}}
        
        for n, cfg in model_dict.items():
            det = do_detection(
                loaded[n],
                img_file,
                conf,
                cfg['classes'],
                cfg['colors'],
                n
            )
            dets.extend(det)
            
            counts = {}
            for d in det:
                cn = d['class_name']
                counts[cn] = counts.get(cn, 0) + 1
            
            img_data['detections'][n] = {
                'count': len(det),
                'classes': counts
            }
            
            if len(det) > 0:
                print(f"  {n}: {counts}")
        
        text_stuff = []
        if do_text and ocr:
            text_stuff = get_text_from_img(img_file, ocr, clf)
            print(f"  text: {len(text_stuff)} elements")
            
            type_counts = {}
            for t in text_stuff:
                typ = t.get('type', 'unknown')
                type_counts[typ] = type_counts.get(typ, 0) + 1
            img_data['text'] = {'count': len(text_stuff), 'types': type_counts, 'elements': text_stuff}
        
        all_results.append(img_data)
        
        frame = cv2.imread(str(img_file))
        frame = render_boxes(frame, dets)
        if text_stuff:
            frame = draw_text_stuff(frame, text_stuff)
        
        save_path = out_path / f"detected_{img_file.name}"
        cv2.imwrite(str(save_path), frame)
        
        if len(dets) == 0 and len(text_stuff) == 0:
            print("  no detections")
    
    # summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    for n in model_dict.keys():
        tot = sum(s['detections'].get(n, {}).get('count', 0) for s in all_results)
        print(f"{n}: {tot} detections")
    
    if do_text:
        txt_tot = sum(s.get('text', {}).get('count', 0) for s in all_results)
        print(f"text: {txt_tot} elements")
    
    print(f"Results: {out_path}")
    
    json_file = out_path / "results.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON: {json_file}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Multi-model inference pipeline")
    
    parser.add_argument('--pipeline', type=str, choices=['slides', 'diagrams'], 
                        help='slides=graphs+images_logos+text, diagrams=graphs+nodes+text')
    
    parser.add_argument('--model', type=str, help='Single model path (legacy mode)')
    parser.add_argument('--graphs', type=str, help='Charts/shapes model path')
    parser.add_argument('--images-logos', type=str, help='Images/logos model path')
    parser.add_argument('--nodes', type=str, help='Diagram nodes model path')
    
    parser.add_argument('--input', type=str, default='.')
    parser.add_argument('--output', type=str, default='./results')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--text', action='store_true', help='Enable text extraction')
    
    args = parser.parse_args()
    
    script_loc = Path(__file__).parent
    
    if args.input == '.':
        in_dir = script_loc
    else:
        in_dir = Path(args.input)
    if args.output == './results':
        out_dir = script_loc / 'results'
    else:
        out_dir = Path(args.output)
    
    ocr = None
    clf = None
    if args.text or args.pipeline:
        print("Loading text models...")
        ocr, clf = setup_text_stuff(script_loc)
    
    models = {}
    
    if args.pipeline:
        # presets
        if args.pipeline == 'slides':
            g_path = script_loc / 'graphs' / 'best.pt'
            il_path = script_loc / 'images_logos' / 'best.pt'
            
            if g_path.exists():
                models['graphs'] = {'path': str(g_path), 'classes': chart_stuff, 'colors': chart_colors}
            else:
                print(f"WARNING: graphs model not found at {g_path}")
            
            if il_path.exists():
                models['images_logos'] = {'path': str(il_path), 'classes': logo_classes, 'colors': logo_colors}
            else:
                print(f"WARNING: images_logos model not found at {il_path}")
            
            args.text = True
            
        elif args.pipeline == 'diagrams':
            g_path = script_loc / 'graphs' / 'best.pt'
            n_path = script_loc / 'nodes' / 'best.pt'
            
            if g_path.exists():
                models['graphs'] = {'path': str(g_path), 'classes': chart_stuff, 'colors': chart_colors}
            else:
                print(f"WARNING: graphs model not found at {g_path}")
            
            if n_path.exists():
                models['nodes'] = {'path': str(n_path), 'classes': node_cls, 'colors': node_colors}
            else:
                print(f"WARNING: nodes model not found at {n_path}")
            
            args.text = True
    
    else:
        # manual
        if args.graphs:
            models['graphs'] = {'path': args.graphs, 'classes': chart_stuff, 'colors': chart_colors}
        
        if args.images_logos:
            models['images_logos'] = {'path': args.images_logos, 'classes': logo_classes, 'colors': logo_colors}
        
        if args.nodes:
            models['nodes'] = {'path': args.nodes, 'classes': node_cls, 'colors': node_colors}
        
        # legacy
        if args.model and not models:
            mdl_str = str(args.model).lower()
            if 'node' in mdl_str or 'diagram' in mdl_str:
                models['nodes'] = {'path': args.model, 'classes': node_cls, 'colors': node_colors}
            elif 'chart' in mdl_str or 'graph' in mdl_str:
                models['graphs'] = {'path': args.model, 'classes': chart_stuff, 'colors': chart_colors}
            else:
                models['images_logos'] = {'path': args.model, 'classes': logo_classes, 'colors': logo_colors}
    
    if args.text and ocr is None:
        ocr, clf = setup_text_stuff(script_loc)
    
    run_everything(models, in_dir, out_dir, args.conf, args.text, ocr, clf)

if __name__ == "__main__":
    main()
