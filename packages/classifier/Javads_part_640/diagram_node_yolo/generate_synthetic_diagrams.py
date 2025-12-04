import os
import random
import math
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
import cv2

clsnames = [
    "diagram_node",
    "arrow",
    "text_label",
    "image_region",
]

palettes = {
    'corporate': ['#2C3E50', '#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6'],
    'pastel': ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFDFba', '#E0BBE4'],
    'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
    'mono': ['#2C3E50', '#34495E', '#7F8C8D', '#95A5A6', '#BDC3C7', '#ECF0F1'],
    'tech': ['#00D4FF', '#7C3AED', '#10B981', '#F59E0B', '#EF4444', '#6366F1'],
}

bgcolors = ['#FFFFFF', '#F5F5F5', '#FFFEF0', '#F0F8FF', '#FFF5EE', '#F8F8FF']


def hex2rgb(hx):
    hx = hx.lstrip('#')
    return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))


def pts_to_obb(pts, imgw, imgh):
    ptsarr = np.array(pts, dtype=np.float32)
    rect = cv2.minAreaRect(ptsarr)
    box = cv2.boxPoints(rect)
    
    norm = []
    for p in box:
        xn = max(0, min(1, p[0] / imgw))
        yn = max(0, min(1, p[1] / imgh))
        norm.extend([xn, yn])
    
    return norm


def bbox_to_obb(x1, y1, x2, y2, imgw, imgh, angle=0):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    cosa = math.cos(angle)
    sina = math.sin(angle)
    
    corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    
    rotated = []
    for dx, dy in corners:
        rx = cx + dx * cosa - dy * sina
        ry = cy + dx * sina + dy * cosa
        rotated.append((rx, ry))
    
    return pts_to_obb(rotated, imgw, imgh)


class ShapeGen:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.pal = random.choice(list(palettes.values()))
    
    def randcol(self):
        return random.choice(self.pal)
    
    def rect(self, draw, x, y, w, h, angle=0, filled=True):
        col = self.randcol()
        
        if angle == 0:
            if filled:
                draw.rectangle([x, y, x+w, y+h], fill=hex2rgb(col), outline=hex2rgb('#2C3E50'), width=2)
            else:
                draw.rectangle([x, y, x+w, y+h], outline=hex2rgb(col), width=3)
            return bbox_to_obb(x, y, x+w, y+h, self.w, self.h)
        else:
            # rotated
            cx = x + w/2
            cy = y + h/2
            cosa = math.cos(angle)
            sina = math.sin(angle)
            corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
            rotated = []
            for dx, dy in corners:
                rx = cx + dx*cosa - dy*sina
                ry = cy + dx*sina + dy*cosa
                rotated.append((rx, ry))
            
            if filled:
                draw.polygon(rotated, fill=hex2rgb(col), outline=hex2rgb('#2C3E50'), width=2)
            else:
                draw.polygon(rotated, outline=hex2rgb(col), width=3)
            
            return pts_to_obb(rotated, self.w, self.h)
    
    def roundedrect(self, draw, x, y, w, h, rad=15):
        col = self.randcol()
        draw.rounded_rectangle([x, y, x+w, y+h], radius=rad, fill=hex2rgb(col), outline=hex2rgb('#2C3E50'), width=2)
        return bbox_to_obb(x, y, x+w, y+h, self.w, self.h)
    
    def ellipse(self, draw, x, y, w, h):
        col = self.randcol()
        draw.ellipse([x, y, x+w, y+h], fill=hex2rgb(col), outline=hex2rgb('#2C3E50'), width=2)
        return bbox_to_obb(x, y, x+w, y+h, self.w, self.h)
    
    def diamond(self, draw, x, y, w, h):
        col = self.randcol()
        cx = x + w/2
        cy = y + h/2
        pts = [(cx, y), (x+w, cy), (cx, y+h), (x, cy)]
        draw.polygon(pts, fill=hex2rgb(col), outline=hex2rgb('#2C3E50'), width=2)
        return pts_to_obb(pts, self.w, self.h)
    
    def parallelogram(self, draw, x, y, w, h, skew=0.2):
        col = self.randcol()
        off = int(w * skew)
        pts = [(x + off, y), (x + w, y), (x + w - off, y + h), (x, y + h)]
        draw.polygon(pts, fill=hex2rgb(col), outline=hex2rgb('#2C3E50'), width=2)
        return pts_to_obb(pts, self.w, self.h)
    
    def hexagon(self, draw, x, y, w, h):
        col = self.randcol()
        cx = x + w/2
        cy = y + h/2
        pts = []
        for i in range(6):
            ang = i * math.pi / 3 - math.pi / 6
            px = cx + (w/2) * math.cos(ang)
            py = cy + (h/2) * math.sin(ang)
            pts.append((px, py))
        draw.polygon(pts, fill=hex2rgb(col), outline=hex2rgb('#2C3E50'), width=2)
        return pts_to_obb(pts, self.w, self.h)
    
    def cylinder(self, draw, x, y, w, h):
        col = hex2rgb(self.randcol())
        outline = hex2rgb('#2C3E50')
        
        ellh = h // 6
        
        draw.ellipse([x, y + h - ellh, x + w, y + h + ellh], fill=col, outline=outline, width=2)
        draw.rectangle([x, y + ellh//2, x + w, y + h - ellh//2], fill=col)
        draw.line([(x, y + ellh//2), (x, y + h - ellh//2)], fill=outline, width=2)
        draw.line([(x + w, y + ellh//2), (x + w, y + h - ellh//2)], fill=outline, width=2)
        draw.ellipse([x, y, x + w, y + ellh], fill=col, outline=outline, width=2)
        
        return bbox_to_obb(x, y, x+w, y+h, self.w, self.h)
    
    def cloud(self, draw, x, y, w, h):
        col = hex2rgb(self.randcol())
        outline = hex2rgb('#2C3E50')
        
        cx = x + w/2
        cy = y + h/2
        
        circles = [
            (cx - w*0.25, cy, w*0.35, h*0.4),
            (cx + w*0.15, cy - h*0.1, w*0.4, h*0.45),
            (cx + w*0.25, cy + h*0.1, w*0.3, h*0.35),
            (cx - w*0.1, cy + h*0.15, w*0.35, h*0.35),
            (cx, cy - h*0.15, w*0.3, h*0.35),
        ]
        
        for cx_, cy_, cw, ch in circles:
            draw.ellipse([cx_ - cw/2, cy_ - ch/2, cx_ + cw/2, cy_ + ch/2], fill=col, outline=outline, width=2)
        
        return bbox_to_obb(x, y, x+w, y+h, self.w, self.h)
    
    def gear(self, draw, x, y, size, teeth=8):
        col = hex2rgb(self.randcol())
        outline = hex2rgb('#2C3E50')
        
        cx = x + size/2
        cy = y + size/2
        outerr = size / 2
        innerr = outerr * 0.7
        toothdepth = outerr * 0.15
        
        pts = []
        for i in range(teeth * 2):
            ang = i * math.pi / teeth
            if i % 2 == 0:
                r = outerr
            else:
                r = outerr - toothdepth
            px = cx + r * math.cos(ang)
            py = cy + r * math.sin(ang)
            pts.append((px, py))
        
        draw.polygon(pts, fill=col, outline=outline, width=2)
        
        holer = innerr * 0.4
        draw.ellipse([cx - holer, cy - holer, cx + holer, cy + holer], fill=hex2rgb('#FFFFFF'), outline=outline, width=2)
        
        return bbox_to_obb(x, y, x+size, y+size, self.w, self.h)
    
    def arrow(self, draw, x1, y1, x2, y2, headsize=15, width=3):
        col = hex2rgb(self.randcol())
        
        draw.line([(x1, y1), (x2, y2)], fill=col, width=width)
        
        ang = math.atan2(y2 - y1, x2 - x1)
        headang = math.pi / 6
        
        leftx = x2 - headsize * math.cos(ang - headang)
        lefty = y2 - headsize * math.sin(ang - headang)
        rightx = x2 - headsize * math.cos(ang + headang)
        righty = y2 - headsize * math.sin(ang + headang)
        
        draw.polygon([(x2, y2), (leftx, lefty), (rightx, righty)], fill=col)
        
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            arroww = max(headsize, width * 3)
            px = -dy / length * arroww / 2
            py = dx / length * arroww / 2
            
            pts = [
                (x1 + px, y1 + py),
                (x1 - px, y1 - py),
                (x2 - px, y2 - py),
                (x2 + px, y2 + py),
            ]
            return pts_to_obb(pts, self.w, self.h)
        
        return bbox_to_obb(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), self.w, self.h)
    
    def curvedarrow(self, draw, x1, y1, x2, y2, headsize=15):
        col = hex2rgb(self.randcol())
        
        cx = (x1 + x2) / 2
        cy = min(y1, y2) - abs(x2 - x1) * 0.3
        
        pts = []
        for t in np.linspace(0, 1, 20):
            px = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
            py = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
            pts.append((px, py))
        
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i+1]], fill=col, width=3)
        
        ang = math.atan2(y2 - pts[-2][1], x2 - pts[-2][0])
        headang = math.pi / 6
        
        leftx = x2 - headsize * math.cos(ang - headang)
        lefty = y2 - headsize * math.sin(ang - headang)
        rightx = x2 - headsize * math.cos(ang + headang)
        righty = y2 - headsize * math.sin(ang + headang)
        
        draw.polygon([(x2, y2), (leftx, lefty), (rightx, righty)], fill=col)
        
        allx = [p[0] for p in pts] + [leftx, rightx]
        ally = [p[1] for p in pts] + [lefty, righty]
        
        return bbox_to_obb(min(allx), min(ally), max(allx), max(ally), self.w, self.h)
    
    def textbox(self, draw, x, y, w, h, txt=""):
        bgcol = hex2rgb('#FFFEF0')
        bordercol = hex2rgb('#2C3E50')
        
        draw.rectangle([x, y, x+w, y+h], fill=bgcol, outline=bordercol, width=1)
        
        if txt == "":
            txt = random.choice(['Label', 'Node', 'Step', 'Process', 'Data', 'Input', 'Output'])
        
        try:
            fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", min(14, h // 2))
        except:
            fnt = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), txt, font=fnt)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        
        tx = x + (w - tw) / 2
        ty = y + (h - th) / 2
        
        draw.text((tx, ty), txt, fill=bordercol, font=fnt)
        
        return bbox_to_obb(x, y, x+w, y+h, self.w, self.h)
    
    def iconplaceholder(self, draw, x, y, size):
        col = hex2rgb(random.choice(['#E8E8E8', '#F0F0F0', '#D8D8D8']))
        border = hex2rgb('#AAAAAA')
        
        draw.rectangle([x, y, x+size, y+size], fill=col, outline=border, width=2)
        draw.line([(x, y), (x+size, y+size)], fill=border, width=1)
        draw.line([(x+size, y), (x, y+size)], fill=border, width=1)
        
        iconsize = size // 3
        ix = x + size//2 - iconsize//2
        iy = y + size//2 - iconsize//2
        draw.ellipse([ix, iy, ix+iconsize, iy+iconsize], fill=border)
        
        return bbox_to_obb(x, y, x+size, y+size, self.w, self.h)
    
    def star(self, draw, x, y, size, pts=5):
        col = hex2rgb(self.randcol())
        outline = hex2rgb('#2C3E50')
        
        cx = x + size/2
        cy = y + size/2
        outerr = size / 2
        innerr = outerr * 0.4
        
        starpts = []
        for i in range(pts * 2):
            ang = i * math.pi / pts - math.pi / 2
            if i % 2 == 0:
                r = outerr
            else:
                r = innerr
            px = cx + r * math.cos(ang)
            py = cy + r * math.sin(ang)
            starpts.append((px, py))
        
        draw.polygon(starpts, fill=col, outline=outline, width=2)
        
        return bbox_to_obb(x, y, x+size, y+size, self.w, self.h)
    
    def triangle(self, draw, x, y, w, h, dir='up'):
        col = hex2rgb(self.randcol())
        outline = hex2rgb('#2C3E50')
        
        if dir == 'up':
            pts = [(x + w/2, y), (x + w, y + h), (x, y + h)]
        elif dir == 'down':
            pts = [(x, y), (x + w, y), (x + w/2, y + h)]
        elif dir == 'left':
            pts = [(x, y + h/2), (x + w, y), (x + w, y + h)]
        else:
            pts = [(x, y), (x + w, y + h/2), (x, y + h)]
        
        draw.polygon(pts, fill=col, outline=outline, width=2)
        
        return pts_to_obb(pts, self.w, self.h)


def gendiagram(imgw=800, imgh=600, minshapes=3, maxshapes=12):
    bgcol = hex2rgb(random.choice(bgcolors))
    img = Image.new('RGB', (imgw, imgh), bgcol)
    draw = ImageDraw.Draw(img)
    
    # grid
    if random.random() > 0.7:
        gridcol = tuple(max(0, c - 20) for c in bgcol)
        for x in range(0, imgw, 50):
            draw.line([(x, 0), (x, imgh)], fill=gridcol, width=1)
        for y in range(0, imgh, 50):
            draw.line([(0, y), (imgw, y)], fill=gridcol, width=1)
    
    gen = ShapeGen(imgw, imgh)
    anns = []
    
    numshapes = random.randint(minshapes, maxshapes)
    occupied = []
    
    def checkoverlap(x, y, w, h, margin=20):
        for ox, oy, ow, oh in occupied:
            if not (x + w + margin < ox or x > ox + ow + margin or y + h + margin < oy or y > oy + oh + margin):
                return True
        return False
    
    def placeshape(minw, maxw, minh, maxh, maxtries=50):
        for _ in range(maxtries):
            w = random.randint(minw, maxw)
            h = random.randint(minh, maxh)
            x = random.randint(20, imgw - w - 20)
            y = random.randint(20, imgh - h - 20)
            
            if not checkoverlap(x, y, w, h):
                occupied.append((x, y, w, h))
                return x, y, w, h
        return None
    
    # shapes
    shapefuncs = [
        ('rectangle', 0, lambda g, d, x, y, w, h: g.rect(d, x, y, w, h, random.uniform(-0.2, 0.2) if random.random() > 0.7 else 0)),
        ('rounded_rect', 0, lambda g, d, x, y, w, h: g.roundedrect(d, x, y, w, h)),
        ('ellipse', 0, lambda g, d, x, y, w, h: g.ellipse(d, x, y, w, h)),
        ('diamond', 0, lambda g, d, x, y, w, h: g.diamond(d, x, y, w, h)),
        ('parallelogram', 0, lambda g, d, x, y, w, h: g.parallelogram(d, x, y, w, h)),
        ('hexagon', 0, lambda g, d, x, y, w, h: g.hexagon(d, x, y, w, h)),
        ('cylinder', 0, lambda g, d, x, y, w, h: g.cylinder(d, x, y, w, h)),
        ('cloud', 0, lambda g, d, x, y, w, h: g.cloud(d, x, y, w, h)),
        ('gear', 0, lambda g, d, x, y, w, h: g.gear(d, x, y, min(w, h))),
        ('star', 0, lambda g, d, x, y, w, h: g.star(d, x, y, min(w, h))),
        ('triangle', 0, lambda g, d, x, y, w, h: g.triangle(d, x, y, w, h, random.choice(['up', 'down', 'left', 'right']))),
        ('text_box', 2, lambda g, d, x, y, w, h: g.textbox(d, x, y, w, h)),
        ('icon', 3, lambda g, d, x, y, w, h: g.iconplaceholder(d, x, y, min(w, h))),
    ]
    
    # place
    for _ in range(numshapes):
        shapetype, classid, drawfunc = random.choice(shapefuncs)
        
        if shapetype in ['text_box']:
            placement = placeshape(60, 150, 25, 50)
        elif shapetype in ['icon']:
            placement = placeshape(40, 100, 40, 100)
        elif shapetype in ['gear', 'star']:
            placement = placeshape(50, 120, 50, 120)
        else:
            placement = placeshape(60, 180, 50, 150)
        
        if placement != None:
            x, y, w, h = placement
            obb = drawfunc(gen, draw, x, y, w, h)
            anns.append((classid, obb))
    
    # arrows
    if len(occupied) >= 2 and random.random() > 0.3:
        numarrows = random.randint(1, min(4, len(occupied) - 1))
        
        for _ in range(numarrows):
            idx1, idx2 = random.sample(range(len(occupied)), 2)
            r1 = occupied[idx1]
            r2 = occupied[idx2]
            
            x1 = r1[0] + r1[2] // 2 + random.randint(-10, 10)
            y1 = r1[1] + r1[3] // 2 + random.randint(-10, 10)
            x2 = r2[0] + r2[2] // 2 + random.randint(-10, 10)
            y2 = r2[1] + r2[3] // 2 + random.randint(-10, 10)
            
            if random.random() > 0.5:
                obb = gen.arrow(draw, x1, y1, x2, y2)
            else:
                obb = gen.curvedarrow(draw, x1, y1, x2, y2)
            
            anns.append((1, obb))
    
    return img, anns


def makeyaml(outpath):
    yamlcontent = f"""path: {os.path.abspath(outpath)}
        train: images
        val: images

        names:
        """
    for i, name in enumerate(clsnames):
        yamlcontent = yamlcontent + f"  {i}: {name}\n"
    
    yamlpath = os.path.join(outpath, 'dataset.yaml')
    f = open(yamlpath, 'w')
    f.write(yamlcontent)
    f.close()
    
    print(f"Created dataset.yaml at {yamlpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./synthetic-diagrams')
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--img_width', type=int, default=800)
    parser.add_argument('--img_height', type=int, default=600)
    parser.add_argument('--min_shapes', type=int, default=3)
    parser.add_argument('--max_shapes', type=int, default=12)
    parser.add_argument('--seed', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.seed != None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    outpath = Path(args.output_path)
    imgdir = outpath / 'images'
    lbldir = outpath / 'labels'
    
    imgdir.mkdir(parents=True, exist_ok=True)
    lbldir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    for n in clsnames:
        stats[n] = 0
    
    print(f"Generating {args.num_images} synthetic diagrams...")
    
    for i in tqdm(range(args.num_images)):
        img, anns = gendiagram(args.img_width, args.img_height, args.min_shapes, args.max_shapes)
        
        # save
        imgpath = imgdir / f"synthetic_{i:05d}.png"
        img.save(imgpath, 'PNG')
        
        # labels
        lblpath = lbldir / f"synthetic_{i:05d}.txt"
        f = open(lblpath, 'w')
        for classid, obbcoords in anns:
            coordstr = ' '.join(f'{c:.6f}' for c in obbcoords)
            f.write(f"{classid} {coordstr}\n")
            stats[clsnames[classid]] = stats[clsnames[classid]] + 1
        f.close()
    
    makeyaml(str(outpath))
    
    print("\n" + "="*50)
    print("Generation Complete!")
    print("="*50)
    print(f"Images generated: {args.num_images}")
    print(f"Output directory: {outpath}")
    print("\nClass distribution:")
    for name, cnt in stats.items():
        print(f"  {name}: {cnt}")


if __name__ == '__main__':
    main()
