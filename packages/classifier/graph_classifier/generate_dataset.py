from io import BytesIO
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.table import Table
import numpy as np
import os
from PIL import Image, ImageFilter
import random
from tqdm import tqdm

# =====================
# CONFIGURATION
# =====================

RANDOM_SEED = 42

# image size (width, height)
IMAGE_WIDTH  = 768
IMAGE_HEIGHT = 768

# number of single object images per class (default: 400 for local GPU, 500 for HPC)
NUM_IMAGES_PER_CLASS = 500

# number of multi-objects images in total (default: 1000 for local GPU, 2000 for HPC)
NUM_MULTI_IMAGES = 2000

# train/validation split
TRAIN_RATIO = 0.8

# objects per multi-object image
MIN_OBJ_PER_MULTI = 3
MAX_OBJ_PER_MULTI = 5  # keep less than 5 to avoid overcrowding

# object size relative to canvas (smaller value to reduce overlapping)
OBJ_MIN_SCALE_SINGLE = 0.4
OBJ_MAX_SCALE_SINGLE = 0.7

OBJ_MIN_SCALE_MULTI = 0.2
OBJ_MAX_SCALE_MULTI = 0.4

# output directory, also create YOLO structure:
# dataset/
#   ├── images/
#   │     ├── train/
#   │     └── val/
#   └── labels/
#         ├── train/
#         └── val/
OUTPUT_DIR = "dataset"

# =====================
# CLASS DEFINITIONS
# =====================

CLASSES = \
[
    "column_chart",                 # 0
    "line_chart",                   # 1
    "pie_chart",                    # 2
    "bar_chart",                    # 3
    "area_chart",                   # 4
    "scatter_chart",                # 5
    "histogram_chart",              # 6
    "waterfall_chart",              # 7
    "flowchart",                    # 8
    "hierarchy_vertical_diagram",   # 9
    "hierarchy_horizontal_diagram", # 10
    "table",                        # 11
    "circle",                       # 12
    "oval",                         # 13
    "triangle_right",               # 14
    "triangle_isosceles",           # 15
    "diamond",                      # 16
    "parallelogram",                # 17
    "trapezoid",                    # 18
    "pentagon",                     # 19
    "hexagon",                      # 20
    "octagon",                      # 21
    "rectangle"                     # 22
]

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =====================
# UTILS FUNCTIONS
# =====================

def ensure_directory():
    for split in ["train", "val"]:
        image_dir = os.path.join(OUTPUT_DIR, "images", split)
        label_dir = os.path.join(OUTPUT_DIR, "labels", split)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

# create white background image
def create_background(width, height):
    background = np.ones((height, width, 3), dtype=np.uint8) * 255
    image      = Image.fromarray(background)

    return image.convert("RGBA")

# add light Gaussian noise and optional blur
def add_noise_and_artifacts(final_image):
    array = np.array(final_image).astype(np.float32)

    # light gaussian noise
    noise_std = random.uniform(2.0, 6.0)
    noise     = np.random.normal(0, noise_std, array.shape)
    array     = np.clip(array + noise, 0, 255).astype(np.uint8)
    image     = Image.fromarray(array)

    # extra light blur
    if random.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.0)))

    return image

# convert a Matplotlib figure to a PIL RGBA image
def convert_figure_to_image(fig, width, height):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.05, dpi=100)
    plt.close(fig)

    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    image = image.resize((width, height), Image.LANCZOS)
    return image

# random color generator
def random_color():
    return (random.random(), random.random(), random.random())

# list of random colors
def random_color_list(n):
    return [random_color() for _ in range(n)]

# =====================
# CHART RENDERERS
# =====================

def render_column_chart(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    n = random.randint(4, 10)
    x = np.arange(n)
    y = np.random.randint(10, 100, size=n)

    axis.bar(x, y, color=random_color_list(n))
    axis.set_xticks(x)
    axis.set_xticklabels([f"C{idx}" for idx in range(n)], rotation=45)

    return convert_figure_to_image(figure, width, height)

def render_line_chart(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    n = random.randint(20, 60)
    x = np.arange(n)
    y = np.cumsum(np.random.randn(n)) + 10
    axis.plot(x, y, marker="o", linewidth=2, color=random_color())

    return convert_figure_to_image(figure, width, height)

def render_pie_chart(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    n     = random.randint(3, 8)
    sizes = np.random.rand(n)

    sizes /= sizes.sum()
    labels = [f"S{idx}" for idx in range(n)]
    colors = random_color_list(n)
    axis.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=random.randint(0, 360), colors=colors)

    return convert_figure_to_image(figure, width, height)

def render_bar_chart(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    n = random.randint(4, 10)
    y = np.random.randint(10, 100, size=n)
    x = np.arange(n)

    axis.barh(x, y, color=random_color_list(n))
    axis.set_yticks(x)
    axis.set_yticklabels([f"C{idx}" for idx in range(n)])

    return convert_figure_to_image(figure, width, height)

def render_area_chart(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    n = random.randint(30, 80)
    x = np.linspace(0, 10, n)
    y = np.abs(np.cumsum(np.random.randn(n))) + 5
    axis.fill_between(x, y, alpha=0.7, color=random_color())

    return convert_figure_to_image(figure, width, height)

def render_scatter_chart(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    n = random.randint(50, 200)
    x = np.random.randn(n)
    y = np.random.randn(n) * 1.5 + 2
    axis.scatter(x, y, s=np.random.randint(20, 80, size=n), color=random_color())

    return convert_figure_to_image(figure, width, height)

def render_histogram_chart(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    data = np.random.randn(500) * random.uniform(0.5, 2.0) + random.uniform(-1, 3)
    axis.hist(data, bins=random.randint(10, 30), color=random_color())

    return convert_figure_to_image(figure, width, height)

def render_waterfall_chart(width, height):
    figure, axis   = plt.subplots(figsize=(width/100, height/100), dpi=100)
    n              = random.randint(5, 10)
    changes        = np.random.randint(-50, 80, size=n)
    cumulative_sum = [0]

    # compute cumulative sums
    for change in changes:
        cumulative_sum.append(cumulative_sum[-1] + change)
    cumulative_sum = np.array(cumulative_sum)

    # bar positions
    x      = np.arange(1, n+2)
    colors = random_color_list(n + 1)

    # individual steps
    for idx in range(n):
        start = cumulative_sum[idx]
        end   = cumulative_sum[idx+1]

        y = min(start, end)
        h = abs(end - start)
        axis.bar(idx+1, h, bottom=y, color=colors[idx])

    # total bar
    total = cumulative_sum[-1]
    axis.bar(n+1, total, bottom=0, color=colors[-1])
    axis.set_xticks(x)
    axis.set_xticklabels([f"S{idx}" for idx in range(1, n+1)] + ["Total"], rotation=45)

    return convert_figure_to_image(figure, width, height)

def render_flowchart(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 10)
    axis.axis("off")

    boxes = \
    [
        (1, 7, 3, 2, "Start"),
        (1, 4, 3, 2, "Step 1"),
        (1, 1, 3, 2, "End"),
        (6, 4, 3, 2, "Decision"),
    ]

    # draw boxes
    for (x, y, w, h, text) in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", linewidth=1, color=random_color())
        axis.add_patch(rect)
        axis.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)

    # draw arrows
    def arrow(x1, y1, x2, y2):
        axis.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", linewidth=1))

    arrow(2.5, 7, 2.5, 6)   # start    -> step1
    arrow(2.5, 4, 2.5, 3)   # step1    -> end
    arrow(4, 5, 6, 5)       # step1    -> decision
    arrow(7.5, 4, 7.5, 3)   # decision -> end

    return convert_figure_to_image(figure, width, height)

def render_hierarchy_vertical_diagram(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 10)
    axis.axis("off")

    BOX_W = 2
    BOX_H = 1

    # nodes: (center_x, center_y, text)
    nodes = \
    [
        (5, 8.5, "CEO"),
        (2.5, 6, "Manager A"),
        (7.5, 6, "Manager B"),
        (1, 3.5, "Team 1"),
        (4, 3.5, "Team 2"),
        (6, 3.5, "Team 3"),
        (9, 3.5, "Team 4"),
    ]

    # store patch references by name for connections
    node_boxes = {}

    # draw nodes
    for x, y, text in nodes:

        # convert center to lower-left corner
        lower_left_x = x - BOX_W / 2
        lower_left_y = y - BOX_H / 2

        rectangle = patches.FancyBboxPatch((lower_left_x, lower_left_y), BOX_W, BOX_H, boxstyle="round,pad=0.2",
                                           linewidth=1.2, edgecolor="black", facecolor=random_color())
        axis.add_patch(rectangle)
        axis.text(x, y, text, ha="center", va="center", fontsize=10)

        node_boxes[text] = rectangle

    # arrow connection
    def connect_nodes(parent_text, child_text):
        parent = node_boxes[parent_text]
        child = node_boxes[child_text]

        # bottom center of parent
        x1 = parent.get_x() + BOX_W / 2
        y1 = parent.get_y()  # bottom edge

        # top center of child
        x2 = child.get_x() + BOX_W / 2
        y2 = child.get_y() + BOX_H  # top edge

        axis.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2))

    # draw connections
    connect_nodes("CEO", "Manager A")
    connect_nodes("CEO", "Manager B")
    connect_nodes("Manager A", "Team 1")
    connect_nodes("Manager A", "Team 2")
    connect_nodes("Manager B", "Team 3")
    connect_nodes("Manager B", "Team 4")

    return convert_figure_to_image(figure, width, height)

def render_hierarchy_horizontal_diagram(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 10)
    axis.axis("off")

    # box drawer with random_color()
    def add_box(x, y, text):
        rectangle = patches.FancyBboxPatch((x, y), 2.5, 0.8, boxstyle="round, pad=0.3", linewidth=1.2, edgecolor="black", facecolor=random_color())
        axis.add_patch(rectangle)
        axis.text(x + 1.25, y + 0.4, text, ha="center", va="center", fontsize=10)
        return rectangle

    # correct connection (touch left/right edges)
    def connect(box1, box2):
        x1 = box1.get_x() + box1.get_width()
        y1 = box1.get_y() + box1.get_height() / 2
        x2 = box2.get_x()
        y2 = box2.get_y() + box2.get_height() / 2
        axis.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4))

    b_main = add_box(1, 4.5, "A")

    b_B = add_box(5, 7, "B")
    b_C = add_box(5, 4.5, "C")
    b_D = add_box(5, 2, "D")
    b_E = add_box(9, 8, "E")
    b_F = add_box(9, 6, "F")
    b_G = add_box(9, 4, "G")

    # draw connections
    connect(b_main, b_B)
    connect(b_main, b_C)
    connect(b_main, b_D)
    connect(b_B, b_E)
    connect(b_B, b_F)
    connect(b_B, b_G)

    return convert_figure_to_image(figure, width, height)

def render_table(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.axis("off")

    n_rows = random.randint(3, 8)
    n_cols = random.randint(3, 6)

    # generate random data
    data       = [[f"{random.randint(0, 999)}" for _ in range(n_cols)] for _ in range(n_rows)]
    col_labels = [f"H{col+1}" for col in range(n_cols)]
    row_labels = [f"R{row+1}" for row in range(n_rows)]

    table       = Table(axis, bbox=[0, 0, 1, 1])
    width_cell  = 1.0 / (n_cols + 1)
    height_cell = 1.0 / (n_rows + 1)

    # header row
    for col, label in enumerate([""] + col_labels):
        table.add_cell(0, col, width_cell, height_cell, text=label, loc="center", facecolor="#CCCCCC")

    # data rows
    for row in range(n_rows):

        # row header
        table.add_cell(row+1, 0, width_cell, height_cell, text=row_labels[row], loc="center", facecolor="#DDDDDD")

        # values
        for col in range(n_cols):
            table.add_cell(row+1, col+1, width_cell, height_cell, text=data[row][col], loc="center", facecolor="white")

    axis.add_table(table)

    return convert_figure_to_image(figure, width, height)

# ====================
# SHAPE RENDERERS
# ====================

def render_circle(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    radius = random.uniform(0.25, 0.45)
    center_x, center_y = 0.5, 0.5

    circle = patches.Circle((center_x, center_y), radius, fill=True, color=random_color(), alpha=0.9, linewidth=1)
    axis.add_patch(circle)

    return convert_figure_to_image(figure, width, height)

def render_oval(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    w = random.uniform(0.4, 0.7)
    h = random.uniform(0.25, 0.5)
    center_x, center_y = 0.5, 0.5

    oval = patches.Ellipse((center_x, center_y), w, h, fill=True, color=random_color(), alpha=0.9, linewidth=1)
    axis.add_patch(oval)

    return convert_figure_to_image(figure, width, height)

def render_triangle_right(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    points = np.array([[0.1, 0.1], [0.1, 0.8], [0.7, 0.1]])

    triangle = patches.Polygon(points, closed=True, color=random_color(), alpha=0.9)
    axis.add_patch(triangle)

    return convert_figure_to_image(figure, width, height)

def render_triangle_isosceles(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    points = np.array([[0.5, 0.85], [0.15, 0.15], [0.85, 0.15]])

    triangle = patches.Polygon(points, closed=True, color=random_color(), alpha=0.9)
    axis.add_patch(triangle)

    return convert_figure_to_image(figure, width, height)

def render_diamond(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    points = np.array([[0.5, 0.85], [0.15, 0.5], [0.5, 0.15], [0.85, 0.5]])

    diamond = patches.Polygon(points, closed=True, color=random_color(), alpha=0.9)
    axis.add_patch(diamond)

    return convert_figure_to_image(figure, width, height)

def render_parallelogram(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    offset = random.uniform(0.15, 0.3)
    points = np.array([[0.2 + offset, 0.8], [0.8 + offset, 0.8], [0.6, 0.2], [0.0, 0.2]])

    par = patches.Polygon(points, closed=True, color=random_color(), alpha=0.9)
    axis.add_patch(par)

    return convert_figure_to_image(figure, width, height)

def render_trapezoid(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    points = np.array([[0.3, 0.8], [0.7, 0.8], [0.85, 0.2], [0.15, 0.2]])

    trap = patches.Polygon(points, closed=True, color=random_color(), alpha=0.9)
    axis.add_patch(trap)

    return convert_figure_to_image(figure, width, height)

def render_polygon_n(width, height, n):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.axis("off")

    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)]) * 0.8

    polygon = patches.Polygon(points, closed=True, color=random_color(), alpha=0.9)
    axis.add_patch(polygon)

    return convert_figure_to_image(figure, width, height)

def render_pentagon(width, height):
    return render_polygon_n(width, height, 5)

def render_hexagon(width, height):
    return render_polygon_n(width, height, 6)

def render_octagon(width, height):
    return render_polygon_n(width, height, 8)

def render_rectangle(width, height):
    figure, axis = plt.subplots(figsize=(width/100, height/100), dpi=100)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    rectangle = patches.Rectangle((0.2, 0.2), 0.6, 0.6, color=random_color(), alpha=0.9)
    axis.add_patch(rectangle)

    return convert_figure_to_image(figure, width, height)

# ==========================
# RENDER FUNCTION MAPPING
# ==========================

RENDER_FUNCS = \
{
    "column_chart": render_column_chart,
    "line_chart": render_line_chart,
    "pie_chart": render_pie_chart,
    "bar_chart": render_bar_chart,
    "area_chart": render_area_chart,
    "scatter_chart": render_scatter_chart,
    "histogram_chart": render_histogram_chart,
    "waterfall_chart": render_waterfall_chart,
    "flowchart": render_flowchart,
    "hierarchy_vertical_diagram": render_hierarchy_vertical_diagram,
    "hierarchy_horizontal_diagram": render_hierarchy_horizontal_diagram,
    "table": render_table,
    "circle": render_circle,
    "oval": render_oval,
    "triangle_right": render_triangle_right,
    "triangle_isosceles": render_triangle_isosceles,
    "diamond": render_diamond,
    "parallelogram": render_parallelogram,
    "trapezoid": render_trapezoid,
    "pentagon": render_pentagon,
    "hexagon": render_hexagon,
    "octagon": render_octagon,
    "rectangle": render_rectangle,
}

# =====================================
# OBJECT PLACEMENT / OVERLAP CONTROL
# =====================================

# check if two boxes overlap beyond a threshold
def boxes_overlap(b1, b2, threshold=0.15):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xa1, ya1 = x1, y1
    xa2, ya2 = x1 + w1, y1 + h1
    xb1, yb1 = x2, y2
    xb2, yb2 = x2 + w2, y2 + h2

    # compute intersections
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False  # no intersection

    # compute intersection over union
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    if union <= 0:
        return False # no union, avoid division by zero

    iou = inter_area / union

    return iou > threshold

# try to place object without overlapping existing ones
def safe_place(canvas_rgba, chart_img, placed_pixel_boxes, W, H,  iou_thr=0.05, max_tries=30, OBJ_MAX_SCALE=0.4):
    ow, oh = chart_img.size

    # try normal placement many times
    for _ in range(max_tries):
        max_x = max(0, W - ow)
        max_y = max(0, H - oh)

        px = random.randint(0, max_x) if max_x > 0 else 0
        py = random.randint(0, max_y) if max_y > 0 else 0

        candidate = (px, py, ow, oh)

        if not any(boxes_overlap(candidate, box, threshold=iou_thr) for box in placed_pixel_boxes):
            return px, py, ow, oh

    # initial fallback: shrink once and try again
    new_w = int(ow * 0.8)
    new_h = int(oh * 0.8)

    if new_w >= 20 and new_h >= 20:
        chart_img = chart_img.resize((new_w, new_h), Image.LANCZOS)
        ow, oh    = new_w, new_h

        # try again with smaller object
        for _ in range(max_tries):
            max_x = max(0, W - ow)
            max_y = max(0, H - oh)
            px = random.randint(0, max_x)
            py = random.randint(0, max_y)

            candidate = (px, py, ow, oh)
            if not any(boxes_overlap(candidate, box, threshold=iou_thr) for box in placed_pixel_boxes):
                return px, py, ow, oh

    # final fallback: guaranteed placement with minimal overlap
    px = random.randint(0, max(0, W - ow))
    py = random.randint(0, max(0, H - oh))

    return px, py, ow, oh

# place object on canvas and return YOLO box
def place_object(canvas_rgba, chart_type, placed_pixel_boxes, OBJ_MIN_SCALE, OBJ_MAX_SCALE):
    W, H = canvas_rgba.size

    # object size as fraction of canvas
    obj_w = int(random.uniform(OBJ_MIN_SCALE, OBJ_MAX_SCALE) * W)
    obj_h = int(random.uniform(OBJ_MIN_SCALE, OBJ_MAX_SCALE) * H)

    chart_img = RENDER_FUNCS[chart_type](obj_w, obj_h)

    # random mild rotation
    angle = random.uniform(-5, 5)
    chart_img = chart_img.rotate(angle, expand=True, resample=Image.BICUBIC)
    ow, oh = chart_img.size

    # place object safely
    px, py, ow, oh = safe_place(canvas_rgba, chart_img, placed_pixel_boxes, W, H, OBJ_MAX_SCALE=OBJ_MAX_SCALE)
    canvas_rgba.alpha_composite(chart_img, dest=(px, py))

    placed_pixel_boxes.append((px, py, ow, oh))

    # normalized YOLO box
    return\
    {
        "class_id": CLASS_NAME_TO_ID[chart_type],
        "x_center_norm": (px + ow / 2) / W,
        "y_center_norm": (py + oh / 2) / H,
        "w_norm": ow / W,
        "h_norm": oh / H,
    }

# save YOLO label file
def save_yolo_label(path, boxes):
    with open(path, "w") as f:
        for box in boxes:
            f.write\
            (
                f"{box['class_id']} "
                f"{box['x_center_norm']:.6f} "
                f"{box['y_center_norm']:.6f} "
                f"{box['w_norm']:.6f} "
                f"{box['h_norm']:.6f}\n"
            )

# ====================================
# SINGLE & MULTI IMAGE GENERATORS
# ====================================

def generate_single_object_images():
    for class_name in CLASSES:
        class_id = CLASS_NAME_TO_ID[class_name]
        for idx in tqdm(range(NUM_IMAGES_PER_CLASS), desc=f"Generating {class_name} singles"):
            split = "train" if random.random() < TRAIN_RATIO else "val"
            image_path = f"{OUTPUT_DIR}/images/{split}/single_{class_id}_{idx:04d}.png"
            label_path = f"{OUTPUT_DIR}/labels/{split}/single_{class_id}_{idx:04d}.txt"

            canvas = create_background(IMAGE_WIDTH, IMAGE_HEIGHT)
            boxes = []
            pixel_boxes = []

            # place the single object
            boxes.append(place_object(canvas, class_name, pixel_boxes, OBJ_MIN_SCALE_SINGLE, OBJ_MAX_SCALE_SINGLE))
            final = add_noise_and_artifacts(canvas.convert("RGB"))
            final.save(image_path)
            save_yolo_label(label_path, boxes)

def generate_multi_object_images():
    for idx in tqdm(range(NUM_MULTI_IMAGES), desc="Generating multi-images"):
        split = "train" if random.random() < TRAIN_RATIO else "val"
        image_path = f"{OUTPUT_DIR}/images/{split}/multi_{idx:06d}.png"
        label_path = f"{OUTPUT_DIR}/labels/{split}/multi_{idx:06d}.txt"

        canvas = create_background(IMAGE_WIDTH, IMAGE_HEIGHT)

        n      = random.randint(MIN_OBJ_PER_MULTI, MAX_OBJ_PER_MULTI)
        chosen = [random.choice(CLASSES) for _ in range(n)]

        boxes = []
        pixel_boxes = []

        # place each object
        for chose in chosen:
            boxes.append(place_object(canvas, chose, pixel_boxes, OBJ_MIN_SCALE_MULTI, OBJ_MAX_SCALE_MULTI))

        final = add_noise_and_artifacts(canvas.convert("RGB"))
        final.save(image_path)
        save_yolo_label(label_path, boxes)

# =================
# MAIN FUNCTION
# =================

if __name__ == "__main__":
    ensure_directory()
    print("Generating single-object images.")
    generate_single_object_images()
    print("Generating multi-object images.")
    generate_multi_object_images()
    print("Generation Completed.")
