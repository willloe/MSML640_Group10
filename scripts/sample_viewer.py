import argparse
import json
import logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleViewer:

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def draw_layout_overlay(self, image: Image.Image, layout: dict) -> Image.Image:

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay, 'RGBA')

        color_map = {
            'title': (255, 0, 0, 100),
            'body': (0, 255, 0, 100),
            'image': (0, 0, 255, 100),
            'logo': (255, 255, 0, 100),
            'caption': (255, 0, 255, 100),
            'footer': (0, 255, 255, 100)
        }

        for elem in layout.get('elements', []):
            x, y, w, h = elem['bbox_xywh']
            elem_type = elem.get('class', 'unknown')
            color = color_map.get(elem_type, (128, 128, 128, 100))

            draw.rectangle([x, y, x + w, y + h], fill=color, outline=color[:3] + (255,), width=2)

            label = f"{elem_type}"
            draw.text((x + 5, y + 5), label, fill=(255, 255, 255, 255))

        return overlay

    def image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def generate_sample_html(self, sample: dict) -> str:
 
        image_path = self.base_dir / sample['image_path']
        layout_path = self.base_dir / sample['layout_json']
        safe_zone_path = self.base_dir / sample['safe_zone_path']
        control_path = self.base_dir / sample['control_map_path']

        image = Image.open(image_path)

        with open(layout_path, 'r') as f:
            layout = json.load(f)

        safe_zone = Image.open(safe_zone_path)
        control_map = Image.open(control_path)

        overlay = self.draw_layout_overlay(image, layout)

        image_b64 = self.image_to_base64(image)
        overlay_b64 = self.image_to_base64(overlay)
        safe_zone_b64 = self.image_to_base64(safe_zone)
        control_b64 = self.image_to_base64(control_map)

        palette = sample.get('palette', {})
        primary = palette.get('primary', '#000000')
        secondary = palette.get('secondary', '#888888')
        accent = palette.get('accent', '#CCCCCC')

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sample {sample['id']}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    margin-top: 20px;
                }}
                .image-box {{
                    border: 1px solid #ddd;
                    padding: 10px;
                    border-radius: 4px;
                }}
                .image-box h3 {{
                    margin-top: 0;
                    color: #555;
                }}
                .image-box img {{
                    width: 100%;
                    height: auto;
                    border-radius: 4px;
                }}
                .palette {{
                    display: flex;
                    gap: 10px;
                    margin-top: 20px;
                }}
                .color-box {{
                    width: 100px;
                    height: 100px;
                    border-radius: 4px;
                    border: 1px solid #ddd;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    color: white;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
                }}
                .metadata {{
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-radius: 4px;
                    margin-top: 20px;
                }}
                .metadata pre {{
                    background-color: #fff;
                    padding: 10px;
                    border-radius: 4px;
                    overflow-x: auto;
                }}
                .nav {{
                    margin-top: 20px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                }}
                .nav a {{
                    margin-right: 10px;
                    padding: 8px 16px;
                    background-color: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                }}
                .nav a:hover {{
                    background-color: #0056b3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Sample: {sample['id']}</h1>

                <div class="palette">
                    <div class="color-box" style="background-color: {primary}">
                        <div>Primary</div>
                        <div>{primary}</div>
                    </div>
                    <div class="color-box" style="background-color: {secondary}">
                        <div>Secondary</div>
                        <div>{secondary}</div>
                    </div>
                    <div class="color-box" style="background-color: {accent}">
                        <div>Accent</div>
                        <div>{accent}</div>
                    </div>
                </div>

                <div class="grid">
                    <div class="image-box">
                        <h3>Background Image</h3>
                        <img src="{image_b64}" alt="Background">
                    </div>
                    <div class="image-box">
                        <h3>Layout Overlay</h3>
                        <img src="{overlay_b64}" alt="Layout Overlay">
                    </div>
                    <div class="image-box">
                        <h3>Safe Zone Mask</h3>
                        <img src="{safe_zone_b64}" alt="Safe Zone">
                        <p style="font-size: 12px; color: #666;">White = safe for background, Black = reserved for elements</p>
                    </div>
                    <div class="image-box">
                        <h3>Control Map</h3>
                        <img src="{control_b64}" alt="Control Map">
                        <p style="font-size: 12px; color: #666;">Element mask for ControlNet</p>
                    </div>
                </div>

                <div class="metadata">
                    <h3>Metadata</h3>
                    <pre>{json.dumps(sample, indent=2)}</pre>
                </div>

                <div class="metadata">
                    <h3>Layout</h3>
                    <pre>{json.dumps(layout, indent=2)}</pre>
                </div>

                <div class="nav">
                    <a href="/">Back to Index</a>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def generate_index_html(self, samples: list) -> str:
        thumbnails_html = ""

        for sample in samples:
            sample_id = sample['id']
            palette = sample.get('palette', {})
            primary = palette.get('primary', '#000000')

            thumbnails_html += f"""
            <div class="sample-card">
                <a href="/sample?id={sample_id}" style="color: inherit; text-decoration: none;">
                    <div class="color-preview" style="background-color: {primary}"></div>
                    <div class="sample-info">
                        <strong>{sample_id}</strong><br>
                        <small>{primary}</small>
                    </div>
                </a>
            </div>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sample Index</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                }}
                .stats {{
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-radius: 4px;
                    margin-bottom: 20px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 15px;
                }}
                .sample-card {{
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    overflow: hidden;
                    transition: box-shadow 0.2s;
                }}
                .sample-card:hover {{
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .color-preview {{
                    width: 100%;
                    height: 120px;
                }}
                .sample-info {{
                    padding: 10px;
                    background-color: white;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Sample Viewer</h1>

                <div class="stats">
                    <strong>Total Samples:</strong> {len(samples)}
                </div>

                <div class="grid">
                    {thumbnails_html}
                </div>
            </div>
        </body>
        </html>
        """

        return html


class ViewerHandler(BaseHTTPRequestHandler):

    samples = []
    samples_dict = {}
    viewer = None
    base_dir = None

    def log_message(self, format, *args):
        logger.info("%s - - [%s] %s" % (
            self.address_string(),
            self.log_date_time_string(),
            format % args
        ))

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        try:
            if path == '/':

                html = self.viewer.generate_index_html(self.samples)
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())

            elif path == '/sample':

                sample_id = query.get('id', [None])[0]

                if sample_id and sample_id in self.samples_dict:
                    sample = self.samples_dict[sample_id]
                    html = self.viewer.generate_sample_html(sample)
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(html.encode())
                else:
                    self.send_error(404, 'Sample not found')

            else:
                self.send_error(404, 'Page not found')

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self.send_error(500, str(e))


def main():
    parser = argparse.ArgumentParser(description='Sample viewer web UI')
    parser.add_argument('--index', type=str, default='data/processed/index.jsonl',
                        help='Path to index.jsonl')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to run server on')

    args = parser.parse_args()

    index_path = Path(args.index)

    if not index_path.exists():
        logger.error(f"Index file not found: {index_path}")
        return

    samples = []
    with open(index_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} samples from {index_path}")

    samples_dict = {s['id']: s for s in samples}

    base_dir = index_path.parent.parent

    viewer = SampleViewer(base_dir)

    ViewerHandler.samples = samples
    ViewerHandler.samples_dict = samples_dict
    ViewerHandler.viewer = viewer
    ViewerHandler.base_dir = base_dir

    server = HTTPServer(('localhost', args.port), ViewerHandler)

    logger.info(f"\nSample viewer running at http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down")
        server.shutdown()


if __name__ == '__main__':
    main()
