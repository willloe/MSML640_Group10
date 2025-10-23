import json, os, uuid, re
from datetime import datetime
import shutil, subprocess, tempfile
try:
    from flask import Flask, request, redirect, url_for, send_from_directory, jsonify, render_template_string, abort
except Exception:
    Flask = None
    request = None
    redirect = None
    url_for = None
    send_from_directory = None
    jsonify = None
    render_template_string = None
    abort = None
from pptx import Presentation

from .core import do_the_processing
from .drawing import make_previews

def load_job_data(jdir: str):
    json_fs = []
    for f in os.listdir(jdir):
        if f.endswith('.spatial.json'):
            json_fs.append(f)
    if not json_fs:
        raise FileNotFoundError('No JSON found')
    json_f = json_fs[0]
    with open(os.path.join(jdir, json_f), 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data, json_f


def save_job_data(jdir: str, data: dict, json_f: str) -> None:
    path = os.path.join(jdir, json_f)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def make_the_app():
    if Flask is None:
        raise RuntimeError("Flask is not installed.")

    setup_web_dirs()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

    INDEX_HTML = """
    <!doctype html>
    <html>
    <head>
      <meta charset=\"utf-8\"/>
      <title>PPTX Spatial Map</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
        .card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; max-width: 760px; }
        .row { margin-top: 16px; }
        input[type=file] { padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; width: 100%; }
        button { background: #2563eb; color: white; border: none; border-radius: 6px; padding: 10px 14px; cursor: pointer; }
        .jobs { margin-top: 32px; max-width: 760px; }
        .job { padding: 10px 0; border-bottom: 1px solid #e2e8f0; }
      </style>
    </head>
    <body>
      <h2>PPTX Spatial Map</h2>
      <div class=\"card\">
        <form action=\"{{ url_for('upload') }}\" method=\"post\" enctype=\"multipart/form-data\">
          <label for=\"file\">Upload a .pptx file</label>
          <div class=\"row\"><input id=\"file\" name=\"file\" type=\"file\" accept=\".pptx\" required /></div>
          <div class=\"row\"><button type=\"submit\">Process</button></div>
        </form>
      </div>
      {% if jobs %}
      <div class=\"jobs\">
        <h3>Recent Jobs</h3>
        {% for j in jobs %}
          <div class=\"job\">
            <a href=\"{{ url_for('view_job', job_id=j['job_id']) }}\">{{ j['pptx_name'] }}</a>
            <span> · {{ j['num_slides'] }} slides</span>
          </div>
        {% endfor %}
      </div>
      {% endif %}
    </body>
    </html>
    """

    JOB_HTML = """
    <!doctype html>
    <html>
    <head>
      <meta charset=\"utf-8\"/>
      <title>Job {{ job_id }} - PPTX Spatial Map</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
        .row { margin: 16px 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }
        .card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }
        img { width: 100%; height: auto; border-radius: 6px; border: 1px solid #e2e8f0; }
        a.button { background: #2563eb; color: white; padding: 8px 12px; border-radius: 6px; text-decoration: none; }
      </style>
    </head>
    <body>
      <a href=\"{{ url_for('index') }}\">← Back</a>
      <h2>{{ pptx_name }}</h2>
      <div class=\"row\">
        <a class=\"button\" href=\"{{ url_for('download_json', job_id=job_id) }}\">Download JSON</a>
      </div>
      <div class=\"grid\">
        {% for i in range(num_slides) %}
          <div class=\"card\">
            <div style=\"display:flex; justify-content: space-between; align-items:center; margin-bottom:6px;\">
              <strong>Slide {{ i }}</strong>
              <a class=\"button\" href=\"{{ url_for('edit_slide', job_id=job_id, slide_index=i) }}\">Edit</a>
            </div>
            <img loading=\"lazy\" src=\"{{ url_for('preview_image', job_id=job_id, filename='slide_' + ('%02d' % i) + '.png') }}\" alt=\"slide {{ i }}\" />
          </div>
        {% endfor %}
      </div>
    </body>
    </html>
    """

    EDITOR_HTML = """
    <!doctype html>
    <html>
    <head>
      <meta charset=\"utf-8\"/>
      <title>Edit Slide {{ slide_index }} - {{ pptx_name }}</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }
        .toolbar { display:flex; gap:8px; align-items:center; margin-bottom: 12px; }
        .button { background:#2563eb; color:#fff; border:none; border-radius:6px; padding:8px 12px; text-decoration:none; cursor:pointer; }
        .layout { display:grid; grid-template-columns: 1fr 1fr; gap:16px; align-items:start; }
        .pane { border:1px solid #e2e8f0; border-radius:8px; padding:8px; }
        .stage { position:relative; width:100%; background:#fff; aspect-ratio: {{ canvas_w }} / {{ canvas_h }}; overflow:hidden; }
        .rect { position:absolute; border:2px solid #111827; background: rgba(17,24,39,0.05); box-sizing: border-box; }
        .rect[data-type=text] { border-color:#1e90ff; background: rgba(30,144,255,0.08); }
        .rect[data-type=icon] { border-color:#9933ff; background: rgba(153,51,255,0.08); }
        .rect[data-type=image] { border-color:#ff8c00; background: rgba(255,140,0,0.10); }
        .rect[data-type=shape] { border-color:#008b8b; background: rgba(0,139,139,0.08); }
        .rect[data-type=table] { border-color:#228b22; background: rgba(34,139,34,0.10); }
        .rect[data-type=chart] { border-color:#dc143c; background: rgba(220,20,60,0.10); }
        .rect[data-type=connector] { border-color:#787878; background: rgba(120,120,120,0.10); }
        .rect[data-type=figure] { border-color:#bdb76b; background: rgba(189,183,107,0.10); }
        .rect[data-type=unknown] { border-color:#666666; background: rgba(100,100,100,0.10); }
        .rect.selected { outline: 2px dashed #111827; }
        .handle { position:absolute; width:12px; height:12px; background:#111827; border-radius:2px; }
        .handle.tl { left:-6px; top:-6px; cursor:nwse-resize; }
        .handle.tr { right:-6px; top:-6px; cursor:nesw-resize; }
        .handle.bl { left:-6px; bottom:-6px; cursor:nesw-resize; }
        .handle.br { right:-6px; bottom:-6px; cursor:nwse-resize; }
        .meta { font-size:12px; color:#334155; margin-top:6px; }
        .legend { display:flex; flex-wrap:wrap; gap:10px; margin-top:8px; }
        .legend .swatch { display:inline-block; width:14px; height:14px; border:2px solid currentColor; vertical-align:middle; margin-right:6px; }
        .slideimg { width:100%; height:auto; border-radius:6px; border:1px solid #e2e8f0; }
        .tooltip { position:absolute; background:#111827; color:#fff; padding:4px 6px; border-radius:4px; font-size:12px; pointer-events:none; transform: translate(-50%, -120%); display:none; white-space:nowrap; }
      </style>
    </head>
    <body>
      <div class=\"toolbar\">
        <a class=\"button\" href=\"{{ url_for('view_job', job_id=job_id) }}\">← Back</a>
        <a class=\"button\" href=\"{{ url_for('download_json', job_id=job_id) }}\">Download JSON</a>
        <span style=\"margin-left:auto\"></span>
        <a class=\"button\" href=\"{{ url_for('edit_slide', job_id=job_id, slide_index=prev_index) }}\">Prev</a>
        <a class=\"button\" href=\"{{ url_for('edit_slide', job_id=job_id, slide_index=next_index) }}\">Next</a>
        <button id=\"deleteBtn\" class=\"button\" style=\"background:#ef4444\">Delete selected</button>
      </div>

      <div class=\"layout\">
        <div class=\"pane\">
          <div id=\"stage\" class=\"stage\"></div>
          <div id=\"meta\" class=\"meta\"></div>
          <div class=\"legend\">
            <span>1 text</span>
            <span>2 image</span>
            <span>3 table</span>
            <span>4 chart</span>
            <span>5 shape</span>
            <span>6 connector</span>
            <span>7 figure</span>
            <span>8 unknown</span>
          </div>
        </div>
        <div class=\"pane\">
          <img class=\"slideimg\" src=\"{{ url_for('slide_image', job_id=job_id, slide_index=slide_index) }}\" alt=\"Slide image\"/>
        </div>
      </div>

      <script>
      const jobId = {{ job_id|tojson }};
      const slideIndex = {{ slide_index|tojson }};
      let slideData = null;
      let selectedId = null;

      function px(n) { return n + 'px'; }

      async function loadSlide() {
        const res = await fetch(`/api/job/${jobId}/slide/${slideIndex}`);
        if (!res.ok) { alert('Failed to load slide'); return; }
        slideData = await res.json();
        renderStage();
      }

      function createHandle(cls) {
        const h = document.createElement('div');
        h.className = 'handle ' + cls;
        return h;
      }

      function renderStage() {
        const stage = document.getElementById('stage');
        stage.innerHTML = '';
        const tip = document.createElement('div');
        tip.id = 'tooltip'; tip.className = 'tooltip';
        stage.appendChild(tip);
        const W = stage.clientWidth;
        const H = stage.clientHeight;
        const comps = slideData.components;
        document.getElementById('meta').textContent = `${comps.length} components`;
        for (const c of comps) {
          if (!c.bbox_rel) continue;
          const [x,y,w,h] = c.bbox_rel;
          const el = document.createElement('div');
          el.className = 'rect';
          el.dataset.id = c.id;
          el.dataset.type = c.type || 'unknown';
          el.style.left = Math.round(x * W) + 'px';
          el.style.top = Math.round(y * H) + 'px';
          el.style.width = Math.max(2, Math.round(w * W)) + 'px';
          el.style.height = Math.max(2, Math.round(h * H)) + 'px';
          const tl = createHandle('tl'); const tr = createHandle('tr');
          const bl = createHandle('bl'); const br = createHandle('br');
          el.appendChild(tl); el.appendChild(tr); el.appendChild(bl); el.appendChild(br);
          const code = (c.type && ({text:1,image:2,table:3,chart:4,shape:5,connector:6,figure:7,unknown:8})[c.type]) || 8;
          const badge = document.createElement('div');
          badge.style.position = 'absolute';
          badge.style.left = '0px'; badge.style.top = '0px';
          badge.style.background = 'rgba(255,255,255,0.9)';
          badge.style.border = '1px solid #111827';
          badge.style.borderBottomRightRadius = '6px';
          badge.style.padding = '1px 4px';
          badge.style.fontSize = '12px';
          badge.textContent = String(code);
          el.appendChild(badge);
          el.addEventListener('mousemove', (e) => {
            const st = c.text_style || {};
            const lines = [];
            if (st.font_pt) lines.push(`font ${st.font_pt.toFixed(1)}pt`);
            if (typeof st.bold === 'boolean') lines.push(st.bold ? 'bold' : 'normal');
            tip.textContent = lines.join(' · ') || '';
            tip.style.left = e.clientX - stage.getBoundingClientRect().left + 'px';
            tip.style.top = e.clientY - stage.getBoundingClientRect().top + 'px';
            tip.style.display = tip.textContent ? 'block' : 'none';
          });
          el.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
          stage.appendChild(el);
        }
        bindInteractions();
      }

      function bindInteractions() {
        const stage = document.getElementById('stage');
        const W = stage.clientWidth; const H = stage.clientHeight;
        let drag = null;
        stage.querySelectorAll('.rect').forEach(el => {
          el.addEventListener('mousedown', (e) => {
            document.querySelectorAll('.rect.selected').forEach(r => r.classList.remove('selected'));
            el.classList.add('selected');
            const rect = el.getBoundingClientRect();
            const start = { x: e.clientX, y: e.clientY, left: rect.left, top: rect.top, width: rect.width, height: rect.height };
            const isHandle = e.target.classList.contains('handle');
            const handle = isHandle ? (e.target.classList.contains('tl') ? 'tl' : e.target.classList.contains('tr') ? 'tr' : e.target.classList.contains('bl') ? 'bl' : 'br') : null;
            drag = { target: el, start, handle };
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp, { once: true });
            e.preventDefault();
          });
        });
        stage.addEventListener('mousedown', (e) => { if (!e.target.closest('.rect')) document.querySelectorAll('.rect.selected').forEach(r => r.classList.remove('selected')); });

        function onMove(e) {
          if (!drag) return;
          const dx = e.clientX - drag.start.x;
          const dy = e.clientY - drag.start.y;
          let left = drag.start.left + dx - stage.getBoundingClientRect().left;
          let top = drag.start.top + dy - stage.getBoundingClientRect().top;
          let width = drag.start.width;
          let height = drag.start.height;
          if (!drag.handle) {
            left = Math.max(0, Math.min(W - width, left));
            top = Math.max(0, Math.min(H - height, top));
          } else {
            if (drag.handle.includes('t')) { height = Math.max(4, Math.min(H, drag.start.height - dy)); top = Math.max(0, Math.min(drag.start.top - stage.getBoundingClientRect().top + drag.start.height - 4, drag.start.top + dy - stage.getBoundingClientRect().top)); }
            if (drag.handle.includes('b')) { height = Math.max(4, Math.min(H, drag.start.height + dy)); }
            if (drag.handle.includes('l')) { width = Math.max(4, Math.min(W, drag.start.width - dx)); left = Math.max(0, Math.min(drag.start.left - stage.getBoundingClientRect().left + drag.start.width - 4, drag.start.left + dx - stage.getBoundingClientRect().left)); }
            if (drag.handle.includes('r')) { width = Math.max(4, Math.min(W, drag.start.width + dx)); }
          }
          drag.target.style.left = left + 'px';
          drag.target.style.top = top + 'px';
          drag.target.style.width = width + 'px';
          drag.target.style.height = height + 'px';
        }
        async function onUp() {
          if (!drag) return;
          window.removeEventListener('mousemove', onMove);
          const el = drag.target; drag = null;
          const left = parseInt(el.style.left)/W; const top = parseInt(el.style.top)/H;
          const width = parseInt(el.style.width)/W; const height = parseInt(el.style.height)/H;
          await fetch(`/api/job/${jobId}/slide/${slideIndex}/component/${encodeURIComponent(el.dataset.id)}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ bbox_rel: [left, top, width, height] })
          });
        }
        document.getElementById('deleteBtn').onclick = async () => {
          const sel = document.querySelector('.rect.selected');
          if (!sel) return;
          if (!confirm('Delete selected component?')) return;
          const ok = await fetch(`/api/job/${jobId}/slide/${slideIndex}/component/${encodeURIComponent(sel.dataset.id)}`, { method: 'DELETE' });
          if (ok) loadSlide();
        };
      }

      window.addEventListener('resize', () => renderStage());
      loadSlide();
      </script>
    </body>
    </html>
    """

    def get_job_list(limit=20):
        jobs = []
        if not os.path.isdir(JOBS_DIR):
            return jobs
        job_ids = sorted(os.listdir(JOBS_DIR), reverse=True)
        for job_id in job_ids:
            jdir = os.path.join(JOBS_DIR, job_id)
            if not os.path.isdir(jdir):
                continue
            
            json_fs = []
            for f in os.listdir(jdir):
                if f.endswith('.spatial.json'):
                    json_fs.append(f)

            if not json_fs:
                continue
            json_f = json_fs[0]
            try:
                with open(os.path.join(jdir, json_f), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                pptx_name = os.path.basename(data.get('file', 'presentation.pptx'))
                num_slides = len((data or {}).get('slides', []))
                jobs.append({"job_id": job_id, "pptx_name": pptx_name, "num_slides": num_slides})
            except Exception:
                continue
            if len(jobs) >= limit:
                break
        return jobs

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML, jobs=get_job_list())

    @app.post("/upload")
    def upload():
        if "file" not in request.files:
            abort(400, "No file part in request")
        f = request.files["file"]
        if f.filename == "":
            abort(400, "No file selected")
        if not is_allowed_file(f.filename):
            abort(400, "Only .pptx files are supported")
        temp_path = os.path.join(UPLOAD_DIR, uuid.uuid4().hex + ".pptx")
        f.save(temp_path)
        job = make_job_from_upload(temp_path, f.filename)
        return redirect(url_for("view_job", job_id=job["job_id"]))

    @app.get("/job/<job_id>")
    def view_job(job_id):
        jdir = os.path.join(JOBS_DIR, job_id)
        if not os.path.isdir(jdir):
            abort(404)
        json_fs = [f for f in os.listdir(jdir) if f.endswith('.spatial.json')]
        if not json_fs:
            abort(404)
        json_f = json_fs[0]
        with open(os.path.join(jdir, json_f), 'r', encoding='utf-8') as f:
            data = json.load(f)
        pptx_name = os.path.basename(data.get('file', 'presentation.pptx'))
        num_slides = len((data or {}).get('slides', []))
        return render_template_string(JOB_HTML, job_id=job_id, pptx_name=pptx_name, num_slides=num_slides)

    @app.get("/job/<job_id>/edit/<int:sidx>")
    def edit_slide(job_id, sidx):
        jdir = os.path.join(JOBS_DIR, job_id)
        if not os.path.isdir(jdir):
            abort(404)
        data, _ = load_job_data(jdir)
        slides = data.get('slides', [])
        if sidx < 0 or sidx >= len(slides):
            abort(404)
        canvas = slides[sidx].get('canvas', {})
        pptx_name = os.path.basename(data.get('file', 'presentation.pptx'))
        return render_template_string(EDITOR_HTML,
                                      job_id=job_id,
                                      slide_index=sidx,
                                      num_slides=len(slides),
                                      prev_index=0 if sidx - 1 < 0 else sidx - 1,
                                      next_index=(len(slides) - 1) if sidx + 1 >= len(slides) else sidx + 1,
                                      pptx_name=pptx_name,
                                      canvas_w=canvas.get('w_emus', 1600),
                                      canvas_h=canvas.get('h_emus', 900))

    @app.get("/api/job/<job_id>/slide/<int:sidx>")
    def get_slide_api(job_id, sidx):
        jdir = os.path.join(JOBS_DIR, job_id)
        data, _ = load_job_data(jdir)
        slides = data.get('slides', [])
        if sidx < 0 or sidx >= len(slides):
            abort(404)
        return jsonify(slides[sidx])

    @app.post("/api/job/<job_id>/slide/<int:sidx>/component/<cid>")
    def update_component_api(job_id, sidx, cid):
        jdir = os.path.join(JOBS_DIR, job_id)
        data, json_f = load_job_data(jdir)
        slides = data.get('slides', [])
        if sidx < 0 or sidx >= len(slides):
            abort(404)
        payload = request.get_json(force=True, silent=True) or {}
        bbox = payload.get('bbox_rel')
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            abort(400, 'bad bbox')
        x,y,w,h = bbox
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))
        w = max(0.0, min(1.0 - x, float(w)))
        h = max(0.0, min(1.0 - y, float(h)))
        updated_comp = None
        for c in slides[sidx].get('components', []):
            if c.get('id') == cid:
                c['bbox_rel'] = [x,y,w,h]
                updated_comp = c
                break
        if updated_comp is None:
            abort(404)
        save_job_data(jdir, data, json_f)
        return jsonify(updated_comp)

    @app.delete("/api/job/<job_id>/slide/<int:sidx>/component/<cid>")
    def delete_component_api(job_id, sidx, cid):
        jdir = os.path.join(JOBS_DIR, job_id)
        data, json_f = load_job_data(jdir)
        slides = data.get('slides', [])
        if sidx < 0 or sidx >= len(slides):
            abort(404)
        comps = slides[sidx].get('components', [])
        before = len(comps)
        comps[:] = [c for c in comps if c.get('id') != cid]
        if len(comps) == before:
            abort(404)
        save_job_data(jdir, data, json_f)
        return ('', 204)

    @app.get("/job/<job_id>/slide_image/<int:sidx>")
    def slide_image_file(job_id, sidx):
        jdir = os.path.join(JOBS_DIR, job_id)
        img_path = find_slide_img(jdir, sidx)
        if not img_path:
            try:
                pptx_files = [f for f in os.listdir(jdir) if f.lower().endswith('.pptx')]
                if pptx_files:
                    export_slide_imgs(os.path.join(jdir, pptx_files[0]), jdir)
                    img_path = find_slide_img(jdir, sidx)
            except Exception:
                img_path = None
        if img_path:
            return send_from_directory(os.path.dirname(img_path), os.path.basename(img_path))
        filename = f"slide_{sidx:02d}.png"
        preview_dirs = [d for d in os.listdir(jdir) if d.endswith("_previews") and os.path.isdir(os.path.join(jdir, d))]
        if not preview_dirs:
            abort(404)
        preview_dir = os.path.join(jdir, preview_dirs[0])
        return send_from_directory(preview_dir, filename)

    @app.get("/job/<job_id>/json")
    def download_json_file(job_id):
        jdir = os.path.join(JOBS_DIR, job_id)
        if not os.path.isdir(jdir):
            abort(404)
        json_fs = [f for f in os.listdir(jdir) if f.endswith('.spatial.json')]
        if not json_fs:
            abort(404)
        json_f = json_fs[0]
        return send_from_directory(jdir, json_f, as_attachment=True, download_name=json_f)

    @app.get("/job/<job_id>/previews/<path:filename>")
    def preview_image_file(job_id, filename):
        jdir = os.path.join(JOBS_DIR, job_id)
        if not os.path.isdir(jdir):
            abort(404)
        preview_dirs = [d for d in os.listdir(jdir) if d.endswith("_previews") and os.path.isdir(os.path.join(jdir, d))]
        if not preview_dirs:
            abort(404)
        preview_dir = os.path.join(jdir, preview_dirs[0])
        return send_from_directory(preview_dir, filename)

    return app


def start_server(host: str = "127.0.0.1", port: int = 5000) -> None:
    # serve
    app = make_the_app()
    app.run(host=host, port=port, debug=False)


WEB_ROOT = os.path.join(os.path.dirname(__file__), "webdata")
UPLOAD_DIR = os.path.join(WEB_ROOT, "uploads")
JOBS_DIR = os.path.join(WEB_ROOT, "jobs")


def setup_web_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(JOBS_DIR, exist_ok=True)


def is_allowed_file(fname):
    return "." in fname and fname.lower().rsplit(".", 1)[-1] in {"pptx"}


def new_id():
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def make_job_from_upload(src_path, orig_name):
    job_id = new_id()
    jdir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(jdir, exist_ok=True)

    base_name = os.path.basename(orig_name)
    stored_pptx = os.path.join(jdir, base_name)
    try:
        os.replace(src_path, stored_pptx)
    except Exception:
        import shutil
        shutil.copy2(src_path, stored_pptx)

    data = do_the_processing(stored_pptx)

    json_path = os.path.join(jdir, os.path.splitext(base_name)[0] + ".spatial.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    previews_dir = os.path.join(jdir, os.path.splitext(base_name)[0] + "_previews")
    make_previews(data, previews_dir, img_w=1200, strk=2, labels=True, legend=True, groups=True)

    try:
        export_slide_imgs(stored_pptx, jdir)
    except Exception:
        pass

    return {
        "job_id": job_id,
        "job_dir": jdir,
        "pptx_name": base_name,
        "json_file": os.path.basename(json_path),
        "previews_rel": os.path.basename(previews_dir),
        "num_slides": len(data.get("slides", [])),
    }


def export_slide_imgs(pptx_file, jdir):
    import shutil, subprocess, tempfile
    odir = os.path.join(jdir, "slides_png")
    os.makedirs(odir, exist_ok=True)

    try:
        expected_count = len(Presentation(pptx_file).slides)
    except Exception:
        expected_count = None

    def find_pngs():
        try:
            return sorted([f for f in os.listdir(odir) if f.lower().endswith('.png')])
        except Exception:
            return []

    existing_pngs = find_pngs()

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not existing_pngs and soffice:
        try:
            subprocess.run([soffice, "--headless", "--convert-to", "png", "--outdir", odir, pptx_file],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        except Exception:
            pass
        existing_pngs = find_pngs()

    pdftoppm = shutil.which("pdftoppm")
    if (not existing_pngs) or (expected_count and len(existing_pngs) < expected_count):
        if soffice and pdftoppm:
            try:
                with tempfile.TemporaryDirectory() as td:
                    pdf_path = os.path.join(td, "slides.pdf")
                    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", td, pptx_file],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
                    if not os.path.isfile(pdf_path):
                        pdfs = [f for f in os.listdir(td) if f.lower().endswith('.pdf')]
                        if not pdfs: raise RuntimeError("No PDF")
                        pdf_path = os.path.join(td, pdfs[0])
                    prefix = os.path.join(odir, "slide")
                    subprocess.run([pdftoppm, "-png", "-r", "144", pdf_path, prefix],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            except Exception:
                pass
        existing_pngs = find_pngs()

    fix_png_names(odir, expected_count)


def fix_png_names(odir: str, expected: int | None) -> None:
    try:
        files = sorted([f for f in os.listdir(odir) if f.lower().endswith('.png')])
    except Exception:
        return
    if not files: return
    import re, os
    
    file_map: dict[str, int] = {}
    for f in files:
        nums = re.findall(r"(\d+)", f)
        if nums:
            try:
                n = int(nums[-1])
                file_map[f] = n
            except Exception:
                pass
    used = set()
    for f, n in file_map.items():
        target = os.path.join(odir, f"slide-{n:02d}.png")
        src = os.path.join(odir, f)
        if os.path.abspath(src) == os.path.abspath(target):
            used.add(f)
            continue
        if os.path.exists(target):
            try: os.remove(target)
            except Exception: pass
        try:
            os.replace(src, target)
            used.add(f)
        except Exception:
            pass
    try:
        remaining = sorted([f for f in os.listdir(odir) if f.lower().endswith('.png') and not re.fullmatch(r"slide-\d{2}\.png", f)])
    except Exception:
        remaining = []
    
    i = 1
    for f in remaining:
        while True:
            target = os.path.join(odir, f"slide-{i:02d}.png")
            i += 1
            if not os.path.exists(target):
                break
        src = os.path.join(odir, f)
        try: os.replace(src, target)
        except Exception: pass


def find_slide_img(jdir, sidx):
    dirp = os.path.join(jdir, 'slides_png')
    if not os.path.isdir(dirp): return None
    unp = os.path.join(dirp, f"slide-{sidx+1}.png")
    if os.path.isfile(unp): return unp
    pad = os.path.join(dirp, f"slide-{sidx+1:02d}.png")
    if os.path.isfile(pad): return pad
    return None


def crop_img(src_img_path, rel_box, out_f, pad=2):
    from PIL import Image
    with Image.open(src_img_path) as im:
        W, H = im.size
        x, y, w, h = rel_box
        x0 = max(0, int(round(x * W)) - pad)
        y0 = max(0, int(round(y * H)) - pad)
        x1 = min(W, int(round((x + w) * W)) + pad)
        y1 = min(H, int(round((y + h) * H)) + pad)
        if x1 <= x0 or y1 <= y1:
            return False
        crop = im.crop((x0, y0, x1, y1))
        os.makedirs(os.path.dirname(out_f), exist_ok=True)
        crop.save(out_f, 'PNG')
        return True
