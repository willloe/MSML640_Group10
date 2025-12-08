# Slides-AI – Diffusion Module (`packages/diffusion/`)

This directory contains the **layout-aware diffusion module** for Slides-AI.

The goal of this component is to adapt **Stable Diffusion XL (SDXL)** with **LoRA** and layout conditioning so that it can generate slide-safe backgrounds that:

- keep title/body regions smooth and readable, and
- push more texture and structure into non-text areas.

The classifier module (under `packages/classifier/`) produces a structured layout for each slide. This module can consume those layouts (or synthetic layouts) together with a style prompt + color palette to generate final backgrounds.

---

## Folder Structure

```text
packages/diffusion/
├─ dataset/
│  ├─ images_flattened/         # Flattened DTD-style textures
│  │  ├─ Abstract_image_0.jpg
│  │  ├─ Abstract_image_1.jpg
│  │  └─ ...
│  └─ synthetic_dataset/
│     ├─ train/
│     │  ├─ synth_000000.control.png
│     │  ├─ synth_000000.safe.png
│     │  └─ ...
│     └─ test/
│        ├─ synth_00xxxx.control.png
│        ├─ synth_00xxxx.safe.png
│        └─ ...
├─ Javad/                        # Synthetic layout/mask generation + helpers
├─ src/                          # Core Python modules
│  ├─ diffusion_pipeline.py
│  ├─ lora_data.py
│  ├─ lora_train.py
│  ├─ infer.py
│  ├─ sdxl.py
│  ├─ style_labeling.py
│  ├─ synthetic.py
│  ├─ validation.py
│  ├─ control.py
│  ├─ data.py
│  ├─ evaluate.py
│  └─ generate.py
├─ Diffusion_Pipeline.ipynb      # Main notebook pipeline (training + A/B eval)
└─ README.md
```

Each synthetic pair consists of:

- `*.control.png` – a rendered **layout control map** (title/body/image/logo regions).
- `*.safe.png` – a **safe-zone mask** where text is intended to appear (should remain smooth).

Both share the same stem (e.g., `synth_000123.control.png` / `synth_000123.safe.png`).

---

## Requirements

The diffusion module uses the **root project environment**:

- Python **3.10+** (recommended)
- GPU with at least ~16 GB VRAM (tested on NVIDIA **L4** and **A100** in Colab)
- Dependencies from the root `requirements.txt`:

```bash
# from repo root (MSML640_GROUP10/)
pip install -r requirements.txt
```

Key libraries: `torch`, `diffusers`, `transformers`, `peft`, `accelerate`, `numpy`, `Pillow`, etc.

---

## Data Setup

All project datasets (classifier + diffusion) are stored in a shared Google Drive folder:

> **Google Drive:**
> https://drive.google.com/drive/folders/1SXTbQYTjI4Jkvj9YsoWpa6FiNlLGDgim

For the **diffusion module**, you should have the following structure inside `packages/diffusion/dataset/`:

```text
dataset/
├─ images_flattened/
│  ├─ Abstract_image_0.jpg
│  ├─ Abstract_image_1.jpg
│  └─ ...
└─ synthetic_dataset/
   ├─ train/
   │  ├─ synth_000000.control.png
   │  ├─ synth_000000.safe.png
   │  ├─ synth_000001.control.png
   │  ├─ synth_000001.safe.png
   │  └─ ...
   └─ test/
      ├─ synth_00xxxx.control.png
      ├─ synth_00xxxx.safe.png
      └─ ...
```

Setup steps:

1. Download the **images_flattened** textures and **synthetic_dataset** folders from Drive.
2. Place them under `packages/diffusion/dataset/` exactly as shown above.
3. If you change any folder names or locations, update the paths at the top of:
   - `Diffusion_Pipeline.ipynb`, and/or
   - `src/diffusion_pipeline.py`, `src/lora_data.py`, `src/data.py`.

---

## Running the Pipeline (Notebook – Recommended)

From the **project root**:

```bash
cd packages/diffusion
```

Then:

1. Open `Diffusion_Pipeline.ipynb` in VS Code, Jupyter Lab, or Google Colab.
2. Select a GPU runtime (e.g., L4 High RAM or A100 in Colab).
3. Run all cells from top to bottom. The notebook will:
   - Load SDXL and set up LoRA on the UNet.
   - Load textures from `dataset/images_flattened/` and synthetic layouts from `dataset/synthetic_dataset/`.
   - Train or load the Slidesafe LoRA weights.
   - Generate A/B images for several layouts and style presets.
   - Compute layout-aware metrics (safe-zone variance, WCAG-style contrast, smoothness).

Outputs (images + metrics CSV) are written to the root-level `outputs/` folder, typically under `outputs/lora_ab/`.

You can quickly test the pipeline using the `sample_layout.json` and `sample_palette.json` files at the repo root by pointing the notebook to those files.

---