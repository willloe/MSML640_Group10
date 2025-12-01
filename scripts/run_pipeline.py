import argparse
import subprocess
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

def run(cmd, **kwargs):
    print("\n>>", " ".join(str(c) for c in cmd))
    res = subprocess.run(cmd, check=False, text=True, **kwargs)
    if res.returncode != 0:
        raise SystemExit(f"Command failed with code {res.returncode}")
    return res

def step_prepare_data():
    data_dir = ROOT / "data" / "synthetic_dataset"
    if data_dir.exists():
        print(f"{data_dir} already exists, skipping generation.")
    else:
        run(["python", SCRIPTS / "generate_synthetic_dataset.py"])

    processed_dir = ROOT / "data" / "synthetic_dataset_processed"
    if processed_dir.exists():
        print(f"{processed_dir} already exists, skipping preprocess.")
    else:
        run(["python", SCRIPTS / "preprocess_pipeline.py"])

    run(["python", SCRIPTS / "create_splits.py"])
    run(["python", SCRIPTS / "make_manifest.py"])

def step_train_lora():
    run_dir = ROOT / "outputs" / "lora" / "runs" / "exp01"
    final_dir = run_dir / "final_lora"
    if final_dir.exists():
        print(f"{final_dir} already exists, skipping LoRA training.")
        return

    images_dir = ROOT / "data" / "images_flattened"
    captions = run_dir / "manifests" / "captions.jsonl"

    cmd = [
        "python", SCRIPTS / "train_lora.py",
        "--images_dir", str(images_dir),
        "--output_dir", str(run_dir),
        "--resolution", "512",
        "--rank", "8",
        "--batch_size", "2",
        "--gradient_accumulation_steps", "4",
        "--max_train_steps", "400",
        "--train_jsonl", str(captions),
        "--checkpoint_steps", "100",
    ]
    run(cmd)

def step_eval_ab():
    run_dir = ROOT / "outputs" / "lora" / "runs" / "exp01"
    final_lora = run_dir / "final_lora"
    out_dir = ROOT / "outputs" / "lora_ab"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", SCRIPTS / "smoke_lora_ab.py",
        "--lora_dir", str(final_lora),
        "--out_dir", str(out_dir),
        "--seed", "777",
        "--width", "1024",
        "--height", "576",
        "--steps", "28",
        "--guidance", "4.0",
        "--control_mode", "none",   # "safe" or "element"
    ]
    run(cmd)

def main():
    ap = argparse.ArgumentParser(description="End-to-end SlidesDiffusion pipeline driver")
    ap.add_argument(
        "stage",
        choices=["prepare_data", "train_lora", "eval_ab", "full"],
        help="Pipeline stage to run",
    )
    args = ap.parse_args()

    if args.stage in ("prepare_data", "full"):
        step_prepare_data()
    if args.stage in ("train_lora", "full"):
        step_train_lora()
    if args.stage in ("eval_ab", "full"):
        step_eval_ab()

if __name__ == "__main__":
    main()
