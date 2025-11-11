import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
sys.path.append(str(SRC))

from lora import build_manifest

def main():
    ap = argparse.ArgumentParser(description="Prepare captions.jsonl for LoRA training.")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_jsonl", default="outputs/lora/manifests/captions.jsonl")
    ap.add_argument("--fallback_caption", default=None)
    args = ap.parse_args()

    images_dir = Path(args.images_dir).resolve()
    out_jsonl = Path(args.out_jsonl).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    samples = build_manifest(images_dir, out_jsonl, fallback_caption=args.fallback_caption)
    print(f"Wrote {out_jsonl} with {len(samples)} samples.")

if __name__ == "__main__":
    main()
