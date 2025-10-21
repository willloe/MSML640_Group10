import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
sys.path.append(str(SRC))

from synthetic import sample_condition_batch, save_control_visuals
from validation import validate_layout, validate_palette


def main():
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = sample_condition_batch(n=2, canvas_size=(768, 1024), seed=42)

    for i, s in enumerate(samples):
        ok_l, errs_l = validate_layout(s["layout"])
        ok_p, errs_p = validate_palette(s["palette"])
        print(f"sample {i}: layout valid={ok_l}, palette valid={ok_p}")
        if errs_l:
            print(" layout errors:", errs_l)
        if errs_p:
            print(" palette errors:", errs_p)

    (out_dir / "sample_layout_validated.json").write_text(json.dumps(samples[0]["layout"], indent=2))
    save_control_visuals(samples, str(out_dir))
    print("Wrote:", out_dir / "sample_layout_validated.json")
    print("Saved control maps to:", out_dir)

if __name__ == "__main__":
    main()
