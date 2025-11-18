import argparse
import json
import sys
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_SRC = REPO_ROOT / "packages" / "diffusion" / "src"
sys.path.insert(0, str(DIFFUSION_SRC))

from generate import create_layout_control_map
from validation import validate_layout, validate_palette
from infer import generate_and_mask


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate slide backgrounds from palette and layout JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--palette',
        type=str,
        required=True,
        help='Path to palette JSON file'
    )
    parser.add_argument(
        '--layout',
        type=str,
        required=True,
        help='Path to layout JSON file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs/generated_slide.png',
        help='Output path for generated image'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=12,
        help='Number of diffusion steps'
    )
    parser.add_argument(
        '--guidance',
        type=float,
        default=5.5,
        help='Guidance scale for generation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1024,
        help='Output width in pixels'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=1024,
        help='Output height in pixels'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use: cuda or cpu'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate JSON schemas'
    )

    args = parser.parse_args()



    try:
        palette = load_json(args.palette)
        print(f"Palette loaded")
    except Exception as e:
        print(f"Error loading palette: {e}")
        return 1

    try:
        layout = load_json(args.layout)
        print(f"Layout loaded")
    except Exception as e:
        print(f"Error loading layout: {e}")
        return 1

    ok_palette, errs_palette = validate_palette(palette)
    if not ok_palette:
        print(f"validation failed:")
        for err in errs_palette:
            print(f"    - {err}")
        return 1
    print(f"Palette is valid")

    ok_layout, errs_layout = validate_layout(layout)
    if not ok_layout:
        print(f"validation failed:")
        for err in errs_layout:
            print(f"    - {err}")
        return 1
    print(f"Layout is valid")

    if args.validate_only:
        return 0

    try:
        control_map, safe_zone = create_layout_control_map(layout)
        print(f"Control map created: {control_map.shape}")
        print(f"Safe zone mask created: {safe_zone.shape}")
    except Exception as e:
        print(f"Error creating control map: {e}")
        return 1

    output_path = Path(args.output)
    output_dir = output_path.parent
    output_name = output_path.name

    try:
        result_path = generate_and_mask(
            palette=palette,
            layout=layout,
            safe_zone=safe_zone,
            control_map=control_map,
            device=args.device,
            steps=args.steps,
            guidance=args.guidance,
            width=args.width,
            height=args.height,
            seed=args.seed,
            out_dir=str(output_dir),
            out_name=output_name,
        )
        print(f"Success")
        return 0
    except Exception as e:
        print(f"Error generating background: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
