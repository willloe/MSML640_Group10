import argparse
import json
from pathlib import Path


def auto_generate_tags(filename):
    name = Path(filename).stem.lower()
    parts = name.replace('-', '_').replace('.', '_').split('_')
    tags = [part for part in parts if not part.isdigit() and len(part) > 1]

    return tags


def make_manifest(folder_path, output_path, auto_tag=False):
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    images = []

    for ext in image_extensions:
        images.extend(folder.glob(f'*{ext}'))
        images.extend(folder.glob(f'*{ext.upper()}'))

    images = sorted(images)

    if len(images) == 0:
        raise ValueError(f"No images: {folder_path}")

    manifest = {
        "folder": str(folder.absolute()),
        "num_images": len(images),
        "images": []
    }

    for img_path in images:
        entry = {
            "path": img_path.name,
            "tags": auto_generate_tags(img_path.name) if auto_tag else []
        }
        manifest["images"].append(entry)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w') as f:
        json.dump(manifest, f, indent=2)

    if auto_tag and len(images) > 0:
        print(f"\nTest tags:")
        for i, entry in enumerate(manifest["images"][:3]):
            print(f"  - {entry['path']}: {entry['tags']}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate a manifest JSON file for background images"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where manifest JSON saved"
    )
    parser.add_argument(
        "--auto-tag",
        action="store_true",
        help="Generate tags from filenames"
    )

    args = parser.parse_args()

    make_manifest(args.folder, args.output, args.auto_tag)


if __name__ == "__main__":
    main()
