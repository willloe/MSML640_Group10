import sys
from pathlib import Path
import csv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "diffusion" / "src"
sys.path.append(str(SRC))

from evaluate import wcag_pass_rate


def main():

    outputs_dir = ROOT / "outputs"
    test_image_path = outputs_dir / "gen_masked.png"

    if not test_image_path.exists():
        print(f"Error: {test_image_path}")

        sample_img = Image.new('RGB', (1024, 768), color=(240, 240, 245))
        test_image_path = outputs_dir / "sample_test.png"
        sample_img.save(test_image_path)

    pass_rate_normal, details_normal = wcag_pass_rate(
        test_image_path,
        text_size="normal",
        sample_grid=10,
        return_details=True
    )

    print(f"Normal text pass rate: {pass_rate_normal:.2%}")
    print(f"Passes: {details_normal['passes']}/{details_normal['total']} sample points")
    print(f"Mean contrast ratio: {details_normal['mean_contrast']:.2f}")
    print(f"Min contrast ratio: {details_normal['min_contrast']:.2f}")
    print(f"Max contrast ratio: {details_normal['max_contrast']:.2f}")

    pass_rate_large, details_large = wcag_pass_rate(
        test_image_path,
        text_size="large",
        sample_grid=10,
        return_details=True
    )

    print(f"Large text pass rate: {pass_rate_large:.2%}")
    print(f"Passes: {details_large['passes']}/{details_large['total']} sample points")
    print(f"Mean contrast ratio: {details_large['mean_contrast']:.2f}")

    csv_path = outputs_dir / "wcag_metrics.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image',
            'text_size',
            'pass_rate',
            'passes',
            'total',
            'threshold',
            'mean_contrast',
            'min_contrast',
            'max_contrast'
        ])

        writer.writerow([
            test_image_path.name,
            'normal',
            f"{pass_rate_normal:.4f}",
            details_normal['passes'],
            details_normal['total'],
            details_normal['threshold'],
            f"{details_normal['mean_contrast']:.2f}",
            f"{details_normal['min_contrast']:.2f}",
            f"{details_normal['max_contrast']:.2f}"
        ])

        writer.writerow([
            test_image_path.name,
            'large',
            f"{pass_rate_large:.4f}",
            details_large['passes'],
            details_large['total'],
            details_large['threshold'],
            f"{details_large['mean_contrast']:.2f}",
            f"{details_large['min_contrast']:.2f}",
            f"{details_large['max_contrast']:.2f}"
        ])

    print("\nSUMMARY:")
    print(f"Overall pass rate for normal text: {pass_rate_normal:.2%}")
    print(f"Overall pass rate for large text:  {pass_rate_large:.2%}")


    target = 0.92
    if pass_rate_normal >= target:
        print(f"\n[PASS]: Normal text pass rate >= {target:.0%}")
    else:
        print(f"\n[FAIL]: Normal text pass rate {pass_rate_normal:.2%} < {target:.0%}")


if __name__ == "__main__":
    main()
