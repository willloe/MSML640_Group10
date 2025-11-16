import argparse
import csv
import sys
import io
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def find_images(outputs_dir: Path) -> List[Path]:
    image_extensions = {'.png', '.jpg', '.jpeg'}
    images = []

    for ext in image_extensions:
        images.extend(outputs_dir.glob(f'*{ext}'))

    return sorted(images, key=lambda p: p.name)


def format_metric(value: str, decimals: int = 2) -> str:
    try:
        num = float(value)
        if decimals == 0:
            return f"{int(num)}"
        return f"{num:.{decimals}f}"
    except (ValueError, TypeError):
        return value


def generate_report(
    outputs_dir: Path,
    report_path: Path,
    metrics_csv: Optional[Path] = None,
    wcag_csv: Optional[Path] = None,
) -> None:

    layout_metrics = load_csv(metrics_csv) if metrics_csv else []
    wcag_metrics = load_csv(wcag_csv) if wcag_csv else []
    images = find_images(outputs_dir)

    layout_lookup = {m['image']: m for m in layout_metrics}
    wcag_lookup = {}
    for m in wcag_metrics:
        if m['image'] not in wcag_lookup:
            wcag_lookup[m['image']] = []
        wcag_lookup[m['image']].append(m)

    report_lines = []
    report_lines.append("Report")
    report_lines.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Images: {len(images)}")
    report_lines.append("")

    if wcag_metrics:
        report_lines.append("Summary Statistics")

        normal_passes = [float(m['pass_rate']) for m in wcag_metrics if m['text_size'] == 'normal']
        if normal_passes:
            avg_pass_rate = sum(normal_passes) / len(normal_passes)
            report_lines.append(f"Average WCAG Pass Rate for Normal Text: {avg_pass_rate:.2%}")

        if layout_metrics:
            overlaps = [float(m['reserved_overlap_percent']) for m in layout_metrics]
            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
                report_lines.append(f"Average Reserved Area Overlap: {avg_overlap:.2f}%")

            safe_zones = [float(m['safe_zone_percent']) for m in layout_metrics]
            if safe_zones:
                avg_safe = sum(safe_zones) / len(safe_zones)
                report_lines.append(f"Average Safe Zone Coverage: {avg_safe:.2f}%")


    report_lines.append("Generate Images")

    if not images:
        report_lines.append("*No images found")
    else:
        for img_path in images:
            img_name = img_path.name
            rel_path = img_path.relative_to(outputs_dir.parent) if outputs_dir.parent else img_path

            report_lines.append(f"### {img_name}")
            report_lines.append("")
            report_lines.append(f"![{img_name}]({rel_path})")
            report_lines.append("")


            if img_name in layout_lookup:
                m = layout_lookup[img_name]
                report_lines.append("Layout Safety Metrics:")
                report_lines.append("")
                report_lines.append("| Metric | Value |")
                report_lines.append("|--------|-------|")
                report_lines.append(f"| Reserved Overlap | {format_metric(m['reserved_overlap_percent'], 2)}% |")
                report_lines.append(f"| Mean Overlap Intensity | {format_metric(m['mean_overlap'], 3)} |")
                report_lines.append(f"| Safe Zone Coverage | {format_metric(m['safe_zone_percent'], 2)}% |")
                report_lines.append(f"| Reserved Area Coverage | {format_metric(m['reserved_percent'], 2)}% |")
                report_lines.append("")

            if img_name in wcag_lookup:
                report_lines.append("WCAG Readability Metrics:")
                report_lines.append("")
                report_lines.append("| Text Size | Pass Rate | Mean Contrast | Min Contrast | Max Contrast |")
                report_lines.append("|-----------|-----------|---------------|--------------|--------------|")

                for m in wcag_lookup[img_name]:
                    pass_rate = float(m['pass_rate']) * 100
                    report_lines.append(
                        f"| {m['text_size'].capitalize()} | "
                        f"{pass_rate:.1f}% ({m['passes']}/{m['total']}) | "
                        f"{format_metric(m['mean_contrast'], 2)} | "
                        f"{format_metric(m['min_contrast'], 2)} | "
                        f"{format_metric(m['max_contrast'], 2)} |"
                    )

                report_lines.append("")

            report_lines.append("---")
            report_lines.append("")

    if layout_metrics:
        report_lines.append("Complete Layout Safety Data")
        report_lines.append("")
        report_lines.append("| Image | Reserved Overlap % | Mean Overlap | Safe Zone % | Reserved % |")
        report_lines.append("|-------|-------------------|--------------|-------------|------------|")

        for m in layout_metrics:
            report_lines.append(
                f"| {m['image']} | "
                f"{format_metric(m['reserved_overlap_percent'], 2)} | "
                f"{format_metric(m['mean_overlap'], 3)} | "
                f"{format_metric(m['safe_zone_percent'], 2)} | "
                f"{format_metric(m['reserved_percent'], 2)} |"
            )

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    if wcag_metrics:
        report_lines.append("Complete WCAG Readability Data")
        report_lines.append("")
        report_lines.append("| Image | Text Size | Pass Rate | Passes/Total | Mean Contrast | Min | Max |")
        report_lines.append("|-------|-----------|-----------|--------------|---------------|-----|-----|")

        for m in wcag_metrics:
            pass_rate = float(m['pass_rate']) * 100
            report_lines.append(
                f"| {m['image']} | "
                f"{m['text_size']} | "
                f"{pass_rate:.1f}% | "
                f"{m['passes']}/{m['total']} | "
                f"{format_metric(m['mean_contrast'], 2)} | "
                f"{format_metric(m['min_contrast'], 2)} | "
                f"{format_metric(m['max_contrast'], 2)} |"
            )

        report_lines.append("")

    report_lines.append("---")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--outputs-dir',
        type=str,
        default='outputs',
        help='Directory containing generated images'
    )
    parser.add_argument(
        '--report-path',
        type=str,
        default='outputs/report.md',
        help='Output path for report'
    )
    parser.add_argument(
        '--metrics-csv',
        type=str,
        default='outputs/metrics.csv',
        help='Path to layout safety metrics CSV'
    )
    parser.add_argument(
        '--wcag-csv',
        type=str,
        default='outputs/wcag_metrics.csv',
        help='Path to WCAG metrics CSV'
    )

    args = parser.parse_args()
    print()

    outputs_dir = Path(args.outputs_dir)
    report_path = Path(args.report_path)
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else None
    wcag_csv = Path(args.wcag_csv) if args.wcag_csv else None

    print(f"Outputs directory: {outputs_dir}")
    print(f"Report output: {report_path}")
    print(f"Layout metrics: {metrics_csv if metrics_csv and metrics_csv.exists() else 'Not found'}")
    print(f"WCAG metrics: {wcag_csv if wcag_csv and wcag_csv.exists() else 'Not found'}")
    print()

    if not outputs_dir.exists():
        print(f"Error: Outputs not found: {outputs_dir}")
        return 1

    try:
        generate_report(
            outputs_dir=outputs_dir,
            report_path=report_path,
            metrics_csv=metrics_csv,
            wcag_csv=wcag_csv,
        )

        images = find_images(outputs_dir)
        print("Report complete")

        return 0
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
