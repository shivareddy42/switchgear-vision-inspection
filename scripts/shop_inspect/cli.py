"""Portable launcher; keeps the archived renderer byte-identical."""
import argparse
from pathlib import Path

from . import generate


def prepare_output(output):
    output = Path(output).resolve()
    benchmark = Path(__file__).resolve().parents[2] / "data" / "shop"
    if output == benchmark or benchmark in output.parents:
        raise ValueError("Use a fresh output directory outside data/shop")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; the renderer deletes images/labels")
    return output


def configure_fonts():
    assets = Path(__file__).with_name("assets")
    generate.FONT_REG = str(assets / "LiberationSans-Regular.ttf")
    generate.FONT_BOLD = str(assets / "LiberationSans-Bold.ttf")


def main():
    parser = argparse.ArgumentParser(description="Regenerate the full 700-image shop benchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        output = prepare_output(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    configure_fonts()
    rows, _ = generate.generate_dataset(output, args.workers)
    generate.build_preview(rows, output, output / "preview.jpg")


if __name__ == "__main__":
    main()
