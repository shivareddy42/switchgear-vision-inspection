"""Verify the complete plan and optionally compare independently rendered seeds."""
import argparse
import csv
import hashlib
import json
import platform
import tempfile
from pathlib import Path

import cv2
import numpy as np
import PIL

from . import generate

SOURCE_SHA256 = "602137072a30e85ad93640fa4bc21b349a897ce5538fe794c353629b1d2ae854"
DATASET = Path(__file__).resolve().parents[2] / "data" / "shop"


def check_plan(dataset=DATASET):
    source = Path(generate.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Archived generator checksum differs")
    plans, counts = generate.build_plan()
    splits = generate.assign_splits(len(plans))
    with (dataset / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(plans) or len(rows) != 700:
        raise ValueError("Expected exactly 700 plan and manifest rows")
    for seed, row in enumerate(rows):
        expected = {
            "filename": f"images/{splits[seed]}/enc_{seed:05d}.jpg",
            "seed": str(seed), "split": splits[seed],
            "class_ids": ",".join(map(str, sorted(plans[seed]))),
            "width": str(generate.IMAGE_W), "height": str(generate.IMAGE_H),
            "generator_version": generate.GENERATOR_VERSION, "synthetic": "true",
        }
        if row != expected:
            raise ValueError(f"Manifest mismatch for seed {seed}: {row} != {expected}")
    return plans, rows, counts.tolist()


def compare_seed(seed, plans, rows, dataset=DATASET):
    row = rows[seed]
    with tempfile.TemporaryDirectory() as directory:
        generate._render_job((seed, plans[seed], row["split"], directory))
        image = Path(row["filename"])
        label = Path("labels") / row["split"] / (image.stem + ".txt")
        return {
            "seed": seed,
            "image_bytes_match": (Path(directory) / image).read_bytes() == (dataset / image).read_bytes(),
            "label_bytes_match": (Path(directory) / label).read_bytes() == (dataset / label).read_bytes(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--seeds", type=int, nargs="+", help="Render selected seeds using the full original plan")
    args = parser.parse_args()
    plans, rows, counts = check_plan(args.dataset)
    if args.seeds and any(seed < 0 or seed >= len(plans) for seed in args.seeds):
        parser.error("Seeds must be in 0..699")
    samples = [compare_seed(seed, plans, rows, args.dataset) for seed in args.seeds or []]
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__,
        "source_sha256": SOURCE_SHA256, "plan_rows_matched": len(rows),
        "per_class": counts, "samples": samples,
        "scope": "Plan only" if not samples else "Plan plus listed seeds only; not a full pixel rebuild",
    }
    print(json.dumps(report, indent=2))
    return int(any(not r["image_bytes_match"] or not r["label_bytes_match"] for r in samples))


if __name__ == "__main__":
    raise SystemExit(main())
