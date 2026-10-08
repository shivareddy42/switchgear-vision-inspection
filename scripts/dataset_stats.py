#!/usr/bin/env python3
"""Print unique-image and split counts. Does not invent augmented totals."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.augment import AUGMENTATION_POLICY
from inspection.classes import CLASS_NAMES
from inspection.labels import parse_yolo_label

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_images(directory: Path) -> Counter:
    counts: Counter = Counter()
    if not directory.exists():
        return counts
    for path in directory.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            counts[path.parent.name] += 1
    return counts


def main() -> int:
    raw_counts = _count_images(ROOT / "data" / "raw")
    cleaned_counts = _count_images(ROOT / "data" / "cleaned")
    annotation_counts: Counter = Counter()
    annotated = ROOT / "data" / "annotated"
    if annotated.exists():
        for path in annotated.rglob("*.txt"):
            try:
                rows = parse_yolo_label(path)
            except (OSError, ValueError):
                continue
            for class_id, *_rest in rows:
                if 0 <= class_id < len(CLASS_NAMES):
                    annotation_counts[CLASS_NAMES[class_id]] += 1
    splits = Counter()
    splits_path = ROOT / "data" / "splits.csv"
    if splits_path.exists():
        with splits_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                splits[row["split"]] += 1
    unique = sum(cleaned_counts.values())
    print(f"unique_cleaned_images {unique}")
    print("images_per_class_cleaned")
    for name in sorted(set(CLASS_NAMES) | set(cleaned_counts)):
        print(f"  {name}: {cleaned_counts.get(name, 0)}")
    print("raw_images_per_folder")
    for name, count in sorted(raw_counts.items()):
        print(f"  {name}: {count}")
    print("annotations_per_class")
    for name in CLASS_NAMES:
        print(f"  {name}: {annotation_counts.get(name, 0)}")
    print(f"train {splits.get('train', 0)} validation {splits.get('val', 0)} test {splits.get('test', 0)}")
    print("augmentation_policy")
    for item in AUGMENTATION_POLICY:
        print(f"  {item['name']}: {item['limits']} — {item['reason']}")
    print(
        "training_samples_seen_after_augmentation: not estimated here. "
        "Read the training log. This script does not multiply images by a guessed factor."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
