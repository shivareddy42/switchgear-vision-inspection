#!/usr/bin/env python3
"""Sweep confidence thresholds on a saved prediction file.

Input JSON is a list of images:
[{"image": "name", "gt": [{"class": "corrosion", "bbox": [x1,y1,x2,y2]}],
  "pred": [{"class": "corrosion", "confidence": 0.8, "bbox": [x1,y1,x2,y2]}]}]

Boxes are pixels. A prediction matches a ground-truth box at IoU >= 0.5.
Image-level false-reject rate is left blank when the file has no confirmed-good image.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.classes import CLASS_NAMES


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _match(gt, pred) -> tuple[int, int, int]:
    used = set()
    true_positive = 0
    for ground in gt:
        best_index = None
        best_iou = 0.5
        for index, candidate in enumerate(pred):
            if index in used or candidate["class"] != ground["class"]:
                continue
            score = _iou(ground["bbox"], candidate["bbox"])
            if score >= best_iou:
                best_iou = score
                best_index = index
        if best_index is not None:
            used.add(best_index)
            true_positive += 1
    false_positive = len(pred) - true_positive
    false_negative = len(gt) - true_positive
    return true_positive, false_positive, false_negative


def main() -> int:
    parser = argparse.ArgumentParser(description="Threshold sweep.")
    parser.add_argument("--predictions", default="outputs/val_predictions.json")
    parser.add_argument("--output", default="outputs/threshold_analysis.csv")
    args = parser.parse_args()
    source = ROOT / args.predictions
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "threshold",
                    "class_name",
                    "precision",
                    "recall",
                    "false_positives",
                    "false_negatives",
                    "false_rejects",
                    "note",
                ],
            )
            writer.writeheader()
        print(f"no predictions at {source}; wrote header-only {destination.relative_to(ROOT)}")
        return 0
    images = json.loads(source.read_text(encoding="utf-8"))
    good_images = sum(1 for image in images if not image.get("gt"))
    fields = [
        "threshold",
        "class_name",
        "precision",
        "recall",
        "false_positives",
        "false_negatives",
        "false_rejects",
        "note",
    ]
    rows = []
    thresholds = [round(step / 10, 1) for step in range(1, 10)]
    for threshold in thresholds:
        groups = {"__all__": []}
        for name in CLASS_NAMES:
            groups[name] = []
        false_rejects = 0
        for image in images:
            kept = [pred for pred in image.get("pred", []) if float(pred["confidence"]) > threshold]
            gt = image.get("gt", [])
            groups["__all__"].append(_match(gt, kept))
            for name in CLASS_NAMES:
                groups[name].append(
                    _match(
                        [box for box in gt if box["class"] == name],
                        [box for box in kept if box["class"] == name],
                    )
                )
            if not gt and kept:
                false_rejects += 1
        for name, matches in groups.items():
            tp = sum(item[0] for item in matches)
            fp = sum(item[1] for item in matches)
            fn = sum(item[2] for item in matches)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            note = ""
            reject_cell = ""
            if name == "__all__":
                if good_images == 0:
                    note = "false reject not estimable: no confirmed-good images in this file"
                else:
                    reject_cell = str(false_rejects)
            rows.append(
                {
                    "threshold": f"{threshold:.1f}",
                    "class_name": "all" if name == "__all__" else name,
                    "precision": f"{precision:.4f}",
                    "recall": f"{recall:.4f}",
                    "false_positives": fp,
                    "false_negatives": fn,
                    "false_rejects": reject_cell,
                    "note": note,
                }
            )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {destination.relative_to(ROOT)} rows {len(rows)} confirmed_good_images {good_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
