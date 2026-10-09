#!/usr/bin/env python3
"""Evaluate a trained detector on the untouched test split.

Also writes validation predictions for the threshold sweep. Numbers come from
this run only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.classes import CLASS_NAMES
from inspection.decision import load_thresholds
from inspection.images import read_bgr
from inspection.labels import parse_yolo_label, yolo_to_xyxy


def _image_records(split: str, model, device: str) -> list[dict]:
    image_dir = ROOT / "data" / split / "images"
    label_dir = ROOT / "data" / split / "labels"
    records = []
    images = sorted(image_dir.glob("*"))
    for image_path in images:
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        image = read_bgr(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]
        gt = []
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for class_id, cx, cy, bw, bh in parse_yolo_label(label_path):
                name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, width, height)
                gt.append({"class": name, "bbox": [x1, y1, x2, y2]})
        predictions = []
        results = model.predict(source=image, device=device, conf=0.001, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                predictions.append({"class": name, "confidence": float(box.conf.item()), "bbox": xyxy})
        records.append({"image": image_path.name, "split": split, "gt": gt, "pred": predictions})
    return records


def _float_list(value) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in list(value)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 checkpoint.")
    parser.add_argument("--weights", default="outputs/training/yolov8/weights/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--data",
        default="configs/data_wikimedia.yaml",
        help="Dataset yaml. The default is the Wikimedia photo split, not data/shop.",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    weights = ROOT / args.weights
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    iou = float(train_cfg.get("iou_threshold", 0.50))
    if not weights.exists():
        summary = {
            "status": "not_run",
            "reason": f"weights not found: {weights}",
            "metrics_reproduced": False,
        }
        path = ROOT / "outputs" / "evaluation_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(summary["reason"])
        return 1
    from ultralytics import YOLO

    model = YOLO(str(weights))
    val_records = _image_records("val", model, args.device)
    test_records = _image_records("test", model, args.device)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "val_predictions.json").write_text(json.dumps(val_records, indent=2) + "\n", encoding="utf-8")
    (ROOT / "outputs" / "test_predictions.json").write_text(json.dumps(test_records, indent=2) + "\n", encoding="utf-8")
    metrics = model.val(
        data=str(ROOT / args.data),
        split="test",
        device=args.device,
        iou=iou,
        plots=True,
        verbose=False,
        project=str(ROOT / "outputs" / "evaluation"),
        name="test",
        exist_ok=True,
    )
    box = metrics.box
    precision = float(box.mp)
    recall = float(box.mr)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    def _by_class(value):
        values = _float_list(value)
        if len(values) != len(CLASS_NAMES):
            return [""] * len(CLASS_NAMES)
        return values

    per_precision = _by_class(getattr(box, "p", None))
    per_recall = _by_class(getattr(box, "r", None))
    per_ap50 = _by_class(getattr(box, "ap50", None))
    per_ap = _by_class(getattr(box, "ap", None))
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "class_name": name,
                "precision": per_precision[index],
                "recall": per_recall[index],
                "ap50": per_ap50[index],
                "ap50_95": per_ap[index],
            }
        )
    csv_path = ROOT / "outputs" / "per_class_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "precision", "recall", "ap50", "ap50_95"])
        writer.writeheader()
        writer.writerows(rows)
    default, per_class, status = load_thresholds(ROOT / "configs" / "thresholds.yaml")
    del default, per_class
    training_val = None
    results_csv = ROOT / "outputs" / "training" / "yolov8" / "results.csv"
    if results_csv.exists():
        with results_csv.open(newline="", encoding="utf-8") as handle:
            logged = list(csv.DictReader(handle))
        if logged:
            last = logged[-1]
            training_val = {
                "split": "val",
                "n_images": len(val_records),
                "epochs": len(logged),
                "precision": float(last["metrics/precision(B)"]),
                "recall": float(last["metrics/recall(B)"]),
                "map50": float(last["metrics/mAP50(B)"]),
                "map50_95": float(last["metrics/mAP50-95(B)"]),
                "source": "outputs/training/yolov8/results.csv",
                "note": "Computed by Ultralytics on the validation split during training. Not a test-set result.",
            }
    summary = {
        "status": "measured",
        "metrics_reproduced": True,
        "weights": str(weights.relative_to(ROOT)),
        "split": "test",
        "device": args.device,
        "nms_iou": iou,
        "n_test_images": len(test_records),
        "n_val_images": len(val_records),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "definitions": {
            "recall": "Of all real defect boxes, the fraction detected.",
            "precision": "Of all predicted boxes, the fraction that match a real defect.",
            "false_reject": "A good part called bad. Not estimable unless the split contains confirmed-good images.",
            "missed_defect_false_accept": "A defective part with no detection above the operating threshold.",
        },
        "threshold_file_status": status,
        "training_loop_validation": training_val,
        "speed_ms": getattr(metrics, "speed", None),
        "confusion_matrix": None,
        "confusion_note": (
            "A class confusion matrix is not reported. The Ultralytics matrix at a very low "
            "confidence is dominated by near-zero-score boxes and does not describe operating-point "
            "errors. Use outputs/threshold_analysis.csv for false positives and false negatives at "
            "each threshold."
        ),
        "note": (
            "These figures are from this repository's test split only. "
            "They are not the resume reference targets."
        ),
    }
    summary_path = ROOT / "outputs" / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("precision", "recall", "f1", "map50", "map50_95", "n_test_images")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
