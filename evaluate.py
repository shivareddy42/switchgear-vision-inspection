#!/usr/bin/env python3
"""Evaluate a trained detector on the untouched split selected by a dataset YAML.

The manual prediction export and Ultralytics metrics are resolved from the same
--data configuration so results cannot silently mix the Wikimedia and synthetic
shop domains.
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


def _resolve_image_dir(data_yaml: Path, split: str) -> Path:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if split not in data:
        raise KeyError(f"{split!r} is not defined in {data_yaml}")

    base = Path(str(data.get("path", ".")))
    if not base.is_absolute():
        base = ROOT / base

    split_path = Path(str(data[split]))
    if not split_path.is_absolute():
        split_path = base / split_path

    split_path = split_path.resolve()
    if split_path.is_file():
        raise ValueError(
            f"{split} in {data_yaml} resolves to a file list ({split_path}). "
            "This repository's prediction export expects directory-based YOLO splits."
        )
    return split_path


def _labels_dir_for(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    indices = [index for index, value in enumerate(parts) if value == "images"]
    if not indices:
        raise ValueError(
            f"cannot infer label directory from {image_dir}; expected an 'images' path component"
        )
    parts[indices[-1]] = "labels"
    return Path(*parts)


def _image_records(data_yaml: Path, split: str, model, device: str) -> list[dict]:
    image_dir = _resolve_image_dir(data_yaml, split)
    label_dir = _labels_dir_for(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"{split} image directory does not exist: {image_dir}")

    records: list[dict] = []
    images = sorted(image_dir.iterdir())
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
        # Keep a very low detector floor here. The operating threshold is analyzed
        # later; using Ultralytics' default 0.25 would erase candidates before the
        # threshold sweep can see them.
        results = model.predict(source=image, device=device, conf=0.001, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                predictions.append(
                    {
                        "class": name,
                        "confidence": float(box.conf.item()),
                        "bbox": xyxy,
                    }
                )

        records.append(
            {
                "image": image_path.name,
                "split": split,
                "dataset": str(data_yaml.relative_to(ROOT) if data_yaml.is_relative_to(ROOT) else data_yaml),
                "gt": gt,
                "pred": predictions,
            }
        )
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
        help="Dataset YAML used for both manual prediction export and Ultralytics metrics.",
    )
    parser.add_argument(
        "--predictions-dir",
        default="outputs",
        help="Directory for val_predictions.json, test_predictions.json, summary, and per-class CSV.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    weights = (ROOT / args.weights).resolve()
    data_path = (ROOT / args.data).resolve()
    output_dir = (ROOT / args.predictions_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data_path}")

    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    iou = float(train_cfg.get("iou_threshold", 0.50))

    if not weights.exists():
        summary = {
            "status": "not_run",
            "reason": f"weights not found: {weights}",
            "metrics_reproduced": False,
            "data": str(data_path.relative_to(ROOT)),
        }
        summary_path = output_dir / "evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(summary["reason"])
        return 1

    from ultralytics import YOLO

    model = YOLO(str(weights))
    val_records = _image_records(data_path, "val", model, args.device)
    test_records = _image_records(data_path, "test", model, args.device)

    (output_dir / "val_predictions.json").write_text(
        json.dumps(val_records, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "test_predictions.json").write_text(
        json.dumps(test_records, indent=2) + "\n", encoding="utf-8"
    )

    metrics = model.val(
        data=str(data_path),
        split="test",
        device=args.device,
        iou=iou,
        plots=True,
        verbose=False,
        project=str(ROOT / "outputs" / "evaluation"),
        name=data_path.stem,
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

    csv_path = output_dir / "per_class_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "precision", "recall", "ap50", "ap50_95"],
        )
        writer.writeheader()
        writer.writerows(rows)

    _default, _per_class, threshold_status = load_thresholds(
        ROOT / "configs" / "thresholds.yaml"
    )

    summary = {
        "status": "measured",
        "metrics_reproduced": True,
        "weights": str(weights.relative_to(ROOT)),
        "data": str(data_path.relative_to(ROOT)),
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
            "false_reject": (
                "A confirmed-good part called bad. Estimable only when confirmed-good "
                "images are present in the evaluated split."
            ),
            "missed_defect_false_accept": (
                "A defective part with no detection above the operating threshold."
            ),
        },
        "threshold_file_status": threshold_status,
        "speed_ms": getattr(metrics, "speed", None),
        "note": (
            "Manual prediction exports and Ultralytics metrics were resolved from the same "
            "dataset YAML. Keep metrics from different data domains separate."
        ),
    }

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "data",
                    "precision",
                    "recall",
                    "f1",
                    "map50",
                    "map50_95",
                    "n_test_images",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
