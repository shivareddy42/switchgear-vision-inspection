#!/usr/bin/env python3
"""Write a labeled grid: original, brightness, blur, noise, rotation, perspective, combined."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.augment import apply_transform, flip_allowed, preview_transforms
from inspection.classes import CLASS_NAMES
from inspection.images import read_bgr, write_bgr
from inspection.labels import parse_yolo_label, yolo_to_xyxy


def _draw(image: np.ndarray, boxes: list, class_ids: list[int], title: str) -> np.ndarray:
    canvas = image.copy()
    for box, class_id in zip(boxes, class_ids):
        height, width = canvas.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in yolo_to_xyxy(*box, width, height)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
        cv2.putText(canvas, name, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    banner = np.zeros((28, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, canvas])


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview augmentations with transformed boxes.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", default="outputs/augmentation_preview.jpg")
    parser.add_argument("--panel", type=int, default=360)
    args = parser.parse_args()
    image_path = Path(args.image)
    label_path = Path(args.labels)
    output_path = Path(args.output)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not label_path.is_absolute():
        label_path = ROOT / label_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    image = read_bgr(image_path)
    if image is None:
        raise SystemExit(f"could not read {image_path}")
    rows = parse_yolo_label(label_path)
    boxes = [(cx, cy, width, height) for _class_id, cx, cy, width, height in rows]
    class_ids = [class_id for class_id, *_rest in rows]
    panels = [("original", image, boxes, class_ids)]
    transforms = preview_transforms(allow_flip=flip_allowed(class_ids), seed=0)
    for name in ("brightness", "blur", "noise", "rotation", "perspective", "combined"):
        transformed, new_boxes, new_ids = apply_transform(image, boxes, class_ids, transforms[name])
        panels.append((name, transformed, new_boxes, new_ids))
    rendered = []
    for name, panel_image, panel_boxes, panel_ids in panels:
        resized = cv2.resize(panel_image, (args.panel, args.panel), interpolation=cv2.INTER_AREA)
        scale_x = args.panel / panel_image.shape[1]
        scale_y = args.panel / panel_image.shape[0]
        scaled_boxes = []
        # Boxes are normalized, so resizing the panel does not change YOLO coords.
        del scale_x, scale_y
        scaled_boxes = panel_boxes
        rendered.append(_draw(resized, scaled_boxes, panel_ids, f"{name} ({len(panel_boxes)} boxes)"))
    grid = np.hstack(rendered)
    write_bgr(output_path, grid)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
