"""YOLO label IO and geometry checks.

A line is ``class_id x_center y_center width height``, all normalized to 0-1.
"""

from __future__ import annotations

from pathlib import Path

from inspection.classes import CLASS_NAMES


def parse_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number} expected 5 fields, got {len(parts)}")
        class_id = int(parts[0])
        cx, cy, width, height = (float(part) for part in parts[1:])
        rows.append((class_id, cx, cy, width, height))
    return rows


def write_yolo_label(path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}" for class_id, cx, cy, width, height in rows]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def yolo_to_xyxy(cx: float, cy: float, width: float, height: float, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x1 = (cx - width / 2.0) * image_width
    y1 = (cy - height / 2.0) * image_height
    x2 = (cx + width / 2.0) * image_width
    y2 = (cy + height / 2.0) * image_height
    return x1, y1, x2, y2


def clip_yolo_box(cx: float, cy: float, width: float, height: float) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, cx - width / 2.0)
    y1 = max(0.0, cy - height / 2.0)
    x2 = min(1.0, cx + width / 2.0)
    y2 = min(1.0, cy + height / 2.0)
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 1e-6 or clipped_h <= 1e-6:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, clipped_w, clipped_h)


def label_errors(rows: list[tuple[int, float, float, float, float]]) -> list[str]:
    errors: list[str] = []
    for index, (class_id, cx, cy, width, height) in enumerate(rows):
        where = f"box {index}"
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            errors.append(f"{where}: class id {class_id} is outside 0..{len(CLASS_NAMES) - 1}")
        if width <= 0 or height <= 0:
            errors.append(f"{where}: zero-area box")
            continue
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
            errors.append(f"{where}: coordinate outside 0..1")
        if cx - width / 2.0 < -1e-6 or cy - height / 2.0 < -1e-6:
            errors.append(f"{where}: box extends above or left of the image")
        if cx + width / 2.0 > 1.0 + 1e-6 or cy + height / 2.0 > 1.0 + 1e-6:
            errors.append(f"{where}: box extends below or right of the image")
    return errors
