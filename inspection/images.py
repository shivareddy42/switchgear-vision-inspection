"""Image decode helpers that tolerate odd filenames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: Path) -> np.ndarray | None:
    payload = np.fromfile(path, dtype=np.uint8)
    if payload.size == 0:
        return None
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    return image


def write_bgr(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    ext = ".jpg" if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else suffix
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    encoded.tofile(path)
