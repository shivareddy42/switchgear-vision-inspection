"""Exact and perceptual hashes used for deduplication."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dhash_image(image: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash. 8x8 yields a 64-bit integer."""
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    difference = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in difference.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()
