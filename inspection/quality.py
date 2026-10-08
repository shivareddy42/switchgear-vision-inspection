"""Image quality checks used by cleaning and validation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from inspection.hashing import dhash_image, hamming, sha256_file
from inspection.images import read_bgr

MIN_SIDE = 96
MAX_ASPECT = 6.0
NEAR_DUPLICATE_BITS = 6


def assess_file(path: Path) -> dict:
    record = {
        "filename": path.name,
        "path": str(path),
        "sha256": "",
        "dhash": "",
        "width": 0,
        "height": 0,
        "status": "ok",
        "reason": "",
    }
    name = path.name.lower()
    if any(token in name for token in ("screenshot", "diagram", "slide")):
        record["status"] = "screenshot_or_diagram"
        record["reason"] = "filename looks like a screenshot or diagram"
        return record
    try:
        record["sha256"] = sha256_file(path)
    except OSError as exc:
        record["status"] = "unreadable"
        record["reason"] = str(exc)
        return record
    image = read_bgr(path)
    if image is None:
        record["status"] = "corrupt"
        record["reason"] = "OpenCV could not decode the file"
        return record
    height, width = image.shape[:2]
    record["width"] = int(width)
    record["height"] = int(height)
    record["dhash"] = f"{dhash_image(image):016x}"
    if min(width, height) < MIN_SIDE:
        record["status"] = "too_small"
        record["reason"] = f"min side {min(width, height)} < {MIN_SIDE}"
        return record
    aspect = max(width, height) / max(1, min(width, height))
    if aspect > MAX_ASPECT:
        record["status"] = "extreme_aspect"
        record["reason"] = f"aspect {aspect:.2f} > {MAX_ASPECT}"
        return record
    if _looks_like_blank_document(image):
        record["status"] = "screenshot_or_diagram"
        record["reason"] = "mostly blank, low-edge image"
        return record
    return record


def _looks_like_blank_document(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_fraction = float(np.mean(gray > 245))
    edges = cv2.Canny(gray, 80, 160)
    edge_fraction = float(np.mean(edges > 0))
    return white_fraction > 0.92 and edge_fraction < 0.015


def flag_duplicates(records: list[dict], near_bits: int = NEAR_DUPLICATE_BITS) -> list[dict]:
    """Mark exact and perceptual duplicates. The first copy stays ok."""
    seen_sha: dict[str, str] = {}
    kept_hashes: list[tuple[str, int]] = []
    flagged: list[dict] = []
    for record in records:
        current = dict(record)
        if current["status"] != "ok":
            flagged.append(current)
            continue
        sha = current["sha256"]
        if sha and sha in seen_sha:
            current["status"] = "duplicate_sha256"
            current["reason"] = f"same bytes as {seen_sha[sha]}"
            flagged.append(current)
            continue
        try:
            perceptual = int(current["dhash"], 16) if current["dhash"] else None
        except ValueError:
            perceptual = None
        if perceptual is not None:
            match = None
            for other_name, other_hash in kept_hashes:
                if hamming(perceptual, other_hash) <= near_bits:
                    match = other_name
                    break
            if match is not None:
                current["status"] = "near_duplicate"
                current["reason"] = f"dhash within {near_bits} bits of {match}"
                flagged.append(current)
                continue
            kept_hashes.append((current["filename"], perceptual))
        if sha:
            seen_sha[sha] = current["filename"]
        flagged.append(current)
    return flagged
