"""Dataset validation: labels, pairings, and split leakage."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from inspection.classes import CLASS_NAMES
from inspection.hashing import hamming
from inspection.labels import label_errors, parse_yolo_label
from inspection.quality import NEAR_DUPLICATE_BITS, assess_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _images_in(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def validate_tree(root: Path, near_bits: int = NEAR_DUPLICATE_BITS) -> dict:
    """Validate ``root/{train,val,test}/{images,labels}``."""
    errors: list[str] = []
    warnings: list[str] = []
    per_split_counts: dict[str, int] = {}
    class_counts: Counter[str] = Counter()
    hashes: list[dict] = []

    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        images = _images_in(image_dir)
        per_split_counts[split] = len(images)
        label_stems = set()
        if label_dir.exists():
            label_stems = {path.stem for path in label_dir.glob("*.txt")}
        image_stems = {path.stem for path in images}
        for stem in sorted(label_stems - image_stems):
            errors.append(f"{split}: label {stem}.txt has no image")
        for image_path in images:
            if image_path.stem not in label_stems:
                errors.append(f"{split}: image {image_path.name} has no label file")
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            try:
                rows = parse_yolo_label(label_path)
            except (OSError, ValueError) as exc:
                errors.append(f"{split}: {exc}")
                continue
            for problem in label_errors(rows):
                errors.append(f"{split}/{image_path.name}: {problem}")
            for class_id, *_rest in rows:
                if 0 <= class_id < len(CLASS_NAMES):
                    class_counts[CLASS_NAMES[class_id]] += 1
            quality = assess_file(image_path)
            if quality["status"] in {"corrupt", "unreadable", "too_small", "extreme_aspect"}:
                errors.append(f"{split}/{image_path.name}: {quality['status']} ({quality['reason']})")
            hashes.append(
                {
                    "split": split,
                    "filename": image_path.name,
                    "sha256": quality["sha256"],
                    "dhash": quality["dhash"],
                    "status": quality["status"],
                }
            )

    by_sha: dict[str, list[dict]] = {}
    for item in hashes:
        if item["sha256"]:
            by_sha.setdefault(item["sha256"], []).append(item)
    for sha, group in by_sha.items():
        if len(group) > 1:
            names = ", ".join(f"{row['split']}/{row['filename']}" for row in group)
            errors.append(f"duplicate sha256 {sha[:12]} across {names}")

    for index, left in enumerate(hashes):
        if not left["dhash"] or left["status"] != "ok":
            continue
        left_hash = int(left["dhash"], 16)
        for right in hashes[index + 1 :]:
            if not right["dhash"] or right["status"] != "ok":
                continue
            if left["split"] == right["split"] and left["filename"] == right["filename"]:
                continue
            distance = hamming(left_hash, int(right["dhash"], 16))
            if distance <= near_bits and left["split"] != right["split"]:
                errors.append(
                    f"near-duplicate leakage: {left['split']}/{left['filename']} and "
                    f"{right['split']}/{right['filename']} differ by {distance} dhash bits"
                )

    for name in CLASS_NAMES:
        if class_counts[name] == 0:
            warnings.append(f"no annotations for class {name}")

    report = {
        "root": str(root),
        "train_count": per_split_counts.get("train", 0),
        "val_count": per_split_counts.get("val", 0),
        "test_count": per_split_counts.get("test", 0),
        "annotations_per_class": {name: int(class_counts[name]) for name in CLASS_NAMES},
        "errors": errors,
        "warnings": warnings,
        "near_duplicate_bits": near_bits,
    }
    return report


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
