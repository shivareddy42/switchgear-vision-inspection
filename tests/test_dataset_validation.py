"""Label checks, split leakage, and duplicate hashes. No GPU."""

from pathlib import Path

import cv2
import numpy as np

from inspection.augment import apply_transform, horizontal_flip_transform
from inspection.hashing import dhash_image, hamming, sha256_file
from inspection.labels import label_errors, write_yolo_label
from inspection.quality import assess_file, flag_duplicates
from inspection.validation import validate_tree


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    image = np.zeros((128, 160, 3), dtype=np.uint8)
    image[:] = color
    cv2.rectangle(image, (20, 20), (80, 90), (0, 255, 0), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def test_duplicate_sha_and_dhash(tmp_path: Path):
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    _write_image(first, (20, 20, 20))
    _write_image(second, (20, 20, 20))
    assert sha256_file(first) == sha256_file(second)
    image = cv2.imread(str(first))
    assert hamming(dhash_image(image), dhash_image(image)) == 0
    other = np.full_like(image, 255)
    assert hamming(dhash_image(image), dhash_image(other)) > 6
    records = flag_duplicates([assess_file(first), assess_file(second)])
    statuses = [row["status"] for row in records]
    assert statuses[0] == "ok"
    assert statuses[1] == "duplicate_sha256"


def test_label_errors_reject_illegal_boxes():
    assert label_errors([(0, 0.5, 0.5, 0.2, 0.2)]) == []
    problems = label_errors([(9, 0.5, 0.5, 0.0, 0.2), (0, 1.2, 0.5, 0.2, 0.2)])
    assert any("class id" in item for item in problems)
    assert any("zero-area" in item for item in problems)
    assert any("outside 0..1" in item for item in problems)


def test_validation_flags_split_leakage(tmp_path: Path):
    root = tmp_path / "data"
    for split in ("train", "val", "test"):
        _write_image(root / split / "images" / f"{split}.jpg", (10, 40, 80))
        write_yolo_label(root / split / "labels" / f"{split}.txt", [(4, 0.4, 0.4, 0.2, 0.3)])
    # Identical bytes in train and val.
    source = root / "train" / "images" / "train.jpg"
    leaked = root / "val" / "images" / "copy.jpg"
    leaked.write_bytes(source.read_bytes())
    write_yolo_label(root / "val" / "labels" / "copy.txt", [(4, 0.4, 0.4, 0.2, 0.3)])
    report = validate_tree(root)
    assert any("duplicate sha256" in error or "near-duplicate" in error for error in report["errors"])


def test_horizontal_flip_moves_the_box():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :80] = (255, 255, 255)
    boxes = [(0.2, 0.5, 0.2, 0.4)]
    transformed, new_boxes, new_ids = apply_transform(image, boxes, [4], horizontal_flip_transform())
    assert transformed.shape == image.shape
    assert new_ids == [4]
    assert len(new_boxes) == 1
    cx, cy, width, height = new_boxes[0]
    assert abs(cx - 0.8) < 1e-5
    assert abs(cy - 0.5) < 1e-5
    assert abs(width - 0.2) < 1e-5
    assert abs(height - 0.4) < 1e-5
