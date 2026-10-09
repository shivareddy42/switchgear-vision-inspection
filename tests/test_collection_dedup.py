from pathlib import Path

import cv2
import numpy as np

from inspection.hashing import sha256_file
from scripts.collect_images import _existing_hashes


def test_existing_hashes_scan_stored_images_across_collection_passes(tmp_path: Path):
    output_root = tmp_path / "raw"
    image_path = output_root / "corrosion" / "sample.jpg"
    image_path.parent.mkdir(parents=True)

    image = np.zeros((128, 160, 3), dtype=np.uint8)
    image[:, :] = (30, 50, 70)
    cv2.rectangle(image, (20, 20), (100, 90), (0, 180, 220), -1)
    assert cv2.imwrite(str(image_path), image)

    shas, perceptual = _existing_hashes(tmp_path / "missing_log.csv", output_root)

    assert sha256_file(image_path) in shas
    assert len(perceptual) == 1
    assert perceptual[0][0].endswith("corrosion/sample.jpg")
