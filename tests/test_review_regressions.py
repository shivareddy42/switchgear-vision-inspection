"""Regression checks for issues found during the final repository review."""
import sys

import numpy as np
import pytest

import train
from export_onnx import _letterbox
from inspection.decision import load_thresholds


def test_onnx_preprocessing_preserves_aspect_ratio_and_rgb():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)
    blob = _letterbox(image, 320)
    assert blob.shape == (1, 3, 320, 320)
    assert np.allclose(blob[0, :, 0, 0], 114 / 255)
    assert np.allclose(blob[0, :, 160, 160], np.array([30, 20, 10]) / 255)


def test_training_preserves_existing_run(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text("model: yolov8n.pt\n")
    (tmp_path / "dataset.yaml").write_text("path: data/shop\ntrain: images/train\nval: images/val\n")
    for split in ("train", "val"):
        path = tmp_path / "data" / "shop" / "images" / split
        path.mkdir(parents=True)
        (path / "image.jpg").write_bytes(b"placeholder")
    run = tmp_path / "outputs" / "training" / "existing"
    run.mkdir(parents=True)
    checkpoint = run / "best.pt"
    checkpoint.write_bytes(b"preserve this checkpoint")
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "config.yaml", "--data", "dataset.yaml", "--run-name", "existing"])
    assert train.main() == 1
    assert checkpoint.read_bytes() == b"preserve this checkpoint"


@pytest.mark.parametrize("value", ["-.1", "1.1", ".nan", ".inf"])
def test_invalid_operating_threshold_is_rejected(tmp_path, value):
    config = tmp_path / "thresholds.yaml"
    config.write_text(f"default_confidence_threshold: {value}\n")
    with pytest.raises(ValueError):
        load_thresholds(config)
