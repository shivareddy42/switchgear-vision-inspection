"""Tests of the recovered renderer against the committed shop benchmark."""
from pathlib import Path

import numpy as np
import pytest
from PIL import ImageFont

from scripts.shop_inspect import generate
from scripts.shop_inspect.cli import configure_fonts, prepare_output
from scripts.shop_inspect.verify import DATASET, check_plan


def test_original_source_and_all_700_manifest_rows():
    plans, rows, counts = check_plan()
    assert counts == [196, 197, 193, 194, 195, 195, 197]
    assert len(plans) == len(rows) == 700
    assert {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")} == {
        "train": 490, "val": 105, "test": 105,
    }


def test_masks_produce_normalized_bounds():
    mask = np.zeros((640, 960), dtype=bool)
    mask[100:110, 200:220] = True
    cx, cy, width, height = generate.mask_to_yolo(mask, 960, 640)
    assert width > 0 and height > 0
    assert (cx - width / 2) * 960 <= 200
    assert (cx + width / 2) * 960 >= 219
    assert (cy - height / 2) * 640 <= 100
    assert (cy + height / 2) * 640 >= 109
    assert generate.mask_to_yolo(np.zeros_like(mask), 960, 640) is None


def test_original_validator_accepts_committed_images_and_labels():
    _, rows, _ = check_plan()
    counts, splits, unique_images = generate.validate(DATASET, rows)
    assert unique_images == 700
    assert counts.tolist() == [196, 197, 193, 194, 195, 195, 197]
    assert splits == {"train": 490, "val": 105, "test": 105}


def test_launcher_preserves_benchmark_and_existing_files(tmp_path):
    benchmark = Path(generate.__file__).resolve().parents[2] / "data" / "shop"
    for target in (benchmark, benchmark / "new"):
        with pytest.raises(ValueError):
            prepare_output(target)
    (tmp_path / "keep.txt").write_text("keep")
    with pytest.raises(ValueError):
        prepare_output(tmp_path)
    assert (tmp_path / "keep.txt").read_text() == "keep"
    assert prepare_output(tmp_path / "fresh") == tmp_path / "fresh"


def test_bundled_fonts_load():
    configure_fonts()
    ImageFont.truetype(generate.FONT_REG, 16)
    ImageFont.truetype(generate.FONT_BOLD, 22)


def test_render_is_deterministic_with_valid_labels():
    plans, _, _ = check_plan()
    image_a, labels_a, mode_a = generate.render_image(2, plans[2])
    image_b, labels_b, mode_b = generate.render_image(2, plans[2])
    assert np.array_equal(image_a, image_b)
    assert labels_a == labels_b and mode_a == mode_b
    assert image_a.shape == (640, 960, 3) and image_a.dtype == np.uint8
    assert [label[0] for label in labels_a] == sorted(plans[2])
    for _, cx, cy, width, height in labels_a:
        assert width > 0 and height > 0
        assert 0 <= cx - width / 2 <= cx + width / 2 <= 1
        assert 0 <= cy - height / 2 <= cy + height / 2 <= 1
