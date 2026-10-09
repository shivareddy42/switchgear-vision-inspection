from pathlib import Path

import yaml

from evaluate import _labels_dir_for, _resolve_image_dir


def test_labels_dir_resolution_for_both_supported_layouts(tmp_path: Path):
    assert _labels_dir_for(tmp_path / "data" / "val" / "images") == tmp_path / "data" / "val" / "labels"
    assert _labels_dir_for(tmp_path / "data" / "shop" / "images" / "test") == tmp_path / "data" / "shop" / "labels" / "test"


def test_resolve_image_dir_uses_selected_dataset_yaml(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    image_dir = dataset_root / "images" / "test"
    image_dir.mkdir(parents=True)

    config = tmp_path / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "scratch"},
            }
        ),
        encoding="utf-8",
    )

    assert _resolve_image_dir(config, "test") == image_dir.resolve()
