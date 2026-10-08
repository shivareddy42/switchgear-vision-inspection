#!/usr/bin/env python3
"""Train YOLOv8 from configs/train.yaml.

Uses the pretrained checkpoint named in the config (transfer learning).
This machine has no CUDA device, so the config selects yolov8n and CPU.
A GPU run can switch the model to yolov8s and device to 0.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8.")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    os.chdir(ROOT)
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_images = _count_images(ROOT / "data" / "train" / "images")
    val_images = _count_images(ROOT / "data" / "val" / "images")
    if train_images == 0 or val_images == 0:
        print(
            "Refusing to train: data/train/images or data/val/images is empty. "
            "Split labeled originals first. No metric will be invented."
        )
        return 1
    import torch
    from ultralytics import YOLO

    device = config.get("device", "cpu")
    if device != "cpu" and not torch.cuda.is_available():
        print(f"CUDA is not available. Overriding device {device!r} to cpu.")
        device = "cpu"
    print(f"torch {torch.__version__} cuda {torch.cuda.is_available()} device {device}")
    print(f"train_images {train_images} val_images {val_images} model {config['model']}")
    model = YOLO(config["model"])
    run_dir = ROOT / "outputs" / "training" / "yolov8"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    model.train(
        data=str(ROOT / "configs" / "data.yaml"),
        epochs=int(config["epochs"]),
        batch=min(int(config["batch"]), train_images),
        imgsz=int(config["imgsz"]),
        device=device,
        workers=int(config["workers"]),
        patience=int(config["patience"]),
        seed=int(config["seed"]),
        optimizer=config["optimizer"],
        lr0=float(config["lr"]),
        mosaic=float(config.get("mosaic", 0.0)),
        mixup=float(config.get("mixup", 0.0)),
        copy_paste=float(config.get("copy_paste", 0.0)),
        degrees=float(config.get("degrees", 0.0)),
        translate=float(config.get("translate", 0.0)),
        scale=float(config.get("scale", 0.0)),
        shear=float(config.get("shear", 0.0)),
        perspective=float(config.get("perspective", 0.0)),
        fliplr=float(config.get("fliplr", 0.0)),
        flipud=float(config.get("flipud", 0.0)),
        hsv_h=float(config.get("hsv_h", 0.0)),
        hsv_s=float(config.get("hsv_s", 0.0)),
        hsv_v=float(config.get("hsv_v", 0.0)),
        erasing=0.0,
        auto_augment=None,
        project=str(ROOT / "outputs" / "training"),
        name="yolov8",
        exist_ok=True,
        pretrained=True,
        plots=True,
        verbose=True,
    )
    saved = run_dir / "train_config_used.yaml"
    saved.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"saved config copy {saved.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
