#!/usr/bin/env python3
"""Split original images into train/val/test before any augmentation.

Near-duplicate clusters stay inside one split. Only images that have a YOLO
label under data/annotated are eligible. Unlabeled domain photos stay in
data/cleaned and are not treated as confirmed-good parts.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.hashing import hamming
from inspection.labels import parse_yolo_label
from inspection.quality import NEAR_DUPLICATE_BITS, assess_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        self.parent[self.find(left)] = self.find(right)


def _largest_remainder(count: int, ratios: list[float]) -> list[int]:
    raw = [count * ratio for ratio in ratios]
    floors = [int(value) for value in raw]
    leftover = count - sum(floors)
    order = sorted(range(len(ratios)), key=lambda index: (raw[index] - floors[index]), reverse=True)
    for step in range(leftover):
        floors[order[step % len(floors)]] += 1
    if count >= 3:
        for index in range(3):
            if floors[index] == 0:
                donor = max(range(3), key=lambda item: floors[item])
                if floors[donor] > 1:
                    floors[donor] -= 1
                    floors[index] += 1
    return floors


def main() -> int:
    parser = argparse.ArgumentParser(description="Split original labeled images.")
    parser.add_argument("--cleaned", default="data/cleaned")
    parser.add_argument("--annotated", default="data/annotated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--near-bits", type=int, default=NEAR_DUPLICATE_BITS)
    args = parser.parse_args()
    ratio_sum = args.train + args.val + args.test
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1, got {ratio_sum}")

    cleaned = ROOT / args.cleaned
    annotated = ROOT / args.annotated
    items = []
    for image_path in sorted(cleaned.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = annotated / image_path.parent.name / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        quality = assess_file(image_path)
        if quality["status"] != "ok":
            print(f"skip {image_path.name}: {quality['status']}")
            continue
        rows = parse_yolo_label(label_path)
        primary = image_path.parent.name
        items.append(
            {
                "image": image_path,
                "label": label_path,
                "class_name": primary,
                "sha256": quality["sha256"],
                "dhash": int(quality["dhash"], 16),
                "n_boxes": len(rows),
            }
        )
    if not items:
        raise SystemExit("no labeled images to split")

    union = UnionFind(len(items))
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if hamming(items[left]["dhash"], items[right]["dhash"]) <= args.near_bits:
                union.union(left, right)
    clusters: dict[int, list[int]] = {}
    for index in range(len(items)):
        clusters.setdefault(union.find(index), []).append(index)
    cluster_ids = sorted(clusters)
    rng = random.Random(args.seed)
    rng.shuffle(cluster_ids)
    counts = _largest_remainder(len(cluster_ids), [args.train, args.val, args.test])
    split_names = ["train", "val", "test"]
    assignment = {}
    cursor = 0
    for split_name, count in zip(split_names, counts):
        for cluster_id in cluster_ids[cursor : cursor + count]:
            assignment[cluster_id] = split_name
        cursor += count

    for split_name in split_names:
        for folder in (ROOT / "data" / split_name / "images", ROOT / "data" / split_name / "labels"):
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for cluster_id, members in clusters.items():
        split_name = assignment[cluster_id]
        for index in members:
            item = items[index]
            image_dest = ROOT / "data" / split_name / "images" / item["image"].name
            label_dest = ROOT / "data" / split_name / "labels" / f"{item['image'].stem}.txt"
            shutil.copy2(item["image"], image_dest)
            shutil.copy2(item["label"], label_dest)
            manifest_rows.append(
                {
                    "filename": item["image"].name,
                    "split": split_name,
                    "cluster_id": cluster_id,
                    "sha256": item["sha256"],
                    "class_name": item["class_name"],
                    "n_boxes": item["n_boxes"],
                }
            )
    manifest_path = ROOT / "data" / "splits.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "split", "cluster_id", "sha256", "class_name", "n_boxes"],
        )
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda row: (row["split"], row["filename"])))
    tally = {name: 0 for name in split_names}
    for row in manifest_rows:
        tally[row["split"]] += 1
    print(
        f"split seed={args.seed} train={tally['train']} val={tally['val']} test={tally['test']} "
        f"clusters={len(cluster_ids)} ratios_requested={args.train}/{args.val}/{args.test}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
