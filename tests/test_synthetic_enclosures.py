"""Synthetic enclosure renderer. No GPU and no full 700-image build."""

from inspection.classes import CLASS_NAMES
from scripts.synthesize_enclosures import assign_splits, image_seed, render_enclosure


def _mask_inside_box(mask, row) -> bool:
    class_id, cx, cy, bw, bh = row
    height, width = mask.shape
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    ys, xs = mask.nonzero()
    return bool(len(xs)) and (xs >= x1 - 1e-6).all() and (xs <= x2 + 1e-6).all() and (ys >= y1 - 1e-6).all() and (ys <= y2 + 1e-6).all() and class_id == row[0]


def test_render_is_deterministic_and_boxed_from_the_mask():
    for class_id in range(len(CLASS_NAMES)):
        image_a, mask_a, rows_a, seed_a = render_enclosure(class_id, 3, master_seed=42)
        image_b, mask_b, rows_b, seed_b = render_enclosure(class_id, 3, master_seed=42)
        assert (image_a == image_b).all()
        assert (mask_a == mask_b).all()
        assert rows_a == rows_b
        assert seed_a == seed_b == image_seed(42, class_id, 3)
        assert len(rows_a) == 1
        assert rows_a[0][0] == class_id
        assert _mask_inside_box(mask_a, rows_a[0])
        assert rows_a[0][3] < 0.95
        assert rows_a[0][4] < 0.95


def test_different_classes_are_not_the_same_picture():
    first, _, _, _ = render_enclosure(0, 0, master_seed=42)
    second, _, _, _ = render_enclosure(1, 0, master_seed=42)
    assert not (first == second).all()


def test_split_is_stratified_and_seeds_do_not_cross():
    records = []
    for class_id, name in enumerate(CLASS_NAMES):
        for index in range(100):
            records.append(
                {
                    "class_id": class_id,
                    "class_name": name,
                    "seed": image_seed(42, class_id, index),
                    "split": "",
                }
            )
    assign_splits(records, seed=42)
    counts = {}
    seeds = {"train": set(), "val": set(), "test": set()}
    for record in records:
        key = (record["class_name"], record["split"])
        counts[key] = counts.get(key, 0) + 1
        seeds[record["split"]].add(record["seed"])
    for name in CLASS_NAMES:
        assert counts[(name, "train")] == 70
        assert counts[(name, "val")] == 15
        assert counts[(name, "test")] == 15
    assert seeds["train"].isdisjoint(seeds["val"])
    assert seeds["train"].isdisjoint(seeds["test"])
    assert seeds["val"].isdisjoint(seeds["test"])
