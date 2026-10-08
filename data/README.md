# Data

`raw/` holds images downloaded by `scripts/collect_images.py`. The folder name is the review decision, not a claim that a public dataset already used that class.

`cleaned/` is the same bytes after `scripts/clean_dataset.py`. Nothing was dropped in the current run.

`annotated/` holds YOLO labels written by hand on 2026-10-08 after looking at the photos. `scratch/`, `dent/`, `weld_crack/`, and `corrosion/` have labels. Weld porosity, misaligned busbar, and missing or loose component do not. A line is `class_id x_center y_center width height`, normalized.

`train/`, `val/`, and `test/` are copies of those labeled originals. They were split before augmentation. Seed 42, requested 70/15/15, realized 16/3/3 because there are 22 labeled images.

`unassigned` images in `raw/` and `cleaned/` are domain photos that were not given a defect box. They are not in the split and they are not treated as confirmed-good parts.

`sample/` is a resized preview of the CC0 rusty-plate photo for demos.

`sources.csv` is the source register. `source_manifest.csv` is the download list. `download_log.csv` is what the collector actually did. `splits.csv` records the split.

Do not add augmented copies here and then split them. That leaks.
