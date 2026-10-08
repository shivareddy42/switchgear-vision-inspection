# Synthetic enclosures

These files are drawn by `scripts/synthesize_enclosures.py`. They are not factory photographs, not Wikimedia files, and not the resume target of about 6,800 annotated factory images.

The 2026-10-08 run used seed 42 and 100 images per class: 700 unique JPEGs. The split of those originals, before any training augmentation, is train 490, validation 105, test 105. Each class is 70/15/15. A generation seed belongs to one image and one split.

`manifest.csv` marks every row with `domain=synthetic`. `splits/*.txt` lists paths from the repository root so training can find `images/` and the matching `labels/`. The Wikimedia copies in `data/train`, `data/val`, and `data/test` are a different split.

A metric on this folder is synthetic-domain only. It is not a reproduced mAP@0.5 of 0.93 and it is not plant performance.
