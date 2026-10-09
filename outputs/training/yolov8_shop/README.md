# Shop-inspect synthetic-domain test

These files are the held-out test of the procedural shop-inspect renders in `data/shop/` only. Generator `shop-inspect-1.0.0`. They are not Wikimedia photos, not the diagram set in `data/synthetic/`, and not plant photographs. A high score on these renders does not transfer to plant photos.

The renders are procedural. This result is not the resume reference target of mAP@0.5 around 0.93, not about 6,800 factory images, and not a Jetson timing of about 42 ms/frame. Those stay under "Resume / project reference targets" in `docs/MODEL_CARD.md`.

Copied artifacts:

- `weights/best.pt` — yolov8s checkpoint, 40 epochs, image size 640, seed 42, trained on MPS
- `best.onnx` — exported graph next to this file, not under `weights/`
- `evaluation_summary.json` — held-out synthetic test, 105 images, evaluated on CPU
- `per_class_metrics.csv` — per-class precision, recall, F1, and average precision on that test split
- `threshold_analysis.csv` — confidence sweep on that test split
- `results.csv` — training-loop log; the last row is validation, not test
- `outputs/shop_test_preview.jpg` — labeled preview of the test pass

Held-out test, from `evaluation_summary.json` (the same measurement rounded to four decimals is precision 0.9839, recall 0.9826, F1 0.9832, mAP@0.5 0.9932, mAP@0.5:0.95 0.8781):

| Metric | Value |
| --- | --- |
| precision | 0.983909 |
| recall | 0.982588 |
| F1 | 0.983248 |
| mAP@0.5 | 0.993238 |
| mAP@0.5:0.95 | 0.878102 |
| test images | 105 |

Precision and recall there are means of per-class precision and recall at each class's max-F1 confidence. At confidence 0.50 and IoU 0.50, the `scope=all` row of `threshold_analysis.csv` is 2 false positives and 4 false negatives (212 true positives). That row also records NMS IoU 0.70. Per-class average precision is the `ap50` column in the json and in `per_class_metrics.csv`.

Validation mAP@0.5 was 0.99071 (0.991 rounded to three decimals). Epoch 40 of `results.csv` records `metrics/mAP50(B)` 0.99071 on the 105 validation images. That row is validation, not the held-out test.

ONNX loads. The summary records a passed checker, an ONNX Runtime session, and a successful dummy forward. Input shape `[1, 3, 640, 640]`. This run was not trained again in this repository.
