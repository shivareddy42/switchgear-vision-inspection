# Model card

## Resume / project reference targets

These numbers come from the project description. This repository did not reproduce them.

- 7 defect classes in production use
- about 6,800 annotated or augmented training images
- YOLOv8 transfer learning
- mAP@0.5 around 0.93
- rare-defect recall from 71% to 89%
- false reject rate below 2.4%
- about 42 ms/frame on an NVIDIA Jetson after TensorRT
- 63% fewer manual inspection hours
- 5-month payback

## Metrics reproduced by this repository

These measurements were copied into this repository. They are not the reference targets above. Each domain is scored on its own split. The shop-inspect images are procedural renders, and a high score on them does not transfer to plant photos.

### Synthetic-domain shop-inspect (`data/shop`)

Held-out synthetic test of the shop-inspect renders only. Generator `shop-inspect-1.0.0`. 105 test images. Model `yolov8s`, 40 epochs, image size 640, seed 42, trained on MPS, evaluated on CPU. This run was not trained again here. The checkpoint is `outputs/training/yolov8_shop/weights/best.pt`. The ONNX file is `outputs/training/yolov8_shop/best.onnx`. The copied measurement is `outputs/training/yolov8_shop/evaluation_summary.json`. A labeled preview is `outputs/shop_test_preview.jpg`.

The json values below are the measurement. Rounded to four decimals they are precision 0.9839, recall 0.9826, F1 0.9832, mAP@0.5 0.9932, mAP@0.5:0.95 0.8781.

| Metric | Value |
| --- | --- |
| precision | 0.983909 |
| recall | 0.982588 |
| F1 | 0.983248 |
| mAP@0.5 | 0.993238 |
| mAP@0.5:0.95 | 0.878102 |
| test images | 105 |

Precision and recall in that table are the means of per-class precision and recall at each class's max-F1 confidence on the synthetic test split. F1 is the harmonic mean of those two fields. mAP@0.5 is mean average precision at IoU 0.50. mAP@0.5:0.95 is mean average precision over IoU 0.50:0.95.

At confidence 0.50 and IoU 0.50, the `scope=all` row of `outputs/training/yolov8_shop/threshold_analysis.csv` is 212 true positives, 2 false positives, and 4 false negatives. That row also records NMS IoU 0.70 and a confidence floor of 0.10 before the threshold is applied. It is a different operating point from the max-F1 means in the table above.

Per-class average precision on the same test split is in `evaluation_summary.json` and `per_class_metrics.csv` (`ap50`): scratch 0.995, dent 0.995, weld_porosity 0.995, weld_crack 0.994032, corrosion 0.995, misaligned_busbar 0.995, missing_or_loose_component 0.983636.

Validation mAP@0.5 was 0.990. The last epoch of `outputs/training/yolov8_shop/results.csv` (epoch 40, 105 validation images) records `metrics/mAP50(B)` 0.99071, which rounds to 0.990. That row is validation, not the held-out test.

ONNX loads. The copied summary records `onnx.checker` passed, an ONNX Runtime session opened, and a dummy forward succeeded. Input name `images`, shape `[1, 3, 640, 640]`, 44,728,911 bytes. That is not a claim that decoded boxes match, and it is not the Wikimedia head shape.

This score is synthetic-domain shop-inspect only. It is not the Wikimedia photo test, not the diagram-set smoke test, and not the reference target of about 0.93.

### Wikimedia photo domain

These numbers are the Wikimedia photo domain only. They are not the shop-inspect set and not the diagram set.

Measured on 2026-10-08. No GPU was available. The checkpoint is `outputs/training/yolov8/weights/best.pt`, started from `yolov8n.pt`. Training was 5 epochs, image size 320, batch 4, AdamW, learning rate 0.001, seed 42, CPU, workers 0. Early-stopping patience was 5 and did not fire. The run is in `outputs/training/yolov8/results.csv`.

Held-out test, 3 images (badly rusted pipes, the VIGV weld crack, and `Corroded_Bolt.jpg`), NMS IoU 0.5, from `python evaluate.py`:

| Metric | Value |
| --- | --- |
| precision | 0.007352941176470588 |
| recall | 0.25 |
| F1 | 0.014285714285714287 |
| mAP@0.5 | 0.12375 |
| mAP@0.5:0.95 | 0.06187499999999999 |
| test images | 3 |

CPU time in `outputs/evaluation_summary.json` for that pass: preprocess 0.12682933341541988 ms, inference 12.12008533335999 ms, postprocess 0.5936600000495673 ms. That is not a Jetson measurement and it is not 42 ms/frame.

Ultralytics did not return a 7-class vector, so `outputs/per_class_metrics.csv` lists the class names with empty metric cells. The aggregate numbers above are the test result. Saved test predictions top out near confidence 0.0027.

During training, Ultralytics scored the 3 validation images. The last row of `results.csv` is precision 0.00794, recall 0.5, mAP@0.5 0.27528, mAP@0.5:0.95 0.04685. The same mAP@0.5 was 0.15393 for epochs 1–3. That is a validation-loop figure, not the test result.

At the example confidence thresholds, `infer.py` on `Corroded_Bolt.jpg` returned PASS with zero detections (end-to-end 1029.5 ms on the first call, which includes process startup). The rust box is still in the label, so this PASS is a missed defect. The validation sweep from 0.10 to 0.90 misses all 4 validation boxes. False-reject rate is not estimable: the split has no confirmed-good image.

ONNX export of the same checkpoint passed `onnx.checker`. ONNX Runtime loaded the graph. The raw head shape is `(1, 11, 2100)`. On the sample image the PyTorch decoder returned no boxes. Raw ONNX output max absolute value was 368.59100341796875 and the mean absolute value was 46.737342834472656. Those are head activations, not a claim that decoded boxes match.

The model in this section is not a plant inspection model. It was trained on 16 public photos. On that photo set, weld porosity, misaligned busbar, and missing or loose component still have no labels, and the photos that do are not switchgear enclosures.

### Diagram-set smoke test (`data/synthetic`)

This is the earlier procedural set, not the shop-inspect training target. `data/synthetic/` holds 700 diagram renders, 100 per class, split 490/105/105 before training augmentation. That count is not the resume target of about 6,800 factory images. The Wikimedia checkpoint above was not retrained on these renders.

Any number from `outputs/training/yolov8_synthetic/` is a CPU smoke test on those diagrams. It is not plant performance, not a shop-inspect result, and not a reproduced mAP@0.5 of 0.93. The command is `python train.py --config configs/train_synthetic.yaml --data configs/data_synthetic.yaml --run-name yolov8_synthetic`.

One epoch, image size 320, batch 8, AdamW, learning rate 0.001, seed 42, CPU, workers 0, horizontal flip off. The single row of `outputs/training/yolov8_synthetic/results.csv` is the validation split of these diagrams (105 images): precision 0.00197, recall 0.50476, mAP@0.5 0.2585, mAP@0.5:0.95 0.16596. The Wikimedia test mAP@0.5 of 0.12375 was not replaced.

