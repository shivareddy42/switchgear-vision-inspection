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

These numbers are the Wikimedia photo domain only. They are not the shop-inspect set, not the diagram set, and not the reference targets above.

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

## Diagram-set smoke test (`data/synthetic`)

This is the earlier procedural set, not the shop-inspect training target. `data/synthetic/` holds 700 diagram renders, 100 per class, split 490/105/105 before training augmentation. That count is not the resume target of about 6,800 factory images. The Wikimedia checkpoint above was not retrained on these renders.

Any number from `outputs/training/yolov8_synthetic/` is a CPU smoke test on those diagrams. It is not plant performance, not a shop-inspect result, and not a reproduced mAP@0.5 of 0.93. The command is `python train.py --config configs/train_synthetic.yaml --data configs/data_synthetic.yaml --run-name yolov8_synthetic`.

One epoch, image size 320, batch 8, AdamW, learning rate 0.001, seed 42, CPU, workers 0, horizontal flip off. The single row of `outputs/training/yolov8_synthetic/results.csv` is the validation split of these diagrams (105 images): precision 0.00197, recall 0.50476, mAP@0.5 0.2585, mAP@0.5:0.95 0.16596. The Wikimedia test mAP@0.5 of 0.12375 was not replaced.

## Shop-inspect training slot (`data/shop`)

`data/shop/` is the current synthetic training set: 700 shop-inspect renders, split 490/105/105. `configs/data.yaml` points at it. This slot is empty.

Checked on 2026-10-09: `/cursor/stores/self/training-cloud/best.pt` and `/cursor/stores/self/training-cloud/evaluation_summary.json` were not present. Nothing was copied into `outputs/training/yolov8_shop/`. No precision, recall, or mAP is recorded for this domain. Do not copy the Wikimedia 0.12375 or the diagram-set 0.2585 into this slot. Do not treat either number as plant performance or as the reference target of about 0.93.
