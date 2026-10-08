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

Measured on 2026-10-08. No GPU was available. The checkpoint is `outputs/training/yolov8/weights/best.pt`, started from `yolov8n.pt`. Training was 5 epochs, image size 320, batch 4, AdamW, learning rate 0.001, seed 42, CPU, workers 0. Early-stopping patience was 5 and did not fire. The run is in `outputs/training/yolov8/results.csv`.

Held-out test, one image (`Rusty_steel_plate.jpg`), NMS IoU 0.5, from `python evaluate.py`:

| Metric | Value |
| --- | --- |
| precision | 0.0 |
| recall | 0.0 |
| F1 | 0.0 |
| mAP@0.5 | 0.0 |
| mAP@0.5:0.95 | 0.0 |
| test images | 1 |

CPU time reported by Ultralytics on that test image: preprocess 0.107 ms, inference 19.901 ms, postprocess 0.635 ms. That is not a Jetson measurement and it is not 42 ms/frame.

Ultralytics did not return a 7-class vector on this one-image test, so `outputs/per_class_metrics.csv` lists the class names with empty metric cells. The aggregate numbers above are the test result.

During training, Ultralytics scored the one validation image (`Clean_pump.jpg`) the same way at every epoch: precision 0.04167, recall 1.0, mAP@0.5 0.06633, mAP@0.5:0.95 0.00663. That is a validation-loop figure, not the test result, and it did not improve across the 5 epochs.

At the example confidence thresholds, `infer.py` on the test image returned PASS with zero detections (end-to-end 898.8 ms on the first call, which includes process startup). The rust box is still in the label, so this PASS is a missed defect. False-reject rate is not estimable: the split has no confirmed-good image.

ONNX export of the same checkpoint passed `onnx.checker`. ONNX Runtime loaded the graph. The raw head shape is `(1, 11, 2100)`. On the sample image the PyTorch decoder returned no boxes. Raw ONNX output max absolute value was 361.72 and the mean absolute value was 43.63. Those are head activations, not a claim that decoded boxes match.

The model is not a plant inspection model. It was trained on five public corrosion photos that are not switchgear.
