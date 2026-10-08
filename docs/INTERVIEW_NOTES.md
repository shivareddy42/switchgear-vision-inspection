# Interview notes

The short answers below are the ones to give. The counts are the ones in this repo, not the resume targets.

## Where did the data come from?

Industrial defect data is difficult to obtain publicly. The prototype therefore uses a documented combination of publicly available industrial surface/welding datasets and publicly accessible imagery where licensing permits. Every source is tracked in sources.csv. Before a real deployment I would replace/extend that with plant-owned labeled imagery.

In this repo that combination is concrete. Twelve Wikimedia Commons files were downloaded on 2026-10-08. Seven are hand-labeled corrosion (a rusty plate, two filiform photos, rusted nails, a pitted pump, a pitted pewter plate, a corroded valve). Five are unlabeled context (switchgear, busbars, a panel, weld distortion, a spatter-removal photo) because they did not show one of the seven defects. NEU-DET, GC10-DET, KolektorSDD2, MVTec AD, RIAWELC, and LoHi-WELD were reviewed and not copied in: missing license, a form gate, radiographs, or no redistribution grant. A Kaggle weld set tagged CC0 was downloaded for inspection (2,028 JPEGs) and then excluded after a porosity-named file showed a Shutterstock watermark.

## Why augment?

Public data does not capture every shop-floor lighting, focus and camera-placement condition. Training augmentation simulates moderate changes in brightness, contrast, blur, noise, orientation and scale. Validation/test data remain untouched.

The training presentations were 5 epochs times 5 images, so 25, with the mild limits in `configs/train.yaml`. The preview grid uses Albumentations so the boxes can be checked. Augmentation did not add scratch, dent, porosity, crack, busbar, or missing-part examples. Those classes have zero labels.

## Why YOLOv8?

We need both classification and localization. YOLOv8 gives a strong speed/accuracy tradeoff and supports near-real-time edge inference.

It is not the universal best detector. A classifier cannot point at the defect. A segmenter can be tighter and costs much more labeling. Faster R-CNN is a strong alternative and is often slower at the edge. YOLOv8 was chosen because one forward pass returns class and box, the tooling exports ONNX, and that ONNX file is the handoff to TensorRT. The checkpoint here is yolov8n because the build machine has no CUDA. yolov8s is the start I would use on a GPU.

## Why not image classification?

A classifier could say a defect exists, but not show the inspector where it is.

## What does confidence mean?

The detector assigns confidence to each predicted defect. We use a threshold to decide whether a detection should trigger a failure/review. Lower thresholds improve sensitivity but increase false rejects.

In code, FAIL requires confidence strictly greater than the class threshold. Equality stays PASS.

## How would you pick the threshold?

Validation curves plus business risk. For severe defects like weld cracks or missing components I may accept more false alarms to reduce missed defects.

`configs/thresholds.yaml` is marked EXAMPLE (scratch and dent 0.55, porosity and busbar 0.45, crack and missing component 0.40, corrosion and the default 0.50). `scripts/analyze_thresholds.py` sweeps 0.10 through 0.90 on `outputs/val_predictions.json`. On this data every score is far below 0.10, so the sweep shows a missed corrosion box and zero false positives, and it says false reject is not estimable because there is no confirmed-good image. That is a data limit, not a tuned operating point.

## How did you handle limited data?

Transfer learning, careful source collection, deduplication, realistic augmentation, monitoring per-class recall, and avoiding leakage between train/test.

Transfer learning started from `yolov8n.pt`. Sources are listed with licenses. Exact SHA-256 and an 8×8 difference hash (6-bit near-duplicate rule) run before the split. The split of the 7 originals is 5/1/1, seed 42, and validation reported no cross-split duplicates. Per-class recall is empty for six classes because there are no boxes to score. I did not paper over that with synthetic defects.

## How would you improve it with real plant data?

Fixed camera/lighting, plant defect taxonomy, more rare-defect examples, operator feedback, threshold calibration, shadow deployment, model monitoring and retraining.

The production note lists the station sequence: plant images, a taxonomy signed by quality, a fixed camera and light, a golden set, severity thresholds, shadow mode, a comparison with inspectors, override, audit, drift checks, retraining when the product or the camera changes, versioned models, an OT review, and a staged rollout.

## Why edge / Jetson?

Latency, local processing, limited dependence on plant internet, reduced bandwidth, and better control of image data.

## Why ONNX?

Portable inference format between the PyTorch training environment and optimized deployment runtimes.

The export in this repo passed the ONNX checker and loaded in ONNX Runtime. Details are in `outputs/onnx_comparison.json`.

## Why TensorRT?

NVIDIA runtime optimized for fast inference on NVIDIA GPUs / Jetson.

It was not run. There is no TensorRT timing in this repository.
