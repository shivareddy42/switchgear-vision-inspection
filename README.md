# Computer Vision Quality Inspection System

An end-to-end computer-vision prototype for detecting manufacturing defects on switchgear-style electrical enclosures and industrial metal assemblies. The system uses YOLOv8 for defect localization/classification, supports edge-oriented ONNX/TensorRT deployment, records inspection decisions to SQL, and exposes quality data for Power BI.

The resume numbers (about 6,800 training images, mAP@0.5 around 0.93, rare-defect recall 71% to 89%, false reject under 2.4%, 42 ms/frame on a Jetson, 63% fewer inspection hours, 5-month payback) are reference targets. They are listed again in `docs/MODEL_CARD.md`. Nothing in this repository measured them.

What this repo did measure is a 12-image public set, 7 hand-labeled corrosion boxes, a 5/1/1 split, a 5-epoch CPU run of YOLOv8n, and a test mAP of 0.0 on one image. That is a method demo, not a plant model.

## 1. Problem

A switchgear enclosure can leave the station with a scratch, a dent, porosity or a crack in a weld, corrosion, a busbar that is not seated, or a fastener that is missing. A human walk-around is slow and uneven. A camera can flag the station, but only if the model can say what the defect is and where it is, and only if the threshold matches the cost of a miss versus a false reject.

## 2. Resume project architecture

The reference design is a camera, a YOLOv8 detector trained by transfer learning, an ONNX/TensorRT export on a Jetson, a PASS/FAIL rule, a SQL log, and a Power BI extract. This repository implements that shape with public data and a CPU baseline. It does not implement the Jetson timing or the business case.

## 3. System architecture

Frame, YOLOv8 boxes, class thresholds, PASS/FAIL, SQLite (SQL Server by swapping the URL), and a flat CSV. `docs/ARCHITECTURE.md` is the short version. `infer.py --demo` runs before any custom weights exist and stamps the overlay MOCK.

## 4. Dataset sourcing / OSINT

Every candidate is a row in `data/sources.csv`, with the URL that was actually opened and the license found there. Twelve Wikimedia files were stored. Larger sets were not:

- NEU-DET, 1,800 images, no license on the official page.
- GC10-DET, license unstated (paper says 3,570 images, Dataset Ninja says 2,300; not remeasured).
- KolektorSDD2, 356 defective and 2,979 good images, license not on the official page.
- MVTec AD, CC BY-NC-SA 4.0, 5,354 images in the paper, form download, no files taken.
- RIAWELC, 24,407 radiographs, no SPDX license.
- LoHi-WELD, 3,022 images, use-with-citation, redistribution not granted.
- Kaggle weld detection tagged CC0. A snapshot had 2,028 JPEGs. A porosity-named file showed a Shutterstock watermark, so the bytes were not vendored.

`scripts/collect_images.py` downloads only the URLs in `data/source_manifest.csv`. It is not a crawler. One switchgear URL returned HTTP 429 and succeeded on retry; both rows are in `data/download_log.csv`.

## 5. Defect taxonomy

| Id | Class | In this repo |
| --- | --- | --- |
| 0 | scratch | no image |
| 1 | dent | no image |
| 2 | weld_porosity | no image |
| 3 | weld_crack | no image |
| 4 | corrosion | 7 labeled photos |
| 5 | misaligned_busbar | 1 busbar photo, unlabeled |
| 6 | missing_or_loose_component | 1 panel photo, unlabeled |

`docs/DATA_STRATEGY.md` says what each class looks like and why the empty ones stayed empty.

## 6. Cleaning and deduplication

`scripts/clean_dataset.py` checks SHA-256 duplicates, difference-hash near duplicates (6 bits), decode failures, images under 96 px on the short side, aspect ratio over 6, and screenshot-like files. The report is `outputs/data_quality_report.csv`. All 12 files passed. Originals are split before any augmentation.

## 7. Annotation

YOLO boxes, one per corrosion image, drawn on 2026-10-08 by looking at the photo. Rules are in `docs/ANNOTATION_GUIDE.md`. The busbar photo was not labeled misaligned, and the weld-distortion photo was not labeled dent. No full-frame weak boxes are used.

## 8. Train/validation/test splitting

`scripts/split_dataset.py --seed 42 --train 0.70 --val 0.15 --test 0.15` on the 7 labeled originals. Realized counts are train 5, validation 1, test 1. Near-duplicate clusters cannot cross the split. Validation found no leakage. The test image is `Rusty_steel_plate.jpg`. The validation image is `Clean_pump.jpg`.

## 9. Augmentation

Augmentation is applied only to training data. Validation and test images remain untouched except for deterministic resizing / preprocessing.

The policy is factory lighting, focus, vibration, sensor noise, a slightly skewed mount, and a small change in distance. It is brightness/contrast, gamma, rotation within 8 degrees, slight perspective, horizontal flip on surface defects only, small blur and noise, scale and translation, mild color, and a soft shadow. No mosaic, mixup, copy-paste, 90-degree rotation, or vertical flip. Reasons are in `inspection/augment.py`.

Training saw 25 presentations (5 epochs × 5 images) under `configs/train.yaml`. The preview command draws the Albumentations version of that policy, which is what was checked visually. Boxes on the filiform training image stayed put under brightness, blur, and noise, and moved under rotation, perspective, and the flipped combined panel.

Augmentation does not replace real rare-defect examples. Six classes still have no labels. Oversampling and class weights are not a substitute. Plant images are required before deployment.

## 10. Why YOLOv8

The station needs the defect type and the location. A classifier cannot show the inspector where to look. Segmentation is tighter and much more expensive to label. Faster R-CNN is strong and often slower on an edge GPU. YOLOv8 is one forward pass, with a maintained export path to ONNX and then TensorRT. It is a practical choice, not a claim that it wins every benchmark.

## 11. Training

`python train.py --config configs/train.yaml` loads `yolov8n.pt` (transfer learning). yolov8s is the better start when a GPU exists. This machine has no CUDA, so the config pins CPU, image size 320, batch 4, 5 epochs, AdamW, learning rate 0.001, seed 42, patience 5. The config is copied next to the weights. The run did not improve the validation scores across epochs.

## 12. Evaluation

`python evaluate.py` scores the test split.

| | Test (1 image) | Training-loop validation (1 image) |
| --- | --- | --- |
| precision | 0.0 | 0.04167 |
| recall | 0.0 | 1.0 |
| F1 | 0.0 | not logged by Ultralytics |
| mAP@0.5 | 0.0 | 0.06633 |
| mAP@0.5:0.95 | 0.0 | 0.00663 |

Recall means: of the real defect boxes, how many were detected. Precision means: of the boxes the model emitted, how many were real defects. A false reject is a good part called bad. A missed defect (false accept) is a defective part allowed through. The test PASS from `infer.py` is a missed defect. False reject cannot be computed: there is no confirmed-good image in the split. Rare-defect recall is not defined here, because the rare classes have no boxes.

CPU inference inside Ultralytics on the test image was 19.9 ms, plus 0.1 ms preprocess and 0.6 ms postprocess. The cold `infer.py` call was 898.8 ms. Neither is a Jetson result.

## 13. Confidence threshold and PASS/FAIL

Confidence threshold selection is not arbitrary. It is chosen using validation performance together with the cost of missed defects versus false rejects.

The default is 0.50. A detection fails the part only when its confidence is strictly above the class threshold. Example class values, labeled EXAMPLE in `configs/thresholds.yaml`: scratch 0.55, dent 0.55, weld_porosity 0.45, weld_crack 0.40, corrosion 0.50, misaligned_busbar 0.45, missing_or_loose_component 0.40. `scripts/analyze_thresholds.py` writes `outputs/threshold_analysis.csv` for 0.10, 0.20, … 0.90. On this validation file the corrosion box is missed at every threshold and false reject is marked not estimable.

## 14. Inference

`infer.py` accepts `--image`, `--dir`, `--video`, and `--webcam` (one frame; if no camera is present it says so). Output is class, box, confidence, PASS/FAIL, latency, and an image under `outputs/predictions/`. `--demo` or `--mock` does not load weights. The overlay reads `MOCK - NOT MODEL INFERENCE`.

The real test-image call returned PASS, zero detections, model version `best`.

## 15. SQL logging

SQLAlchemy models in `db/models.py`. Default URL is SQLite at `outputs/inspections.sqlite`. Set `INSPECTION_DATABASE_URL` for SQL Server. An inspection row stores id, time, image name, model version, PASS/FAIL, max confidence, milliseconds, device, the threshold note, and notes. A detection row stores class, confidence, and `x1,y1,x2,y2`. `scripts/show_recent_inspections.py` prints them. The logged test inspection is the PASS above.

## 16. Dashboard integration

`scripts/export_dashboard_data.py` writes `outputs/inspection_summary.csv` with date, hour, total inspections, passes, fails, failure rate, average milliseconds, defect class, defect count, and average confidence. There is no PBIX. A Power BI model on that extract can show defect trends, defects by class, throughput, pass/fail rate, station trends (add a station column when more than one exists), confidence, and rare-defect counts. The current file has one hour and one PASS. That PASS missed the rust. It is not a quality KPI.

## 17. ONNX / TensorRT / Jetson

`export_onnx.py` converts the PyTorch checkpoint, checks it with ONNX, and runs ONNX Runtime if it is installed. The checker passed. The raw head is shape `(1, 11, 2100)`. Comparison notes are in `outputs/onnx_comparison.json`. TensorRT and Jetson were not run. See `docs/EDGE_DEPLOYMENT.md`. 42 ms/frame stays a reference target.

## 18. Limitations

- Six of seven classes have no training image.
- The seven corrosion photos are not switchgear enclosures.
- Test and validation are one image each. mAP 0.0 and the validation mAP 0.066 are not comparable to 0.93.
- False reject is not estimable.
- The example thresholds are not calibrated.
- No Jetson, no TensorRT, no plant time study, no payback model.
- The Kaggle CC0 tag was not trusted after a Shutterstock watermark was visible.

## 19. Production deployment

`docs/PRODUCTIONIZATION.md` is the plant sequence: owned images, a taxonomy with quality, fixed camera and light, a golden set, severity thresholds, shadow mode, a human comparison, override, audit, drift monitoring, retraining, versioned models, an OT review, and a staged rollout.

## 20. Reproducibility

Seed 42. Package pins are in `requirements.txt` (CPU PyTorch index included). Commands below were run on 2026-10-08 with Python 3.12, torch 2.14.1+cpu, ultralytics 8.4.174, albumentations 2.0.8. No GPU.

## 21. Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/collect_images.py --manifest data/source_manifest.csv
python scripts/clean_dataset.py
python scripts/split_dataset.py --seed 42 --train 0.70 --val 0.15 --test 0.15
python scripts/validate_dataset.py
python scripts/dataset_stats.py

python scripts/preview_augmentations.py \
    --image data/train/images/Filiform_corrosion.jpg \
    --labels data/train/labels/Filiform_corrosion.txt \
    --output outputs/augmentation_preview.jpg

python train.py --config configs/train.yaml
python evaluate.py --weights outputs/training/yolov8/weights/best.pt --device cpu
python scripts/analyze_thresholds.py

python infer.py --image data/test/images/Rusty_steel_plate.jpg --weights outputs/training/yolov8/weights/best.pt
python infer.py --demo --image data/sample/rusty_steel_plate_preview.jpg --no-log

python scripts/show_recent_inspections.py
python scripts/export_dashboard_data.py
python export_onnx.py --weights outputs/training/yolov8/weights/best.pt --imgsz 320

uvicorn app:app --host 0.0.0.0 --port 8000
pytest
```

`GET /health` returns `{"status": "ok"}`. `POST /predict` takes a multipart image and returns `result`, `model_version`, `inference_ms`, and `detections` with `class`, `confidence`, and `bbox`. If weights are missing, the response is flagged `"mock": true`.
