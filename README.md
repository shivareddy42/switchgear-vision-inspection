# Computer Vision Quality Inspection System

An end-to-end computer-vision prototype for detecting manufacturing defects on switchgear-style electrical enclosures and industrial metal assemblies. The system uses YOLOv8 for defect localization/classification, supports edge-oriented ONNX/TensorRT deployment, records inspection decisions to SQL, and exposes quality data for Power BI.

The resume numbers (about 6,800 training images, mAP@0.5 around 0.93, rare-defect recall 71% to 89%, false reject under 2.4%, 42 ms/frame on a Jetson, 63% fewer inspection hours, 5-month payback) are reference targets. They are listed again in `docs/MODEL_CARD.md`. Nothing in this repository measured them.

Four layers sit next to each other. They are not one dataset and they are not one metric.

1. Wikimedia photos in `data/raw` and the 16/3/3 labeled split. Real photographs, a small set: 27 files, 22 hand-labeled, 5 unlabeled. Most classes have one or two photos or none. The only photo-domain test measurement is mAP@0.5 0.12375 on 3 images.
2. Diagram renders in `data/synthetic/`. An earlier procedural set, 100 images per class, split 490/105/105. It is not the training target. Its 1-epoch smoke (precision 0.00197, recall 0.50476, mAP@0.5 0.2585, mAP@0.5:0.95 0.16596) stays in `outputs/training/yolov8_synthetic/` and is not a shop or plant result.
3. Shop-inspect renders in `data/shop/`. The current synthetic training set: 700 images, split 490/105/105, generator `shop-inspect-1.0.0`. `configs/data.yaml` points here. No shop metric is recorded yet.
4. Resume reference targets (about 6,800 factory images, mAP@0.5 around 0.93, and the rest of that list). Still unmeasured.

## 1. Problem

A switchgear enclosure can leave the station with a scratch, a dent, porosity or a crack in a weld, corrosion, a busbar that is not seated, or a fastener that is missing. A human walk-around is slow and uneven. A camera can flag the station, but only if the model can say what the defect is and where it is, and only if the threshold matches the cost of a miss versus a false reject.

## 2. Resume project architecture

The reference design is a camera, a YOLOv8 detector trained by transfer learning, an ONNX/TensorRT export on a Jetson, a PASS/FAIL rule, a SQL log, and a Power BI extract. This repository implements that shape with public data and a CPU baseline. It does not implement the Jetson timing or the business case.

## 3. System architecture

Frame, YOLOv8 boxes, class thresholds, PASS/FAIL, SQLite (SQL Server by swapping the URL), and a flat CSV. `docs/ARCHITECTURE.md` is the short version. `infer.py --demo` runs before any custom weights exist and stamps the overlay MOCK.

## 4. Dataset sourcing / OSINT

Every candidate is a row in `data/sources.csv`, with the URL that was actually opened and the license found there. Twenty-seven Wikimedia files were stored. A second Commons search on 2026-10-08 added scratch, dent, weld-crack, and more corrosion photos. Larger sets were not:

- NEU-DET, 1,800 images, no license on the official page.
- GC10-DET, license unstated (paper says 3,570 images, Dataset Ninja says 2,300; not remeasured).
- KolektorSDD2, 356 defective and 2,979 good images, license not on the official page.
- MVTec AD, CC BY-NC-SA 4.0, 5,354 images in the paper, form download, no files taken.
- RIAWELC, 24,407 radiographs, no SPDX license.
- LoHi-WELD, 3,022 images, use-with-citation, redistribution not granted.
- Kaggle weld detection tagged CC0. A snapshot had 2,028 JPEGs. A porosity-named file showed a Shutterstock watermark, so the bytes were not vendored.

`scripts/collect_images.py` downloads only the URLs in a manifest. It is not a crawler. The first pass used `data/source_manifest.csv`. The second pass used `data/source_manifest_pass2.csv`. Two URLs returned HTTP 429 and succeeded on retry (the switchgear photo, and the Longfellow Bridge beams). Both attempts are in `data/download_log.csv`.

## 5. Defect taxonomy

| Id | Class | In this repo |
| --- | --- | --- |
| 0 | scratch | 1 labeled photo |
| 1 | dent | 2 labeled photos |
| 2 | weld_porosity | no image |
| 3 | weld_crack | 2 labeled photos |
| 4 | corrosion | 17 labeled photos |
| 5 | misaligned_busbar | 1 busbar photo, unlabeled |
| 6 | missing_or_loose_component | 1 panel photo, unlabeled |

`docs/DATA_STRATEGY.md` says what each class looks like and why the empty photo classes stayed empty. The table above is the Wikimedia set only. `data/synthetic/` is the earlier diagram set. `data/shop/` is the shop-inspect training set. Neither count is additional factory images.

## 6. Cleaning and deduplication

`scripts/clean_dataset.py` checks SHA-256 duplicates, difference-hash near duplicates (6 bits), decode failures, images under 96 px on the short side, aspect ratio over 6, and screenshot-like files. The report is `outputs/data_quality_report.csv`. All 27 files passed. Originals are split before any augmentation.

## 7. Annotation

YOLO boxes, drawn on 2026-10-08 by looking at the photo and then drawn back on to check the fit. Rules are in `docs/ANNOTATION_GUIDE.md`. The busbar photo was not labeled misaligned, the weld-distortion photo was not labeled dent, and undercut was not labeled as a crack. No full-frame weak boxes are used. Box counts are scratch 1, dent 4, weld crack 2, corrosion 18.

## 8. Train/validation/test splitting

`scripts/split_dataset.py --seed 42 --train 0.70 --val 0.15 --test 0.15` on the 22 labeled originals. Realized counts are train 16, validation 3, test 3. Near-duplicate clusters cannot cross the split. Validation found no leakage. The test files are the badly rusted pipes, the VIGV weld crack, and `Corroded_Bolt.jpg`. The validation files are the dented field gates, the Longfellow Bridge beams, and the Owalla valve.

## 9. Augmentation

Augmentation is applied only to training data. Validation and test images remain untouched except for deterministic resizing / preprocessing.

The policy is factory lighting, focus, vibration, sensor noise, a slightly skewed mount, and a small change in distance. It is brightness/contrast, gamma, rotation within 8 degrees, slight perspective, horizontal flip on surface defects only, small blur and noise, scale and translation, mild color, and a soft shadow. No mosaic, mixup, copy-paste, 90-degree rotation, or vertical flip. Reasons are in `inspection/augment.py`.

Training saw 80 presentations (5 epochs × 16 images) under `configs/train.yaml`. The preview command draws the Albumentations version of that policy, which is what was checked visually. On `Metal_Dented_Defect.jpg` both boxes stayed at `(0.24, 0.45, 0.32, 0.34)` and `(0.74, 0.42, 0.32, 0.40)` under brightness, blur, and noise. Rotation and perspective moved them. The combined panel, which flips, moved the centers to about `0.82` and `0.27`.

Augmentation does not replace real rare-defect examples. On the Wikimedia photos, weld porosity, busbar misalignment, and missing hardware still have no labels. Diagram renders and shop-inspect renders draw those classes in their own folders. Oversampling and class weights are not a substitute. Plant images are required before deployment.

## 10. Why YOLOv8

The station needs the defect type and the location. A classifier cannot show the inspector where to look. Segmentation is tighter and much more expensive to label. Faster R-CNN is strong and often slower on an edge GPU. YOLOv8 is one forward pass, with a maintained export path to ONNX and then TensorRT. It is a practical choice, not a claim that it wins every benchmark.

## 11. Training

The photo run was `python train.py --config configs/train.yaml --data configs/data_wikimedia.yaml --run-name yolov8`. It loads `yolov8n.pt` (transfer learning). yolov8s is the better start when a GPU exists. This machine has no CUDA, so the config pins CPU, image size 320, batch 4, 5 epochs, AdamW, learning rate 0.001, seed 42, patience 5. On the Wikimedia validation split, mAP@0.5 in `outputs/training/yolov8/results.csv` stayed at 0.15393 for the first three epochs, rose to 0.27528 at epoch 4, and stayed there at epoch 5.

`configs/data.yaml` now points at `data/shop` for the next train. That run has not been started here. A cloud agent is training separately. When it is scored, the numbers belong in `outputs/training/yolov8_shop/` and stay labeled shop-inspect synthetic, not Wikimedia and not 0.93.

## 12. Evaluation

`python evaluate.py` scores the test split.

| | Test (3 images) | Training-loop validation (3 images) |
| --- | --- | --- |
| precision | 0.007352941176470588 | 0.00794 |
| recall | 0.25 | 0.5 |
| F1 | 0.014285714285714287 | not logged by Ultralytics |
| mAP@0.5 | 0.12375 | 0.27528 |
| mAP@0.5:0.95 | 0.06187499999999999 | 0.04685 |

Recall means: of the real defect boxes, how many were detected. Precision means: of the boxes the model emitted, how many were real defects. A false reject is a good part called bad. A missed defect (false accept) is a defective part allowed through. `infer.py` on `Corroded_Bolt.jpg` returned PASS with zero detections, so that PASS is a missed defect. The highest confidence written into `outputs/test_predictions.json` is about 0.0027, which is why mAP at the Ultralytics matching threshold and the PASS/FAIL rule at 0.50 do not tell the same story. False reject cannot be computed: there is no confirmed-good image in the split. `outputs/per_class_metrics.csv` has empty metric cells because Ultralytics did not return a 7-class vector.

CPU time in `outputs/evaluation_summary.json` for that test pass: preprocess 0.12682933341541988 ms, inference 12.12008533335999 ms, postprocess 0.5936600000495673 ms. The cold `infer.py` call on the corroded bolt was 1029.5 ms. Neither is a Jetson result.

## 13. Confidence threshold and PASS/FAIL

Confidence threshold selection is not arbitrary. It is chosen using validation performance together with the cost of missed defects versus false rejects.

The default is 0.50. A detection fails the part only when its confidence is strictly above the class threshold. Example class values, labeled EXAMPLE in `configs/thresholds.yaml`: scratch 0.55, dent 0.55, weld_porosity 0.45, weld_crack 0.40, corrosion 0.50, misaligned_busbar 0.45, missing_or_loose_component 0.40. `scripts/analyze_thresholds.py` writes `outputs/threshold_analysis.csv` for 0.10, 0.20, … 0.90. On the 3 validation images every threshold from 0.10 through 0.90 misses all 4 labeled boxes, and false reject is marked not estimable.

## 14. Inference

`infer.py` accepts `--image`, `--dir`, `--video`, and `--webcam` (one frame; if no camera is present it says so). Output is class, box, confidence, PASS/FAIL, latency, and an image under `outputs/predictions/`. `--demo` or `--mock` does not load weights. The overlay reads `MOCK - NOT MODEL INFERENCE`.

The call on `data/test/images/Corroded_Bolt.jpg` returned PASS, zero detections, model version `best`.

## 15. SQL logging

SQLAlchemy models in `db/models.py`. Default URL is SQLite at `outputs/inspections.sqlite`. Set `INSPECTION_DATABASE_URL` for SQL Server. An inspection row stores id, time, image name, model version, PASS/FAIL, max confidence, milliseconds, device, the threshold note, and notes. A detection row stores class, confidence, and `x1,y1,x2,y2`. `scripts/show_recent_inspections.py` prints them. The logged test inspection is the PASS above.

## 16. Dashboard integration

`scripts/export_dashboard_data.py` writes `outputs/inspection_summary.csv` with date, hour, total inspections, passes, fails, failure rate, average milliseconds, defect class, defect count, and average confidence. There is no PBIX. A Power BI model on that extract can show defect trends, defects by class, throughput, pass/fail rate, station trends (add a station column when more than one exists), confidence, and rare-defect counts. The current file has one hour, 2 inspections, 2 passes, 0 fails, failure rate 0.0000, and average inference 964.140 ms. Both passes have no detection. One is an earlier run on the rusty plate and one is this run on the corroded bolt. Neither is a quality KPI.

## 17. ONNX / TensorRT / Jetson

`export_onnx.py` converts the PyTorch checkpoint, checks it with ONNX, and runs ONNX Runtime if it is installed. The checker passed. The raw head is shape `(1, 11, 2100)`. Comparison notes are in `outputs/onnx_comparison.json`. TensorRT and Jetson were not run. See `docs/EDGE_DEPLOYMENT.md`. 42 ms/frame stays a reference target.

## 18. Limitations

- On the Wikimedia photos, weld porosity, misaligned busbar, and missing hardware still have no image. Scratch, dent, and weld crack have one or two photos, and they are not enclosures.
- `data/synthetic/` is the earlier diagram set (700 images, 100 per class). It is not the training target, and its smoke mAP is not a shop or plant result.
- `data/shop/` is the shop-inspect training set (700 images, split 490/105/105). No shop metric is in this repository yet. That count is not about 6,800 factory images.
- The corrosion photos are pipe, bolts, beams, coupons, a plate, a tailgate, nails, a pump, pewter, and a valve. They are not switchgear enclosures.
- Test and validation are 3 images each. Test mAP@0.5 0.12375 and validation mAP@0.5 0.27528 are not comparable to 0.93.
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

python scripts/synthesize_enclosures.py --per-class 100 --seed 42
# writes data/synthetic/ and outputs/synthetic_enclosure_preview.jpg
# does not touch data/train, data/val, or data/test

python scripts/preview_augmentations.py \
    --image data/train/images/Metal_Dented_Defect.jpg \
    --labels data/train/labels/Metal_Dented_Defect.txt \
    --output outputs/augmentation_preview.jpg

python train.py --config configs/train.yaml --data configs/data_wikimedia.yaml --run-name yolov8
python evaluate.py --weights outputs/training/yolov8/weights/best.pt --data configs/data_wikimedia.yaml --device cpu

# Next train uses the shop-inspect set. Not started in this change.
# python train.py --config configs/train.yaml --data configs/data.yaml --run-name yolov8_shop

# Diagram-set smoke. Does not replace the Wikimedia checkpoint or the shop set.
python train.py --config configs/train_synthetic.yaml --data configs/data_synthetic.yaml --run-name yolov8_synthetic
python scripts/analyze_thresholds.py

python infer.py --image data/test/images/Corroded_Bolt.jpg --weights outputs/training/yolov8/weights/best.pt --device cpu
python infer.py --demo --image data/sample/rusty_steel_plate_preview.jpg --no-log

python scripts/show_recent_inspections.py
python scripts/export_dashboard_data.py
python export_onnx.py --weights outputs/training/yolov8/weights/best.pt --imgsz 320

uvicorn app:app --host 0.0.0.0 --port 8000
pytest
```

`GET /health` returns `{"status": "ok"}`. `POST /predict` takes a multipart image and returns `result`, `model_version`, `inference_ms`, and `detections` with `class`, `confidence`, and `bbox`. If weights are missing, the response is flagged `"mock": true`.
