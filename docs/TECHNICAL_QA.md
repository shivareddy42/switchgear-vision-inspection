# Technical Q&A

This page captures the engineering rationale behind the public prototype. It is intentionally written around what the repository actually contains and measures.

## Where did the data come from?

Industrial defect imagery is difficult to obtain publicly, especially for switchgear-specific assembly defects.

The public-data path in this repository uses documented, license-reviewed sources. `data/sources.csv` records the source URL, license information, class mapping, and ingestion notes. The current Wikimedia collection contains 27 files, of which 22 have manually reviewed YOLO boxes.

The repository also includes 700 procedural shop-inspection renders so the full seven-class pipeline can be exercised end to end. Those renders are synthetic and are kept separate from the photograph domain.

A production deployment would replace or extend both with plant-owned imagery from the actual enclosure and busbar families.

## Why augment?

The limited public photographs do not capture every realistic acquisition condition. Training augmentation simulates moderate changes in:

- lighting / exposure
- focus and motion blur
- sensor noise
- camera angle
- scale and framing

Validation and test data remain untouched except for deterministic model preprocessing.

Augmentation is not a substitute for real rare-defect examples. It improves robustness around examples that already exist; it does not create trustworthy domain coverage by itself.

## Why YOLOv8?

The station needs both the defect category and its location.

A pure classifier can say that an image looks defective but cannot point the operator to the defect. Segmentation can provide a tighter boundary but substantially increases annotation cost. YOLOv8 gives a practical object-detection speed/accuracy tradeoff and a straightforward ONNX export path for edge deployment.

It is a prototype choice, not a claim that YOLOv8 is universally optimal.

## What does the confidence score mean?

Each predicted box has a model confidence. The decision layer compares that value with the configured threshold for the predicted class.

A lower threshold generally increases sensitivity but can increase false rejects. A higher threshold reduces false alarms but can increase missed defects.

The values in `configs/thresholds.yaml` are examples, not plant operating points.

## How should thresholds be selected?

Use a reviewed validation set and optimize around actual quality risk.

For a severe defect such as a weld crack or a missing component, the acceptable tradeoff may favor recall even if that creates more human reviews. Cosmetic classes may use a different operating point.

The repository includes `scripts/analyze_thresholds.py` to sweep confidence thresholds and expose precision, recall, false positives, and false negatives.

## How was limited data handled?

The prototype uses:

- transfer learning from pretrained YOLO weights
- source provenance tracking
- exact and perceptual-hash deduplication
- train/validation/test separation before augmentation
- realistic augmentation on training only
- per-class evaluation rather than relying only on aggregate mAP
- a separate synthetic domain for exercising classes that lack public photo coverage

The most important remaining limitation is still real domain coverage.

## Why run at the edge?

An edge device such as an NVIDIA Jetson can keep image inference close to the station.

Potential benefits:

- lower latency
- less dependence on plant internet connectivity
- reduced bandwidth
- better control over image-data movement
- simpler separation between plant equipment and cloud services

A production design still requires IT/OT cybersecurity review.

## Why ONNX?

ONNX provides a portable representation between the PyTorch training environment and deployment runtimes.

The included export is validated with ONNX tooling. The next deployment step on NVIDIA hardware would be TensorRT.

## Why TensorRT?

TensorRT is NVIDIA's optimized inference runtime. It can apply graph optimization and validated reduced-precision execution to improve throughput / latency on NVIDIA GPUs and Jetson devices.

No Jetson/TensorRT latency number is claimed by this public build.

## What would change in a real switchgear plant?

The deployment sequence would be:

1. collect plant-owned good and defective parts
2. define the defect taxonomy with quality engineering
3. standardize camera / lens / distance / lighting
4. build a reviewed golden validation set
5. calibrate thresholds by defect severity
6. run the model in shadow mode
7. compare predictions with human inspection on the same parts
8. provide operator override
9. keep a complete audit trail
10. monitor reject rates, score distributions, and image drift
11. retrain when product, supplier, camera, or lighting changes
12. complete IT/OT security review
13. stage deployment from one station outward

See `docs/PRODUCTIONIZATION.md` for the detailed version.

## What do the current metrics prove?

The high synthetic-domain score demonstrates that the training/inference/evaluation pipeline can learn the procedural defect taxonomy.

It does **not** prove plant performance.

The real-photo domain-gap run is intentionally included because it shows the opposite side of the problem: a detector that performs very well on rendered cabinets can still misread seams, handles, meters, shadows, and other real visual structure. That is why plant-owned validation data is the gating item before production use.
