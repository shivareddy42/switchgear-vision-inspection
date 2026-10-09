# Repository verification — 2026-10-09

This review restores the previously merged hardening and generator changes
after the default branch returned to the older exported checkout. Restoration
is a new commit on the current branch; it does not rewrite existing history
or reassign earlier authorship.

## Checks performed

- Installation from `requirements.txt` on macOS ARM64/Python 3.12.14, and
  `pip check`. Platform markers preserve Linux CPU PyTorch pins while allowing
  standard macOS/Windows wheels.
- Full test suite, including API, database, dataset validation, decision policy,
  original-generator identity and plan, portable fonts, ONNX input preprocessing,
  preservation of existing training runs, and photo attribution.
- Public-photo dataset: 16/3/3 split, no validation errors; three classes lack
  photo annotations, as documented.
- Shop dataset: 700 images, original generator plan and label validation. The
  earlier full renderer rebuild report remains in
  `scripts/shop_inspect/verification.json`; JPEG identity is not claimed.
- Real CLI and FastAPI inference on `enc_00001.jpg`: `mock=false`, a detected
  missing/loose component, and a FAIL decision under the example thresholds.
- Included shop ONNX graph accepted by `onnx.checker`; ONNX Runtime executed a
  real-image input with output shape `[1, 11, 8400]` and finite values. This does
  not establish decoded-box equality or Jetson/TensorRT performance.
- Re-evaluation of all 105 held-out shop test images on CPU: precision 0.989092,
  recall 0.984482, mAP@0.5 0.993780, mAP@0.5:0.95 0.878790, NMS IoU 0.50.
  These differ slightly from the copied original result at NMS IoU 0.70;
  the original artifact is retained rather than overwritten.
- Relative Markdown links and a targeted credential-pattern scan of tracked
  text files: no broken links or matching credentials found. This is a limited
  scan, not a security audit or a check of every historical revision.
- Source, creator, license, and modification records restored for all 67
  annotated photo-domain examples. The repository's MIT code license does not
  replace image/font licenses.

## Fixes made during this review

The API returns HTTP 503 when configured weights are missing; mock inference
requires `MODEL_MOCK=1`. Training refuses to delete or replace an existing run.
ONNX preprocessing letterboxes instead of stretching rectangular images, with
matching square preprocessing for the PyTorch smoke check. Thresholds outside
0..1 or containing NaN/infinity are rejected. Quick-start commands preserve
included checkpoints and use an actual test image.

The video/webcam CLI currently checks the first frame. Continuous acquisition,
plant validation, production authentication, hardware integration, retraining,
and Jetson/TensorRT benchmarking remain outside this prototype's verification.

The repository is suitable for sharing as a public/synthetic-data inspection
prototype. Its high synthetic score is not evidence of factory performance.
