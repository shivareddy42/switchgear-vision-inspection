# Edge deployment

```
CAMERA -> JETSON -> TENSORRT YOLO ENGINE -> DEFECT DETECTIONS -> PASS/FAIL -> SQL and quality dashboard
```

PyTorch is the training and development stack. The checkpoint in this repo was trained on CPU from `yolov8n.pt`.

ONNX is the portable file between that PyTorch environment and a runtime that is not Python. `export_onnx.py` wrote `outputs/training/yolov8/weights/best.onnx`, `onnx.checker` accepted it, and ONNX Runtime executed it. See `outputs/onnx_comparison.json` for the numeric notes. Decoded boxes were not claimed to match, because the PyTorch decoder returned none on the sample and the ONNX tensor is the raw head.

TensorRT is the NVIDIA runtime that builds an engine for a specific GPU. It is the step that would sit on the Jetson. It was not installed and no engine was built.

A Jetson is a small NVIDIA computer mounted at the station. The reasons to put the model there, rather than streaming every frame to a server, are latency, less dependence on the plant network, less bandwidth, images that stay on site, a station that still inspects if the uplink drops, and a cleaner split between the inspection PC and the rest of the OT network.

No Jetson was available. The 42 ms/frame figure is a reference target only. The timing measured here is CPU, from `outputs/evaluation_summary.json` on the 3-image test pass: preprocess 0.12682933341541988 ms, inference 12.12008533335999 ms, postprocess 0.5936600000495673 ms. A cold `infer.py` call on `Corroded_Bolt.jpg` was 1029.5 ms, which includes process startup. Neither number is a Jetson benchmark.
