# Architecture

```
camera frame
    -> preprocess (resize, keep aspect information)
    -> YOLOv8 detector (class, box, confidence)
    -> threshold table
    -> PASS or FAIL
    -> SQL inspection row and detection rows
    -> CSV for a Power BI model
```

Training is PyTorch with Ultralytics. The runnable demo does not need weights: `infer.py --demo` and the API, if `best.pt` is missing, return a detection that is stamped MOCK on the image and `"mock": true` in JSON.

`infer.py` and `app.py` share `inspection.inference.Inspector`. The FastAPI process keeps one inspector. It does not load the network on every request.

A FAIL is any detection whose confidence is strictly greater than that class's threshold. The default is 0.50. The per-class numbers in `configs/thresholds.yaml` are marked EXAMPLE. They were not fit on this data.

The database is SQLite through SQLAlchemy. The same tables can sit on SQL Server by changing `INSPECTION_DATABASE_URL`. Rows are `inspections` and `detections`.

`scripts/export_dashboard_data.py` writes one CSV. It does not build a PBIX.

The edge path is documented in `docs/EDGE_DEPLOYMENT.md`: ONNX, then TensorRT on a Jetson. TensorRT was not run here. The ONNX file is produced by `export_onnx.py` and checked with ONNX Runtime on CPU.
