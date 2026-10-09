# Domain gap: shop-inspect weights on licensed photographs

The synthetic-domain shop-inspect test mAP@0.5 is 0.993238 (0.993). That number does not transfer to these photographs. The renders in `data/shop/` are procedural. This page is what `outputs/training/yolov8_shop/weights/best.pt` actually output on real licensed photos. It is not a new mAP, and it is not plant performance.

No training was run for this check. The photos were not relabeled to match the model. Folder names below describe the licensed-photo review set used for this check.

## What was run

67 photographs, on CPU, image size 640, NMS IoU 0.70. A detection in the summary is a model box at or above confidence 0.25, which is the Ultralytics predict default. Annotated copies, long side at most 1600 pixels, are in `outputs/real_photo_predictions/`. The short record is `outputs/real_photo_predictions/summary.json`.

| Filing | Photos |
| --- | --- |
| `corrosion/` | 2 |
| `missing_or_loose_component/` | 1 |
| `unlabeled/` | 64 |

56 of the 67 photos produced at least one box at or above 0.25. 48 of the 67 produced at least one box at or above 0.50. The 285 boxes were:

| Class | Boxes |
| --- | --- |
| scratch | 98 |
| dent | 27 |
| weld_porosity | 5 |
| weld_crack | 9 |
| corrosion | 40 |
| misaligned_busbar | 5 |
| missing_or_loose_component | 101 |

The highest confidence was scratch 0.956513, on `unlabeled/021-square-d-electric-switchboard.jpg`, over a meter bezel.

## Filed defect photos

`corrosion/008-geevor-tin-mine-electrical-switchgear.jpg` drew six boxes: scratch 0.857091, scratch 0.603452, scratch 0.55145, scratch 0.416632, dent 0.41083, and scratch 0.3658. None is corrosion. They sit on seams, handles, and shadows. I would not defend them as scratches or dents.

`corrosion/032-really-old-rusty-electrical-cabinet-at-porth-penrhyn-bangor-geograph-org-uk-6261.jpg` is a rusty cabinet. The only box at or above 0.25 is corrosion 0.478726, on the top rim, not on the rusted face. The next corrosion box is 0.158923.

`missing_or_loose_component/030-dornbirn-damaged-distribution-board-01asd.jpg` has broken hardware at the lower left. The model drew scratch 0.504656 and scratch 0.486453 on the upper-right interior. It did not draw `missing_or_loose_component` at or above 0.25.

## What a person could defend

`box_on_defensible_defect` in the JSON is a look at the drawn boxes. It is not a new label.

20 annotated images were opened. On 18 of those, no drawn box sat on a defect of the predicted class that a person could defend. Two unlabeled photos were the exception, and both files stay unlabeled:

- `unlabeled/029-broken-switchgear-celn-teplice-img-2613.jpg` drew `missing_or_loose_component` at 0.759336 and 0.502559 on a shattered meter. A person could defend that broken meter. The same photo also drew corrosion at 0.597165 and 0.352749, which I would not defend as corrosion.
- `unlabeled/033-burned-distribution-board.jpg` drew `missing_or_loose_component` at 0.661729 and 0.625222 over the charred interior. A person could defend destroyed components there. The weld_crack box at 0.355964 is not a crack I would defend. Burn is not one of the seven classes.

47 annotated images were not opened box by box. Their flag is null. Nothing in those 47 was given a defect label.

Other opened misses, still unlabeled:

- `unlabeled/028-broken-switchgear-celn-teplice-img-2611.jpg` has a damaged left door. The model called that region corrosion 0.60634. I would not defend that class.
- `unlabeled/049-cb-truck-showing-surface-discharge-damage.jpg` shows tracking on an insulator. No box reached 0.25. The highest box was scratch 0.053189.

Opened photos of ordinary gear drew the same classes at high confidence on handles, meters, louvers, busbars, seams, and shadows. Examples: corrosion 0.760851 and dent 0.676466 on `unlabeled/001-electrical-switchgear.jpg`; dent 0.900766 on `unlabeled/065-rankweil-photovoltaic-power-plant-kindergarten-bredreis-pv-distribution-board-wr.jpg`; `missing_or_loose_component` 0.914211 on intact busbar hardware in `unlabeled/014-3-phase-ac-sub-bus-bar-001.jpg`.

## What this is not

This is not a measured mAP on photographs. The 0.993 figure stays a synthetic-domain test result. No plant-level accuracy or Jetson latency claim is made from this run.
