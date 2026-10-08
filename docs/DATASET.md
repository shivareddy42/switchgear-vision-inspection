# Dataset

This repository stores 12 unique images, all downloaded from Wikimedia Commons on 2026-10-08 by `scripts/collect_images.py`. Seven are labeled `corrosion`. Five are unlabeled domain photos. No other class has an image.

The 6,800-image figure on the resume is a reference target. It is not the size of this dataset.

## What was included

| File | Class | Pixels | License | Why it is here |
| --- | --- | --- | --- | --- |
| Rusty_steel_plate.jpg | corrosion | 4000×3000 | CC0 | Rust on plate steel |
| Filiform_corrosion.jpg | corrosion | 4160×3120 | CC BY-SA 4.0 | Filiform tracks on painted steel |
| Filiform_corrosion_on_painted_aluminum.jpg | corrosion | 6000×4000 | CC BY-SA 4.0 | Filiform coupon |
| Swelling_of_Iron_nails_due_to_rusting.jpg | corrosion | 3120×4160 | CC BY 4.0 | Rusted nails |
| Clean_pump.jpg | corrosion | 2592×1944 | Public domain | Pitted pump flange |
| Pewter_plate_with_corrosion.jpg | corrosion | 4297×2951 | CC BY-SA 4.0 | Pitted pewter |
| Owalla_Dam_rusted_valve_winch_mechanism_closeup_Osun.jpg | corrosion | 4284×5712 | CC BY 4.0 | Corroded mechanism |
| 4.16kV_switchgear.jpg | unlabeled | 2048×1536 | CC0 | Switchgear, no defect identified |
| 2500A_copper_busbars_in_motor_control_panel.jpg | unlabeled | 2236×3976 | CC BY-SA 4.0 | Busbars that were not called misaligned |
| Electrical_panel_in_Swiss_industrial_building.jpg | unlabeled | 2624×2848 | CC BY 4.0 | Panel hardware that was not called missing |
| Weld-def-1.jpg | unlabeled | 2396×1691 | CC BY-SA 3.0 | Weld distortion, not a dent or a crack |
| Welding_processes_160303-F-OV732-016.jpg | unlabeled | 4256×2832 | Public domain | Spatter removal; spatter is not a project class |

Artists and file pages are in `data/sources.csv`. CC BY and CC BY-SA files require attribution. The image bytes are not under the MIT license that covers the code.

Each corrosion image has one hand-drawn box in `data/annotated/corrosion/`. The boxes were checked by drawing them back onto the photos. They are not inherited from NEU, GC10, or any other detection set.

## What was reviewed and not included

- NEU-DET. 1,800 images, 300 per class, no license on the official page.
- GC10-DET. The paper says 3,570 images; Dataset Ninja says 2,300. Original license unstated. The Roboflow page fetched on 2026-10-08 showed no license. Counts were not remeasured.
- KolektorSDD2. 356 defective and 2,979 defect-free images on the official page. That page did not state a license. Other sites disagree with each other.
- MVTec AD. CC BY-NC-SA 4.0, 5,354 images in the paper, form-gated download, no bytes retrieved, masks not converted to the seven classes.
- RIAWELC. 24,407 radiographic images. Wrong modality, and the README does not name an SPDX license.
- LoHi-WELD. 3,022 images, including pores. The README allows use with citation but does not clearly allow redistributing a copy here.
- Kaggle "Welding Defect - Object Detection", also mirrored at `l985215117/welding-defect-object-detection`. The Kaggle page says CC0. A snapshot counted 2,028 JPEGs (1,619 train, 283 valid, 126 test). Classes are Bad Weld, Good Weld, and Defect. A porosity-named test image shows a Shutterstock watermark. 122 filenames contain Screenshot. Six contain `260nw`. 1,315 filename groups remain if the Roboflow `.rf.` suffix is stripped. None of those bytes are in this repo.
- A CC0 seamless "scratched steel" texture. No discrete scratch to box.

## Split

Labeled originals were split before augmentation. Seed 42. Requested ratios 70/15/15. With 7 images the largest-remainder split, keeping at least one image in each split, is train 5, validation 1, test 1.

| Split | File |
| --- | --- |
| train | Filiform_corrosion.jpg, Filiform_corrosion_on_painted_aluminum.jpg, Owalla dam valve, Pewter plate, rusted nails |
| val | Clean_pump.jpg |
| test | Rusty_steel_plate.jpg |

Difference-hash clustering (6 bits) found no near-duplicate pair, so each image is its own cluster. `scripts/validate_dataset.py` reported 0 errors. The five unlabeled photos are not in the split. They are not negatives: a negative would mean a reviewed good part, and these were only reviewed as "do not force a class."

## Augmentation

Augmentation is applied only to training data. Validation and test images remain untouched except for deterministic resizing / preprocessing.

The training run saw each of the 5 training images once per epoch for 5 epochs: 25 presentations. Ultralytics applied the limits in `configs/train.yaml` (rotation within 8 degrees, small translation, scale, shear, perspective, horizontal flip, mild HSV). Mosaic, mixup, copy-paste, random erasing, and RandAugment were off. Vertical flip is off because a fixtured part is not inspected upside down.

The preview grid is Albumentations, not the Ultralytics loader. On `Filiform_corrosion.jpg` the photometric panels kept the box at `(0.42, 0.58, 0.46, 0.42)`. Rotation and perspective moved it. The combined panel, which flips, moved the center x to about `0.64`. That is the check that boxes follow the pixels. The two implementations follow one policy and are not pixel-identical.

Augmentation does not create a rare defect the dataset does not contain. Scratch, dent, porosity, cracks, busbar misalignment, and missing hardware are still absent. More real examples are required. Class weights and oversampling are not a substitute for those photos. Factory images are required before any deployment claim.

## Counts a script will print

`python scripts/dataset_stats.py` prints the 12 cleaned images, 7 corrosion boxes, and the 5/1/1 split. It does not multiply by an augmentation factor.
