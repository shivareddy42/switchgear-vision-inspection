# Dataset

This repository stores 27 unique images, all downloaded from Wikimedia Commons on 2026-10-08 by `scripts/collect_images.py`. A second collection pass the same day added 15 files. `scripts/clean_dataset.py` kept 27 of 27.

22 images have hand-drawn boxes. Five older domain photos stay unlabeled. The labeled counts are scratch 1, dent 2, weld_crack 2, and corrosion 17. Weld porosity, misaligned busbar, and missing or loose component still have no image.

The 6,800-image figure on the resume is a reference target. It is not the size of this dataset.

## What was included

The first pass is still here: a rusty plate, two filiform photos, rusted nails, a pitted pump, a pewter plate, a corroded valve, plus five unlabeled photos (switchgear, busbars, a panel, weld distortion, spatter removal). Artists and licenses for those files are in `data/sources.csv`.

The second pass preferred industrial metal where Commons had a photo that actually showed the defect:

| File | Class | Pixels | License | Why it is here |
| --- | --- | --- | --- | --- |
| Close_up_of_scratches_from_adjustable_spanner.jpg | scratch | 2167×1409 | CC BY-SA 2.5 | Scratches on a steel pedal |
| Metal_Dented_Defect.jpg | dent | 1315×926 | CC0 | Two dented metal cans |
| Dented_field_gates_-_geograph.org.uk_-_7648834.jpg | dent | 4032×3024 | CC BY-SA 2.0 | Dented metal gate rails |
| Cracks_in_weld.jpg | weld_crack | 1094×893 | CC0 | Cracks in a weld bead |
| CRACKS_IN_WELDS_ON_VIGV_-_NARA_-_17471187.jpg | weld_crack | 4870×6250 | Public domain | Crack in a vane weld |
| Rust_Pits_in_surface_of_Steel_Water_Pipe_GN04592.jpg | corrosion | 3264×2448 | CC0 | Rust pits on steel pipe |
| Badly rusted steel pipes (GN01224) | corrosion | 3264×2448 | CC0 | Rusted pipe stock |
| Corroded_Bolt.jpg | corrosion | 4135×2859 | CC BY-SA 4.0 | Rusted bolt and nut |
| Rusty_Bolts_on_Pipe_Flange.jpg | corrosion | 1453×1081 | CC BY-SA 4.0 | Rust on flange bolts |
| Corroded_pipe.jpg | corrosion | 676×1024 | Public domain | Rusted pipe |
| Pitting_corrosion_on_duplex_stainless_steel_specimens.jpg | corrosion | 2700×2035 | CC BY 4.0 | Pitted stainless coupons |
| Chittenden Locks pipe sections | corrosion | 2848×4288 | CC BY-SA 4.0 | Corroded pipe sections |
| Rust_Bolt.jpg | corrosion | 1600×1200 | CC BY-SA 3.0 | Rusted bolt |
| Walnut Street steel beams | corrosion | 3648×2432 | CC BY-SA 2.0 | Rust on bridge beams |
| Longfellow Bridge steel beams | corrosion | 576×768 | Public domain | Rust on bridge beams |

CC BY and CC BY-SA files require attribution. The image bytes are not under the MIT license that covers the code.

Boxes are in `data/annotated/`. They were drawn on 2026-10-08 by looking at the photos, then drawn back onto the photos and adjusted. They are not inherited from NEU, GC10, or any other detection set. The duplex photo has two boxes. The can photo and the gate photo have two boxes each. The other labeled photos have one.

The car-tailgate filiform photo, the pewter plate, and the nails are still labeled corrosion. Newer industrial photos are in the set as well. The augmentation preview is the can photo, not the tailgate.

## What was reviewed and not included

- NEU-DET. 1,800 images, 300 per class, no license on the official page.
- GC10-DET. The paper says 3,570 images; Dataset Ninja says 2,300. Original license unstated. The Roboflow page fetched on 2026-10-08 showed no license. Counts were not remeasured.
- KolektorSDD2. 356 defective and 2,979 defect-free images on the official page. That page did not state a license. Other sites disagree with each other.
- MVTec AD. CC BY-NC-SA 4.0, 5,354 images in the paper, form-gated download, no bytes retrieved, masks not converted to the seven classes.
- RIAWELC. 24,407 radiographic images. Wrong modality, and the README does not name an SPDX license.
- LoHi-WELD. 3,022 images, including pores. The README allows use with citation but does not clearly allow redistributing a copy here.
- Kaggle "Welding Defect - Object Detection", also mirrored at `l985215117/welding-defect-object-detection`. The Kaggle page says CC0. A snapshot counted 2,028 JPEGs (1,619 train, 283 valid, 126 test). Classes are Bad Weld, Good Weld, and Defect. A porosity-named test image shows a Shutterstock watermark. 122 filenames contain Screenshot. Six contain `260nw`. 1,315 filename groups remain if the Roboflow `.rf.` suffix is stripped. None of those bytes are in this repo. This pass did not add them.
- A CC0 seamless "scratched steel" texture. No discrete scratch to box.
- `File:Porosity.jpg` (CC0). Holes in a flat sheet, not in a weld bead.
- `File:Cast porosity defect.jpg` (CC0, 235×215). Casting pores, not a weld.
- `File:Weld Undercut.jpg` (CC BY-SA 4.0). Undercut is not a project class.
- `File:Stress-Corrosion-Cracking-caused-by-weld-stress-01.jpg`. A fracture surface, not a bead crack.
- `File:Stress corrosion cracking revealed by magnetic particles.JPG`. Magnetic-particle indications, not a camera view of the crack.
- `File:SCC Mitigation in Weld.jpg` and the crevice-corrosion desalination figure. Arrows and captions are drawn on the images.
- NASA/NARA injector photo whose caption says cracked weld. The crack was not visible, so it was not stored.
- `File:USS Missouri Dented Rail.jpg`. The rail looks bent, and a person stands in front of it. Not labeled dent.
- Three wider frames of the same rusted water pipe (GN04590A, GN04591, GN04591A). The close view GN04592 was kept.

A Commons search for pores in a weld bead, a misaligned busbar, and a missing or loose fastener did not return a photo that could be boxed. Those three classes stay empty.

## Split

Labeled originals were split before augmentation. Seed 42. Requested ratios 70/15/15. With 22 labeled images the realized split is train 16, validation 3, test 3.

| Split | Files |
| --- | --- |
| train | Chittenden pipes, clean pump, spanner scratches, corroded pipe, cracks in weld, both filiform photos, dented cans, pewter plate, duplex coupons, rust bolt, rust pits, Walnut Street beams, flange bolts, rusty plate, nails |
| val | dented field gates, Longfellow Bridge beams, Owalla valve |
| test | badly rusted pipes, VIGV weld crack, corroded bolt |

Difference-hash clustering (6 bits) found no near-duplicate pair across the split. `scripts/validate_dataset.py` reported 0 errors and 3 warnings, one for each class that still has no boxes. The five unlabeled photos are not in the split. They are not negatives.

## Augmentation

Augmentation is applied only to training data. Validation and test images remain untouched except for deterministic resizing / preprocessing.

The training run saw each of the 16 training images once per epoch for 5 epochs: 80 presentations. That is not 80 new original images. Ultralytics applied the limits in `configs/train.yaml`. Mosaic, mixup, copy-paste, random erasing, and RandAugment were off. Vertical flip is off.

The preview grid is Albumentations, not the Ultralytics loader. It uses `data/train/images/Metal_Dented_Defect.jpg`. Brightness, blur, and noise left both boxes at `(0.24, 0.45, 0.32, 0.34)` and `(0.74, 0.42, 0.32, 0.40)`. Rotation and perspective moved them. The combined panel, which flips, moved the centers to about `0.82` and `0.27`. The two implementations follow one policy and are not pixel-identical.

Augmentation does not create weld porosity, a misaligned busbar, or a missing fastener. Those classes are still empty. Factory images are required before any deployment claim.

## Counts a script will print

`python scripts/dataset_stats.py` prints the 27 cleaned images, the per-class box counts, and the 16/3/3 split. It does not multiply by an augmentation factor.
