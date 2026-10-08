# Data strategy

The detector needs a box and a class. A public photo is usable only when the defect is visible and the license allows the file to be stored here. Classification datasets, radiographs, and stock photos with a conflicting copyright mark were recorded and left out.

## scratch

A scratch is a narrow linear mark where the surface is cut or burnished. It is a surface defect. The mark is often brighter than the surrounding metal and has a direction.

Public detection data exists. NEU-DET includes a scratches class (300 of the 1,800 published images) with boxes, but the official page states no license, so those bytes are not in this repo. KolektorSDD2 mentions scratches among mixed surface defects, and its official page did not state a license when fetched on 2026-10-08. A CC0 Wikimedia file of a "heavily scratched" stainless texture was reviewed and rejected: it is a seamless texture made in an editor, not a discrete mark on a part.

Annotation is moderate. The box should touch the visible mark, not the whole panel. Grinding lay, die lines, and seams look similar.

No image in this repository is labeled scratch.

## dent

A dent is a local plastic depression. Light pools on one side and a shadow falls on the other. It is a surface defect, not missing material.

Public close-ups of dents on industrial enclosures are scarce. The MVTec AD paper uses dents as an example of the anomalies in that dataset. The license is CC BY-NC-SA 4.0, but the download is form-gated and no files were retrieved, so nothing was mapped. `File:Weld-def-1.jpg` shows a bulge in an unrestrained welded plate. That is weld distortion, not a dent, and it was left unlabeled.

Annotation is harder than a scratch because the edge is soft. A tight box around the deformed region is still required.

No image in this repository is labeled dent.

## weld_porosity

Porosity is a cluster of small pits or wormholes in the weld metal, not rust and not a single stop crater. It is a surface or near-surface weld defect when it breaks the surface; subsurface porosity needs radiography, which this camera pipeline does not use.

RIAWELC has a porosity class, but the images are radiographs and the README does not name an SPDX license. LoHi-WELD has pores and an author grant that does not clearly allow redistributing a copy here. A Kaggle set tagged CC0 includes a file named for porosity that visibly carries a Shutterstock watermark, so the set was not ingested. GC10-DET's "welding line" class is a weld bead feature, not porosity, and its original license is unstated.

Annotation is moderate when the pits are in the photo. The box covers the porous patch, not the whole bead. A crater or slag island is not porosity.

No image in this repository is labeled weld_porosity.

## weld_crack

A weld crack is a sharp break in or beside the bead: longitudinal, transverse, or at the toe. It is a weld defect. Undercut and lack of fusion can look similar in a single photo.

The same sources as porosity were reviewed. RIAWELC cracks are radiographic. The Kaggle/Hugging Face weld set uses Bad Weld / Good Weld / Defect, which is not a crack class, and it contains stock-watermarked and screenshot files. No Wikimedia photo reviewed here showed a crack clearly enough to box.

Annotation is hard. The box is the crack, not the bead. If the line might be undercut, it is not labeled.

No image in this repository is labeled weld_crack.

## corrosion

Corrosion is oxide or under-film attack: red rust, pitting, or filiform tracks under paint. It is a surface defect. It is not weld porosity and not a stain of oil.

This is the only class with images in the repository. Seven photos were downloaded from Wikimedia Commons on 2026-10-08 and boxed by hand:

- rusty steel plate (CC0)
- filiform corrosion on a painted tailgate (CC BY-SA 4.0)
- filiform corrosion on a painted aluminum coupon (CC BY-SA 4.0)
- rusted iron nails (CC BY 4.0)
- pitted pump housing (public domain)
- pitted pewter plate (CC BY-SA 4.0)
- corroded valve winch (CC BY 4.0)

None of them is a switchgear enclosure. Annotation is easy when the rust is obvious and harder when pitting is fine. The boxes cover the visible attacked region.

## misaligned_busbar

The bar, or its joint, does not sit on the design position relative to the supports or the neighboring phase. It is an assembly defect. A photo of straight busbars is not a positive example.

Wikimedia has busbar photos. `2500A copper busbars in motor control panel.jpg` shows bars that look parallel and seated. It was kept unlabeled. The isolated-phase bus at Bui Dam is an installation in progress, not a finished misalignment, and it was not downloaded. No public photo reviewed here showed a measured misalignment.

Annotation needs a fixture reference. Without a known correct position, a box would be a guess.

No image in this repository is labeled misaligned_busbar.

## missing_or_loose_component

A fastener, terminal, breaker, or bracket is absent or obviously not seated, and the empty location is visible. It is an assembly defect. An intact panel is not a positive.

`Electrical panel in Swiss industrial building.jpg` shows a ground/neutral link that is present. It was not labeled. No reviewed photo showed a missing fastener at a known location.

Annotation is only valid when the missing part's location can be seen. Boxing a whole cabinet is not acceptable.

No image in this repository is labeled missing_or_loose_component.

## What this means

Public data does not support all seven classes. Forcing weld, busbar, or fastener photos into the empty classes would make the metrics look broader than the evidence. The training run below uses the seven corrosion boxes only, and the empty classes stay empty.
