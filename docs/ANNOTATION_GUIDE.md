# Annotation guide

Labels are YOLO detection lines: `class_id x_center y_center width height`, normalized to the image. Class ids are in `configs/data.yaml`.

Boxes in this repository were drawn by looking at the file on 2026-10-08. They were not converted from another dataset's labels. They are ordinary detection boxes around the visible region, not pixel masks.

Review rule: a second person should reject a box that includes a large area of intact surface just to make the label easier, and should reject a class that the photo does not show.

## scratch

Counts: a narrow mark that breaks or burnishes the surface.

Does not count: grain from rolling, a gasket edge, a label, a cable, a shadow line.

Box the mark. Do not box the panel. If two marks are separate, use two boxes. If the mark might be a seam, leave it unlabeled.

## dent

Counts: a local depression with a highlight and a shadow.

Does not count: a designed rib, a weld ripple, or a plate that bowed because it was not clamped (`Weld-def-1.jpg` is that case and is unlabeled).

Box the deformed patch. The edge is soft, so a small margin is acceptable. Do not box the part.

## weld_porosity

Counts: a group of pits or wormholes in the weld metal.

Does not count: one stop crater, slag, spatter, rust, or a dirty lens.

Box the porous patch. If the pits are not clearly in the weld, do not use this class.

## weld_crack

Counts: a sharp linear break in the bead or along the toe.

Does not count: undercut, a scratch on the parent metal, or a drawn line on a diagram.

Box the crack. If the photo cannot distinguish a crack from undercut, leave it unlabeled.

## corrosion

Counts: rust, pits from corrosion, or filiform filaments under paint.

Does not count: oil, dust, a heat tint that is not oxide, or weld porosity.

Box the attacked region. On the rusty plate the box is the main rusted field on the right, not every speck on the rim. On the tailgate the box is the tracked paint, not the car. On the pump the box is the pitted flange. On the nails the box is the nails.

## misaligned_busbar

Counts: a bar or joint that is visibly off the supports or the neighboring phase, in a view where the correct position is obvious.

Does not count: a normal bus compartment, perspective from a handheld camera, or bars on temporary scaffolding during installation.

Box the misaligned bar or the bad joint, not the whole lineup. If the correct position is unknown, do not label.

## missing_or_loose_component

Counts: a fastener, terminal, breaker, or bracket that is absent or visibly unseated, when the empty location can be seen.

Does not count: a component hidden behind a cover, a design that never had that part, or a whole cabinet with no local evidence.

Box the empty location or the loose part. Do not box the panel.

## Review

- Split originals before any augmentation. Do not draw boxes on augmented copies and then split.
- Near-duplicate frames stay in one split. `scripts/split_dataset.py` clusters difference-hashes.
- Empty classes stay empty. Do not invent a box to fill a taxonomy gap.
- Full-frame boxes are weak supervision. This repo does not use them, and they must not be mixed into mAP if they are added later.
