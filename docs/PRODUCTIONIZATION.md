# Production deployment

A Siemens-like switchgear line would not deploy the checkpoint in this repository. The public set does not contain the product, the camera, or six of the seven defect classes. The sequence below is the work that has to happen first.

1. Collect plant-owned images of the enclosures and bus assemblies actually built, including the rare defects and a large set of good parts.
2. Freeze the defect taxonomy with the quality engineers. Rename or drop classes that do not match the drawings.
3. Fix the camera position, lens, and working distance so the part fills the frame the same way every cycle.
4. Fix the lighting. Do not rely on augmentation to invent the station.
5. Label a golden validation set that never enters training, with boxes reviewed by a second person.
6. Set a threshold per defect and per severity from that set, using the cost of a missed crack or a missing fastener against the cost of a false reject.
7. Run in shadow mode: the model scores parts, and it does not stop the line.
8. Compare those scores with the inspectors on the same parts.
9. Give the operator an override and record who used it.
10. Keep the audit trail already sketched here: image name, model version, threshold file, boxes, PASS/FAIL, time.
11. Watch drift: score histograms, reject rate, and a sample of images each week.
12. Retrain when the product, the design, the camera, the light, or the supplier changes.
13. Version the weights, the data snapshot, and the threshold file together.
14. Put the station through the plant cybersecurity and OT review before it touches a controller.
15. Roll out one station, then the cell, then the line. Do not start with an automatic stop.

The business figures cited with the original project (about 63% fewer inspection hours, about 5 months payback) are reference targets. This repository has no time study and no cost model behind them.
