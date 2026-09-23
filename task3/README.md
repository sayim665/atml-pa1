# Task 3 -- Domain Generalization

## Structural note (deviation from the suggested layout)
This repo keeps `pacs.py`, `pacs_protocol.py`, `mmd.py`, `backbone.py`, and `classifier_head.py`
under `task2/shared/` and `task2/models/` rather than a top-level `shared/` folder, since they
were built for Task 2 first. Task 3's scripts import directly from those Task 2 paths (see the
`sys.path.insert` lines at the top of each file) rather than duplicating the code, so Task 2 and
Task 3 are GUARANTEED to use identical splits, backbone architecture, and MMD implementation --
which is what the spec actually requires, even though the folder layout differs cosmetically
from the suggested tree.

## Methods
- `methods/erm.py` -- loads Task 2's `source_only.pth` UNCHANGED (no retraining). This is the
  Task 3 ERM baseline.
- `methods/dan_dg.py` -- pairwise MMD alignment across Photo/Art/Cartoon ONLY. Sketch is never
  loaded anywhere in this file.
- `methods/sam.py` + `methods/sam_optimizer.py` -- Sharpness-Aware Minimization. The SAM
  optimizer wrapper follows the standard two-step ascent/descent pattern from Foret et al.
  (2021), matching the widely-used public reference implementation at
  https://github.com/davda54/sam (AdamW used as the base optimizer, per spec).

## Constraint checklist (per spec, "Before You Submit")
- [ ] No Sketch image loaded by Task 3 training, source-side diagnostics, checkpoint selection,
      or hyperparameter selection -- verified by inspection: `dan_dg.py` and `sam.py` never
      construct a target-domain loader at all.
- [ ] Task 2's Sketch results never used to revise any Task 3 setting.
- [ ] ERM checkpoint reused unchanged from Task 2.
