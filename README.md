# ATML PA1 — Beyond IID: Inductive Biases, Domain Shift & Open-Set Recognition

Course: EE-5102/CS-6304 — Advanced Topics in Machine Learning, Fall 2026
Author: Sayim (sayim665)

## Repo layout

```
pa1-beyond-iid/
  common/            shared utilities (seeding, logging, plotting, metrics)
  task1/             Inductive Biases and Feature Representations
  task2/             Unsupervised Domain Adaptation (PACS, target = Sketch)
  task3/             Domain Generalization (PACS, unseen target = Sketch)
  task4/             Open-Set Recognition (CIFAR-10 known / CIFAR-100 unknown)
  report/            NeurIPS-format LaTeX report + figures
```

Note: task2's shared/ and models/ folders (pacs.py, pacs_protocol.py, mmd.py, backbone.py,
classifier_head.py) are imported directly by task3's scripts rather than duplicated, so both
tasks are guaranteed to use identical splits, backbone architecture, and MMD implementation.
See task3/README.md for details.

## Setup

    python -m venv .venv && source .venv/bin/activate     # or use Colab
    pip install -r requirements.txt

All experiments that specify a seed use **6304**.

## Reproducing results

- Task 1: run `task1/ATML_PA1_Task1_skeleton.ipynb` top to bottom (Colab, T4 GPU). Results land in `task1/results/`.
- Task 2: `task2/methods/source_only.py`, `dan.py`, `dann.py`, `cdan.py` (run in that order),
  then `task2/evaluation/evaluate_final.py`, then `task2/methods/controlled_study.py`.
- Task 3: `task3/methods/erm.py` (loads Task 2's source_only checkpoint, does not retrain),
  then `dan_dg.py`, `sam.py`.
- Task 4: `python task4/train.py --method vanilla|gcsc|proser`, then `python task4/evaluate_osr.py`.

## Attribution

- Task 1 Step 3 (shape/texture cue conflicts): AdaIN style transfer via `naoto0804/pytorch-AdaIN`
  (MIT license, github.com/naoto0804/pytorch-AdaIN).
- Task 3 SAM optimizer: two-step ascent/descent pattern follows the reference implementation at
  github.com/davda54/sam.

## Status

- [x] Task 1
- [ ] Task 2 (results done, report pending)
- [ ] Task 3 (in progress)
- [ ] Task 4
- [ ] Report
