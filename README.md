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

## Setup

    python -m venv .venv && source .venv/bin/activate     # or use Colab
    pip install -r requirements.txt

All experiments that specify a seed use **6304**.

## Reproducing results

- Task 1: run `task1/ATML_PA1_Task1_skeleton.ipynb` top to bottom (Colab, T4 GPU). Results land in `task1/results/`.
- Task 2 / 3: `python task2/train.py --method source_only|dan|dann|cdan`, then `python task2/evaluate_final.py`. Task 3 reuses the Task 2 source-only checkpoint.
- Task 4: `python task4/train.py --method vanilla|gcsc|proser`, then `python task4/evaluate_osr.py`.

## Attribution

External implementations used (AdaIN style transfer, DANN gradient-reversal layer, etc.) go here as they're integrated:
- TODO: add attribution entries as you bring in public code, per the assignment's README requirement.

## Status

- [ ] Task 1
- [ ] Task 2
- [ ] Task 3
- [ ] Task 4
- [ ] Report
