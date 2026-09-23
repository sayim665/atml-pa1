# Task 4 -- Open-Set Recognition

## Structure
- `models/resnet_cifar.py` -- CIFAR ResNet-18 (3x3 stride-1 stem, no max-pool). Returns
  (logits, penultimate_feature) so post-hoc scores can use both.
- `data/cifar10.py` -- known classes: stratified 90/10 split, seed 6304, standard vs GCSC
  (RandAugment) train transforms.
- `data/cifar100_unknowns.py` -- fixed near/far unknown groups (8 classes each, 800 images
  each, straight from CIFAR-100's test partition -- no subsampling needed since CIFAR-100
  test has exactly 100 images/class). CIFAR-100 TRAIN is never used anywhere in Task 4.
- `methods/vanilla.py` -- trains BOTH Vanilla and GCSC (same function, `gcsc=True/False` flag,
  since they're identical recipes except for the extra augmentation).
