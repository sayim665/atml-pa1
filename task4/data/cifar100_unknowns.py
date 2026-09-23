"""
CIFAR-100 unknown-class loading: fixed near/far unknown groups, per spec. CIFAR-100's test
partition has exactly 100 images per fine class, so 8 classes x 100 images = 800 per group
automatically -- no subsampling needed to hit the spec's "800 images per group."
CIFAR-100 TRAINING images are never used anywhere in Task 4 (evaluation-only).
"""
import torch
import torchvision
from torchvision import transforms as T

from cifar10 import CIFAR10_MEAN, CIFAR10_STD  # reuse CIFAR-10's normalization for the shared classifier

NEAR_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_CLASSES = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]


def get_eval_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


class CIFAR100UnknownSubset(torch.utils.data.Dataset):
    """CIFAR-100 TEST images restricted to the given fine-class names. Labels returned are
    just the CIFAR-100 fine-class index (not remapped to CIFAR-10's 10 classes) -- OSR scoring
    only needs to know these ARE unknowns, not which unknown class they are, though we keep
    the original label for the optional per-class failure analysis the spec asks for."""
    def __init__(self, data_root, class_names, transform):
        base = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)
        name_to_idx = {name: i for i, name in enumerate(base.classes)}
        for name in class_names:
            assert name in name_to_idx, f"'{name}' not found in CIFAR-100 fine classes"
        wanted_idx = {name_to_idx[name] for name in class_names}

        self.samples = [(img, label, base.classes[label]) for img, label in
                         zip(base.data, base.targets) if label in wanted_idx]
        self.transform = transform
        self.base_classes = base.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_arr, label, class_name = self.samples[i]
        from PIL import Image
        img = Image.fromarray(img_arr)
        if self.transform:
            img = self.transform(img)
        return img, label, class_name


def build_unknown_datasets(data_root="./data"):
    eval_tf = get_eval_transform()
    near_ds = CIFAR100UnknownSubset(data_root, NEAR_CLASSES, eval_tf)
    far_ds = CIFAR100UnknownSubset(data_root, FAR_CLASSES, eval_tf)
    print(f"Near unknowns: {len(near_ds)} images ({NEAR_CLASSES})")
    print(f"Far unknowns: {len(far_ds)} images ({FAR_CLASSES})")
    return near_ds, far_ds
