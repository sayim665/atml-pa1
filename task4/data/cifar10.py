"""
CIFAR-10 known-class loading: stratified 90/10 split of the official train partition
(seed 6304), full official test set for final known-class evaluation.
"""
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 6304

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def get_transforms():
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, eval_tf


def get_gcsc_train_transform():
    """GCSC's stronger recipe: RandAugment inserted after crop/flip, before ToTensor/Normalize."""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_cifar10_splits(data_root="./data", seed=SEED):
    """Returns (train_subset, val_subset, test_dataset), transforms applied by the caller
    via separate dataset instances (since train/val need different transforms on the SAME
    underlying indices)."""
    base_train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
    labels = np.array(base_train.targets)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    return train_idx, val_idx


class CIFAR10Indexed(torch.utils.data.Dataset):
    """CIFAR-10 restricted to a fixed set of indices, with its own transform."""
    def __init__(self, data_root, indices, train_partition, transform):
        self.base = torchvision.datasets.CIFAR10(root=data_root, train=train_partition, download=True)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_cifar10_datasets(data_root="./data", gcsc=False, seed=SEED):
    train_idx, val_idx = get_cifar10_splits(data_root, seed)
    train_tf, eval_tf = get_transforms()
    if gcsc:
        train_tf = get_gcsc_train_transform()

    train_ds = CIFAR10Indexed(data_root, train_idx, train_partition=True, transform=train_tf)
    val_ds = CIFAR10Indexed(data_root, val_idx, train_partition=True, transform=eval_tf)
    test_ds = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)
    return train_ds, val_ds, test_ds
