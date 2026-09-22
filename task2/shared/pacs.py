"""
PACS dataset loading + stratified 80/20 source splits (seed 6304).
Loads from the Hugging Face mirror `flwrlabs/pacs` (no manual zip download needed).
Shared by Task 2 (UDA) and Task 3 (DG) -- both MUST reuse the exact same source splits,
so this module is the single source of truth for that.
"""
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

from pacs_protocol import SEED, SOURCE_DOMAINS, TARGET_DOMAIN, CLASSES, CLASS_TO_IDX

SPLITS_PATH = os.path.join(os.path.dirname(__file__), "..", "splits", "pacs_sketch_seed6304.json")


def load_pacs_hf():
    """Loads the full PACS dataset via Hugging Face `datasets`. Returns a single
    HF Dataset with columns: image (PIL), domain (str), label (int, HF's own indexing)."""
    from datasets import load_dataset
    ds = load_dataset("flwrlabs/pacs", split="train")  # the HF mirror ships everything under one split
    return ds


class PACSDomainDataset(Dataset):
    """A torch Dataset over a single PACS domain's examples, given explicit indices
    into the underlying HF dataset. Remaps HF's own label ids to our CLASSES ordering
    so label indices are guaranteed consistent with pacs_protocol.CLASSES."""
    def __init__(self, hf_dataset, indices, transform=None):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.transform = transform
        # Build a mapping from the HF dataset's own label feature names -> our CLASSES order
        hf_label_names = hf_dataset.features["label"].names
        self.hf_idx_to_our_idx = {
            hf_i: CLASS_TO_IDX[name] for hf_i, name in enumerate(hf_label_names)
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.hf_dataset[int(self.indices[i])]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = self.hf_idx_to_our_idx[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_or_load_splits(hf_dataset, seed=SEED):
    """Builds (once) a stratified 80/20 split of each SOURCE domain's indices, seed 6304.
    Saved to splits/pacs_sketch_seed6304.json so Task 2 and Task 3 load the IDENTICAL split
    every time, rather than risking a different random draw on a later run.
    Returns: dict like {"photo": {"train": [...], "val": [...]}, ..., "sketch": {"all": [...]}}"""
    os.makedirs(os.path.dirname(SPLITS_PATH), exist_ok=True)

    if os.path.exists(SPLITS_PATH):
        with open(SPLITS_PATH) as f:
            return json.load(f)

    domains = np.array(hf_dataset["domain"])
    splits = {}

    for dom in SOURCE_DOMAINS:
        dom_indices = np.where(domains == dom)[0]
        dom_labels = np.array(hf_dataset["label"])[dom_indices]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_rel_idx, val_rel_idx = next(sss.split(np.zeros(len(dom_indices)), dom_labels))
        splits[dom] = {
            "train": dom_indices[train_rel_idx].tolist(),
            "val": dom_indices[val_rel_idx].tolist(),
        }

    target_indices = np.where(domains == TARGET_DOMAIN)[0]
    splits[TARGET_DOMAIN] = {"all": target_indices.tolist()}

    with open(SPLITS_PATH, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


def get_pacs_datasets(transform_train=None, transform_eval=None):
    """Convenience entry point: returns a dict of PACSDomainDataset objects:
      {"photo": {"train": ..., "val": ...}, "art_painting": {...}, "cartoon": {...},
       "sketch": {"all": ...}}
    Use transform_train for every *_train split, transform_eval for every *_val / target split."""
    hf_dataset = load_pacs_hf()
    splits = build_or_load_splits(hf_dataset)

    out = {}
    for dom in SOURCE_DOMAINS:
        out[dom] = {
            "train": PACSDomainDataset(hf_dataset, splits[dom]["train"], transform_train),
            "val": PACSDomainDataset(hf_dataset, splits[dom]["val"], transform_eval),
        }
    out[TARGET_DOMAIN] = {
        "all": PACSDomainDataset(hf_dataset, splits[TARGET_DOMAIN]["all"], transform_eval)
    }
    return out
