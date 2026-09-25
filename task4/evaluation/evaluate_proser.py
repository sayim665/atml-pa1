"""
Task 4, Step 4 continued: evaluate PROSER two ways, per spec:
  (a) MLS on the KNOWN-class logits only (directly comparable to Vanilla/GCSC's MLS row).
  (b) PROSER's own placeholder-based detection score, as an additional row.
CSA (closed-set accuracy) uses ONLY the 10 known-class logits, per spec.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from resnet_cifar import resnet18_cifar
from cifar10 import build_cifar10_datasets, get_transforms
from cifar100_unknowns import build_unknown_datasets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "methods"))
from proser import ProserHead, N_KNOWN, N_DUMMY

sys.path.insert(0, os.path.dirname(__file__))
from osr_metrics import evaluate_score_full

import torch.nn as nn


@torch.no_grad()
def extract_full_logits(model, head, loader, device):
    model.eval(); head.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        imgs, labels = batch[0], batch[1]
        imgs = imgs.to(device)
        feat = model.forward_features(imgs)
        logits = head(feat)  # (B, N_KNOWN + N_DUMMY)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(np.asarray(labels))
    return np.concatenate(all_logits), np.concatenate(all_labels)


def mls_known_only(full_logits):
    known_logits = full_logits[:, :N_KNOWN]
    return -known_logits.max(axis=1)  # u_MLS on known-only logits


def placeholder_score(full_logits):
    """u_placeholder(x) = max(dummy_logits) - max(known_logits). Larger => the strongest
    dummy response outweighs the strongest known-class response => more novel/unknown.
    This is the standard combination used in PROSER reference implementations: the dummy
    classifiers directly compete with the known classes for "winning" the example."""
    known_max = full_logits[:, :N_KNOWN].max(axis=1)
    dummy_max = full_logits[:, N_KNOWN:].max(axis=1)
    return dummy_max - known_max


def main(data_root, proser_checkpoint_path, results_dir, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)

    model = resnet18_cifar(num_classes=N_KNOWN).to(device)
    model.fc = nn.Identity()  # PROSER bypassed the backbone's own fc during training
    head = ProserHead(model.feat_dim).to(device)

    ckpt = torch.load(proser_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])

    _, eval_tf = get_transforms()
    train_ds, val_ds, test_ds = build_cifar10_datasets(data_root, gcsc=False)
    train_ds.transform = eval_tf
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    near_ds, far_ds = build_unknown_datasets(data_root)
    near_loader = DataLoader(near_ds, batch_size=256, shuffle=False, num_workers=2)
    far_loader = DataLoader(far_ds, batch_size=256, shuffle=False, num_workers=2)

    val_logits, val_labels = extract_full_logits(model, head, val_loader, device)
    test_logits, test_labels = extract_full_logits(model, head, test_loader, device)
    near_logits, near_labels = extract_full_logits(model, head, near_loader, device)
    far_logits, far_labels = extract_full_logits(model, head, far_loader, device)

    # Closed-set accuracy: KNOWN-class logits only.
    test_known_preds = test_logits[:, :N_KNOWN].argmax(axis=1)
    csa = float((test_known_preds == test_labels).mean())
    csa_f1 = float(f1_score(test_labels, test_known_preds, average="macro"))

    results = {"closed_set_accuracy": csa, "closed_set_macro_f1": csa_f1}

    # Row (a): MLS on known-only logits.
    mls_val = mls_known_only(val_logits)
    mls_test = mls_known_only(test_logits)
    mls_near = mls_known_only(near_logits)
    mls_far = mls_known_only(far_logits)
    results["MLS_known_only"] = evaluate_score_full(mls_val, mls_test, mls_near, mls_far)

    # Row (b): PROSER's own placeholder-based score.
    ph_val = placeholder_score(val_logits)
    ph_test = placeholder_score(test_logits)
    ph_near = placeholder_score(near_logits)
    ph_far = placeholder_score(far_logits)
    results["placeholder_score"] = evaluate_score_full(ph_val, ph_test, ph_near, ph_far)

    print(json.dumps(results, indent=2))
    with open(os.path.join(results_dir, "proser_scores_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    DATA_ROOT = "./data"
    PROSER_CKPT = "/content/drive/MyDrive/atml_pa1_task4/checkpoints/proser.pth"
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task4/results"
    main(DATA_ROOT, PROSER_CKPT, RESULTS_DIR)
