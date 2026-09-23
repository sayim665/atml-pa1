"""
Task 3 FINAL Sketch evaluation. This is the ONLY script in Task 3 that loads Sketch labels.
It must only be run after every Task 3 model, setting, and checkpoint has been fixed --
this script does not select checkpoints, does not tune anything, and its results must never
be used to revise any earlier Task 3 decision (per spec).
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "methods"))
from pacs_protocol import SOURCE_DOMAINS, TARGET_DOMAIN, CLASSES, N_CLASSES
from pacs import get_pacs_datasets
from backbone import build_backbone
from classifier_head import ClassifierHead
from source_only import get_transforms

METHODS = ["erm", "dan_dg", "sam"]


@torch.no_grad()
def get_predictions(backbone, head, loader, device):
    backbone.eval(); head.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        logits = head(backbone(imgs.to(device)))
        all_preds.append(logits.argmax(-1).cpu())
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def per_class_accuracy(preds, labels, class_names=CLASSES):
    out = {}
    for i, name in enumerate(class_names):
        mask = labels == i
        out[name] = float((preds[mask] == labels[mask]).mean()) if mask.sum() > 0 else None
    return out


def main(task3_checkpoint_dir, results_dir, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    _, eval_tf = get_transforms()
    datasets = get_pacs_datasets(transform_train=None, transform_eval=eval_tf)
    sketch_loader = DataLoader(datasets[TARGET_DOMAIN]["all"], batch_size=64, shuffle=False, num_workers=2)

    comparison = {}
    per_class = {}
    confusions = {}

    for method in METHODS:
        if method == "erm":
            ckpt_path = os.path.join(
                task3_checkpoint_dir.replace("atml_pa1_task3", "atml_pa1_task2"), "source_only.pth"
            )
        else:
            ckpt_path = os.path.join(task3_checkpoint_dir, f"{method}.pth")

        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found, skipping {method}")
            continue

        backbone, feat_dim = build_backbone(device)
        head = ClassifierHead(feat_dim, N_CLASSES).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])

        preds, labels = get_predictions(backbone, head, sketch_loader, device)
        acc = float((preds == labels).mean())
        f1 = float(f1_score(labels, preds, average="macro"))

        per_class[method] = per_class_accuracy(preds, labels)
        cm = confusion_matrix(labels, preds, labels=list(range(N_CLASSES)))
        confusions[method] = cm.tolist()

        comparison[method] = {"sketch_acc": acc, "sketch_macro_f1": f1}
        print(f"\n=== {method} on Sketch ===")
        print(json.dumps(comparison[method], indent=2))

    # Change relative to ERM
    if "erm" in comparison:
        for method in METHODS:
            if method == "erm" or method not in comparison:
                continue
            comparison[method]["sketch_acc_change_vs_erm"] = (
                comparison[method]["sketch_acc"] - comparison["erm"]["sketch_acc"]
            )
            comparison[method]["sketch_f1_change_vs_erm"] = (
                comparison[method]["sketch_macro_f1"] - comparison["erm"]["sketch_macro_f1"]
            )
        comparison["erm"]["sketch_acc_change_vs_erm"] = 0.0
        comparison["erm"]["sketch_f1_change_vs_erm"] = 0.0

    # Per-class delta vs ERM
    per_class_delta = {}
    if "erm" in per_class:
        for method in METHODS:
            if method == "erm" or method not in per_class:
                continue
            per_class_delta[method] = {
                c: (per_class[method][c] - per_class["erm"][c])
                if per_class[method][c] is not None and per_class["erm"][c] is not None
                else None
                for c in CLASSES
            }

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "task3_sketch_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    with open(os.path.join(results_dir, "task3_sketch_per_class.json"), "w") as f:
        json.dump(per_class, f, indent=2)
    with open(os.path.join(results_dir, "task3_sketch_per_class_delta_vs_erm.json"), "w") as f:
        json.dump(per_class_delta, f, indent=2)
    with open(os.path.join(results_dir, "task3_sketch_confusion_matrices.json"), "w") as f:
        json.dump({"class_order": CLASSES, "matrices": confusions}, f, indent=2)

    print("\n\n=== FINAL COMPARISON (Sketch) ===")
    print(json.dumps(comparison, indent=2))
    print("\n=== PER-CLASS DELTA vs ERM ===")
    print(json.dumps(per_class_delta, indent=2))

    return comparison, per_class, per_class_delta, confusions


if __name__ == "__main__":
    TASK3_CKPT_DIR = "/content/drive/MyDrive/atml_pa1_task3/checkpoints"
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task3/results"
    main(TASK3_CKPT_DIR, RESULTS_DIR)
