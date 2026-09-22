"""
Task 2, Section 5 -- Common Evaluation and Alignment Diagnostic.
Loads all four trained checkpoints (source_only, dan, dann, cdan) and produces:
  1. The comparison table: per-source val acc/F1, target acc/F1, target change vs Source-only.
  2. Domain separability score per method: freeze the backbone, collect equal numbers of
     source-val and target features, 70/30 split (seed 6304), balanced logistic regression
     (C=1) to distinguish source vs target -- held-out accuracy is the separability score.
  3. Per-class target accuracy for each method, with the largest improvements/degradations
     relative to Source-only and their dominant confusions.

IMPORTANT: this script only READS existing checkpoints -- it does not retrain anything, and
never uses target LABELS for anything except this final analysis stage, per spec.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "methods"))
from pacs_protocol import SEED, SOURCE_DOMAINS, TARGET_DOMAIN, CLASSES, N_CLASSES
from pacs import get_pacs_datasets
from backbone import build_backbone
from classifier_head import ClassifierHead
from source_only import get_transforms, evaluate

METHODS = ["source_only", "dan", "dann", "cdan"]


@torch.no_grad()
def get_predictions_and_features(backbone, head, loader, device):
    backbone.eval(); head.eval()
    all_preds, all_labels, all_feats = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)
        logits = head(feats)
        all_preds.append(logits.argmax(-1).cpu())
        all_labels.append(labels)
        all_feats.append(feats.cpu())
    return (torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy(),
            torch.cat(all_feats).numpy())


def domain_separability_score(source_feats, target_feats, seed=SEED):
    """Balanced logistic regression, source vs target, 70/30 split, C=1.
    Returns held-out accuracy; 50% = chance (domain info fully removed)."""
    n = min(len(source_feats), len(target_feats))  # balance classes
    rng = np.random.RandomState(seed)
    src_idx = rng.choice(len(source_feats), n, replace=False)
    tgt_idx = rng.choice(len(target_feats), n, replace=False)

    X = np.concatenate([source_feats[src_idx], target_feats[tgt_idx]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def per_class_accuracy(preds, labels, class_names=CLASSES):
    out = {}
    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.sum() == 0:
            out[name] = None
            continue
        out[name] = float((preds[mask] == labels[mask]).mean())
    return out


def main(results_dir, checkpoint_dir, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    _, eval_tf = get_transforms()
    datasets = get_pacs_datasets(transform_train=None, transform_eval=eval_tf)

    # Balanced source-val pool across the 3 source domains, for domain-separability comparison.
    source_val_loaders = {
        dom: DataLoader(datasets[dom]["val"], batch_size=64, shuffle=False, num_workers=2)
        for dom in SOURCE_DOMAINS
    }
    target_loader = DataLoader(datasets[TARGET_DOMAIN]["all"], batch_size=64, shuffle=False, num_workers=2)

    comparison_table = {}
    per_class_table = {}
    confusions_table = {}

    source_only_target_preds = None  # for computing per-class DELTA vs Source-only

    for method in METHODS:
        ckpt_path = os.path.join(checkpoint_dir, f"{method}.pth")
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found, skipping {method}")
            continue

        backbone, feat_dim = build_backbone(device)
        head = ClassifierHead(feat_dim, N_CLASSES).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])

        # Per-source-domain val acc/F1
        per_source = {}
        source_val_feats_all = []
        for dom in SOURCE_DOMAINS:
            acc, f1 = evaluate(backbone, head, source_val_loaders[dom], device)
            per_source[dom] = {"acc": acc, "macro_f1": f1}
            _, _, feats = get_predictions_and_features(backbone, head, source_val_loaders[dom], device)
            source_val_feats_all.append(feats)
        source_val_feats_all = np.concatenate(source_val_feats_all, axis=0)

        mean_source_acc = float(np.mean([per_source[d]["acc"] for d in SOURCE_DOMAINS]))
        mean_source_f1 = float(np.mean([per_source[d]["macro_f1"] for d in SOURCE_DOMAINS]))

        # Target
        target_preds, target_labels, target_feats = get_predictions_and_features(
            backbone, head, target_loader, device
        )
        target_acc = float((target_preds == target_labels).mean())
        from sklearn.metrics import f1_score
        target_f1 = float(f1_score(target_labels, target_preds, average="macro"))

        # Domain separability
        sep_score = domain_separability_score(source_val_feats_all, target_feats)

        # Per-class target accuracy
        pc_acc = per_class_accuracy(target_preds, target_labels)
        per_class_table[method] = pc_acc

        # Confusion matrix (target)
        cm = confusion_matrix(target_labels, target_preds, labels=list(range(N_CLASSES)))
        confusions_table[method] = cm.tolist()

        if method == "source_only":
            source_only_target_preds = target_preds.copy()
            target_change = 0.0
        else:
            target_change = target_acc - comparison_table["source_only"]["target_acc"]

        comparison_table[method] = {
            "per_source_val": per_source,
            "mean_source_val_acc": mean_source_acc,
            "mean_source_val_macro_f1": mean_source_f1,
            "target_acc": target_acc,
            "target_macro_f1": target_f1,
            "target_acc_change_vs_source_only": target_change,
            "domain_separability_score": sep_score,
        }

        print(f"\n=== {method} ===")
        print(json.dumps(comparison_table[method], indent=2))

    # Per-class DELTA vs source_only
    per_class_delta = {}
    if "source_only" in per_class_table:
        for method in METHODS:
            if method == "source_only" or method not in per_class_table:
                continue
            per_class_delta[method] = {
                c: (per_class_table[method][c] - per_class_table["source_only"][c])
                if per_class_table[method][c] is not None and per_class_table["source_only"][c] is not None
                else None
                for c in CLASSES
            }

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "task2_comparison_table.json"), "w") as f:
        json.dump(comparison_table, f, indent=2)
    with open(os.path.join(results_dir, "task2_per_class_accuracy.json"), "w") as f:
        json.dump(per_class_table, f, indent=2)
    with open(os.path.join(results_dir, "task2_per_class_delta_vs_source_only.json"), "w") as f:
        json.dump(per_class_delta, f, indent=2)
    with open(os.path.join(results_dir, "task2_confusion_matrices.json"), "w") as f:
        json.dump({"class_order": CLASSES, "matrices": confusions_table}, f, indent=2)

    print("\n\n=== PER-CLASS DELTA vs Source-only ===")
    print(json.dumps(per_class_delta, indent=2))

    return comparison_table, per_class_table, per_class_delta, confusions_table


if __name__ == "__main__":
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task2/results"
    CHECKPOINT_DIR = "/content/drive/MyDrive/atml_pa1_task2/checkpoints"
    main(RESULTS_DIR, CHECKPOINT_DIR)
