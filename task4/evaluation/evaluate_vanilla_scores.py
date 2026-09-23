"""
Task 4, Step 1-2: Clean Baseline + Post-hoc Novelty Scores, on the FROZEN Vanilla model.
Extracts logits/features ONCE for train (unaugmented, for Mahalanobis fit), val (threshold
calibration), test (known-class eval), and the fixed near/far CIFAR-100 unknowns -- then
computes all four scores from that single extraction, per spec ("all four scores must use
exactly the same saved logits and features").
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scores"))
from resnet_cifar import resnet18_cifar
from cifar10 import build_cifar10_datasets, get_transforms
from cifar100_unknowns import build_unknown_datasets
from osr_scores import (extract_logits_and_features, fit_mahalanobis_params, compute_all_scores)

sys.path.insert(0, os.path.dirname(__file__))
from osr_metrics import evaluate_score_full

SCORE_NAMES = ["MSP", "MLS", "Energy", "Mahalanobis"]


def main(data_root, checkpoint_path, results_dir, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)

    model = resnet18_cifar(num_classes=10).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Rebuild datasets with the CLEAN (unaugmented) eval transform for train too -- Mahalanobis
    # explicitly requires UNAUGMENTED CIFAR-10 training features.
    train_ds, val_ds, test_ds = build_cifar10_datasets(data_root, gcsc=False)
    _, eval_tf = get_transforms()
    train_ds.transform = eval_tf  # override: unaugmented, per spec

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    near_ds, far_ds = build_unknown_datasets(data_root)
    near_loader = DataLoader(near_ds, batch_size=256, shuffle=False, num_workers=2)
    far_loader = DataLoader(far_ds, batch_size=256, shuffle=False, num_workers=2)

    print("Extracting features/logits for train (clean), val, test, near, far...")
    train_logits, train_feats, train_labels = extract_logits_and_features(model, train_loader, device)
    val_logits, val_feats, val_labels = extract_logits_and_features(model, val_loader, device)
    test_logits, test_feats, test_labels = extract_logits_and_features(model, test_loader, device)
    near_logits, near_feats, near_labels = extract_logits_and_features(model, near_loader, device)
    far_logits, far_feats, far_labels = extract_logits_and_features(model, far_loader, device)

    # --- Step 1: Clean Baseline ---
    test_preds = test_logits.argmax(-1)
    test_acc = float((test_preds == test_labels).mean())
    test_f1 = float(f1_score(test_labels, test_preds, average="macro"))
    test_probs = torch.softmax(torch.from_numpy(test_logits), dim=-1).numpy()
    mean_max_conf = float(test_probs.max(axis=1).mean())

    clean_baseline = {"top1_acc": test_acc, "macro_f1": test_f1, "mean_max_confidence": mean_max_conf}
    print("Clean baseline:", json.dumps(clean_baseline, indent=2))

    # --- Step 2: Post-hoc scores ---
    print("Fitting Mahalanobis params on UNAUGMENTED train features...")
    mahal_means, mahal_inv_diag_cov = fit_mahalanobis_params(train_feats, train_labels, n_classes=10)

    val_scores = compute_all_scores(val_logits, val_feats, mahal_means, mahal_inv_diag_cov)
    test_scores = compute_all_scores(test_logits, test_feats, mahal_means, mahal_inv_diag_cov)
    near_scores = compute_all_scores(near_logits, near_feats, mahal_means, mahal_inv_diag_cov)
    far_scores = compute_all_scores(far_logits, far_feats, mahal_means, mahal_inv_diag_cov)

    score_comparison = {}
    for name in SCORE_NAMES:
        score_comparison[name] = evaluate_score_full(
            val_scores[name], test_scores[name], near_scores[name], far_scores[name]
        )
        print(f"\n=== {name} ===")
        print(json.dumps(score_comparison[name], indent=2))

    results = {
        "clean_baseline": clean_baseline,
        "post_hoc_scores": score_comparison,
    }
    with open(os.path.join(results_dir, "vanilla_step1_step2_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save raw score arrays too, for the score-distribution / ROC figure later.
    np.savez(
        os.path.join(results_dir, "vanilla_raw_scores.npz"),
        **{f"val_{k}": v for k, v in val_scores.items()},
        **{f"test_{k}": v for k, v in test_scores.items()},
        **{f"near_{k}": v for k, v in near_scores.items()},
        **{f"far_{k}": v for k, v in far_scores.items()},
    )

    print("\n\nDone. Saved to vanilla_step1_step2_results.json + vanilla_raw_scores.npz")
    return results


if __name__ == "__main__":
    DATA_ROOT = "./data"
    CHECKPOINT_PATH = "/content/drive/MyDrive/atml_pa1_task4/checkpoints/vanilla.pth"
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task4/results"
    main(DATA_ROOT, CHECKPOINT_PATH, RESULTS_DIR)
