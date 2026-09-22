"""
Task 2, Section 6 -- Controlled Design Study.
Sweeps lambda_MMD in {0.1, 1, 10} for DAN (the main comparison already used lambda_MMD=1,
i.e. the existing dan.pth checkpoint -- this script does NOT retrain that one, it reuses it,
and only trains the 0.1 and 10 variants, per the spec's instruction not to replace the main
comparison's setting with a post-hoc winner).

Everything else (seed, architecture, optimizer, schedule, batch composition) stays fixed --
only lambda_MMD changes. Reports source performance, domain separability, and target
performance/change for all three settings.
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
from pacs_protocol import SOURCE_DOMAINS, TARGET_DOMAIN, N_CLASSES
from pacs import get_pacs_datasets
from backbone import build_backbone
from classifier_head import ClassifierHead
from source_only import get_transforms, evaluate
from evaluate_final import get_predictions_and_features, domain_separability_score

import dan  # the DAN training module itself, to call dan.main() with different lambda_mmd

LAMBDAS = [0.1, 1.0, 10.0]


def main(results_dir, checkpoint_dir, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    _, eval_tf = get_transforms()
    datasets = get_pacs_datasets(transform_train=None, transform_eval=eval_tf)
    source_val_loaders = {
        dom: DataLoader(datasets[dom]["val"], batch_size=64, shuffle=False, num_workers=2)
        for dom in SOURCE_DOMAINS
    }
    target_loader = DataLoader(datasets[TARGET_DOMAIN]["all"], batch_size=64, shuffle=False, num_workers=2)

    sweep_results = {}

    for lam in LAMBDAS:
        tag = f"dan_lambda{lam}".replace(".0", "") if lam != 1.0 else "dan"  # lambda=1 reuses the existing dan.pth
        ckpt_path = os.path.join(checkpoint_dir, f"{tag}.pth")

        if not os.path.exists(ckpt_path):
            print(f"\n--- Training DAN with lambda_MMD={lam} (tag={tag}) ---")
            dan.main(results_dir, checkpoint_dir, lambda_mmd=lam, tag=tag)
        else:
            print(f"\n--- lambda_MMD={lam}: checkpoint {tag}.pth already exists, reusing (main comparison) ---")

        backbone, feat_dim = build_backbone(device)
        head = ClassifierHead(feat_dim, N_CLASSES).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])

        per_source = {}
        source_feats_all = []
        for dom in SOURCE_DOMAINS:
            acc, f1 = evaluate(backbone, head, source_val_loaders[dom], device)
            per_source[dom] = {"acc": acc, "macro_f1": f1}
            _, _, feats = get_predictions_and_features(backbone, head, source_val_loaders[dom], device)
            source_feats_all.append(feats)
        source_feats_all = np.concatenate(source_feats_all, axis=0)

        target_preds, target_labels, target_feats = get_predictions_and_features(
            backbone, head, target_loader, device
        )
        target_acc = float((target_preds == target_labels).mean())
        from sklearn.metrics import f1_score
        target_f1 = float(f1_score(target_labels, target_preds, average="macro"))
        sep_score = domain_separability_score(source_feats_all, target_feats)

        sweep_results[str(lam)] = {
            "per_source_val": per_source,
            "mean_source_val_acc": float(np.mean([per_source[d]["acc"] for d in SOURCE_DOMAINS])),
            "target_acc": target_acc,
            "target_macro_f1": target_f1,
            "domain_separability_score": sep_score,
        }
        print(json.dumps(sweep_results[str(lam)], indent=2))

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "task2_dan_lambda_sweep.json"), "w") as f:
        json.dump(sweep_results, f, indent=2)

    print("\n\n=== FULL LAMBDA_MMD SWEEP ===")
    print(json.dumps(sweep_results, indent=2))
    return sweep_results


if __name__ == "__main__":
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task2/results"
    CHECKPOINT_DIR = "/content/drive/MyDrive/atml_pa1_task2/checkpoints"
    main(RESULTS_DIR, CHECKPOINT_DIR)
