"""
Task 3, Section 5 -- Controlled Design Study: sweep lambda_DG in {0.1, 1, 10} for DAN-DG.
lambda_DG=1 (the main comparison) is REUSED from the existing dan_dg.pth checkpoint, not
retrained -- only 0.1 and 10 are trained fresh here, per spec's rule not to replace the main
setting with a post-hoc winner.

Per spec, this controlled study MAY look at Sketch as part of its analysis (unlike DAN-DG's
own training/selection, which never may) -- it runs strictly AFTER the main Section 4
comparison is already fixed, and its results must not be used to revise lambda_DG=1 or any
earlier Task 3 decision.
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "methods"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
from pacs_protocol import SOURCE_DOMAINS, TARGET_DOMAIN, N_CLASSES
from pacs import get_pacs_datasets
from backbone import build_backbone
from classifier_head import ClassifierHead
from source_only import get_transforms, evaluate
from evaluate_diagnostics import source_domain_separability

sys.path.insert(0, os.path.dirname(__file__))
import dan_dg

LAMBDAS = [0.1, 1.0, 10.0]


@torch.no_grad()
def get_predictions(backbone, head, loader, device):
    backbone.eval(); head.eval()
    preds, labels = [], []
    for imgs, ys in loader:
        logits = head(backbone(imgs.to(device)))
        preds.append(logits.argmax(-1).cpu())
        labels.append(ys)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def main(results_dir, checkpoint_dir, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    _, eval_tf = get_transforms()
    datasets = get_pacs_datasets(transform_train=None, transform_eval=eval_tf)
    val_loaders = {
        dom: DataLoader(datasets[dom]["val"], batch_size=64, shuffle=False, num_workers=2)
        for dom in SOURCE_DOMAINS
    }
    sketch_loader = DataLoader(datasets[TARGET_DOMAIN]["all"], batch_size=64, shuffle=False, num_workers=2)

    sweep_results = {}

    for lam in LAMBDAS:
        tag = "dan_dg" if lam == 1.0 else f"dan_dg_lambda{lam}".replace(".0", "")
        ckpt_path = os.path.join(checkpoint_dir, f"{tag}.pth")

        if not os.path.exists(ckpt_path):
            print(f"\n--- Training DAN-DG with lambda_DG={lam} (tag={tag}) ---")
            dan_dg.main(results_dir, checkpoint_dir, lambda_dg=lam, tag=tag)
        else:
            print(f"\n--- lambda_DG={lam}: checkpoint {tag}.pth already exists, reusing (main comparison) ---")

        backbone, feat_dim = build_backbone(device)
        head = ClassifierHead(feat_dim, N_CLASSES).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])

        per_source = {}
        for dom in SOURCE_DOMAINS:
            acc, f1 = evaluate(backbone, head, val_loaders[dom], device)
            per_source[dom] = {"acc": acc, "macro_f1": f1}
        mean_source_acc = float(np.mean([per_source[d]["acc"] for d in SOURCE_DOMAINS]))

        sep_score = source_domain_separability(backbone, val_loaders, device)

        sketch_preds, sketch_labels = get_predictions(backbone, head, sketch_loader, device)
        sketch_acc = float((sketch_preds == sketch_labels).mean())
        sketch_f1 = float(f1_score(sketch_labels, sketch_preds, average="macro"))

        sweep_results[str(lam)] = {
            "per_source_val": per_source,
            "mean_source_val_acc": mean_source_acc,
            "source_domain_separability_score": sep_score,
            "sketch_acc": sketch_acc,
            "sketch_macro_f1": sketch_f1,
        }
        print(json.dumps(sweep_results[str(lam)], indent=2))

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "task3_dan_dg_lambda_sweep.json"), "w") as f:
        json.dump(sweep_results, f, indent=2)

    print("\n\n=== FULL LAMBDA_DG SWEEP (source + Sketch analysis) ===")
    print(json.dumps(sweep_results, indent=2))
    return sweep_results


if __name__ == "__main__":
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task3/results"
    CHECKPOINT_DIR = "/content/drive/MyDrive/atml_pa1_task3/checkpoints"
    main(RESULTS_DIR, CHECKPOINT_DIR)
