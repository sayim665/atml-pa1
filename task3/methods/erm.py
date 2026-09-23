"""
Task 3 ERM baseline. Per spec: this IS the Task 2 Source-only checkpoint, loaded unchanged --
never retrained here. This script only re-evaluates it on the source validation domains
(mean/worst-domain) for Task 3's own reporting; Sketch is evaluated only in the final,
separate evaluate_sketch.py, never here or during any Task 3 model-selection step.
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "methods"))
from pacs_protocol import SOURCE_DOMAINS, N_CLASSES
from pacs import get_pacs_datasets
from backbone import build_backbone
from classifier_head import ClassifierHead
from source_only import get_transforms, evaluate


def main(task2_checkpoint_dir, results_dir, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)

    _, eval_tf = get_transforms()
    datasets = get_pacs_datasets(transform_train=None, transform_eval=eval_tf)
    val_loaders = {
        dom: DataLoader(datasets[dom]["val"], batch_size=64, shuffle=False, num_workers=2)
        for dom in SOURCE_DOMAINS
    }

    backbone, feat_dim = build_backbone(device)
    head = ClassifierHead(feat_dim, N_CLASSES).to(device)
    ckpt_path = os.path.join(task2_checkpoint_dir, "source_only.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    print(f"Loaded Task 2's source_only.pth UNCHANGED as Task 3's ERM baseline (from {ckpt_path})")

    per_source = {}
    for dom in SOURCE_DOMAINS:
        acc, f1 = evaluate(backbone, head, val_loaders[dom], device)
        per_source[dom] = {"acc": acc, "macro_f1": f1}

    accs = [per_source[d]["acc"] for d in SOURCE_DOMAINS]
    f1s = [per_source[d]["macro_f1"] for d in SOURCE_DOMAINS]
    results = {
        "per_source_val": per_source,
        "mean_source_acc": float(np.mean(accs)),
        "mean_source_macro_f1": float(np.mean(f1s)),
        "worst_source_acc": float(np.min(accs)),
        "worst_source_macro_f1": float(np.min(f1s)),
        "note": "This is Task 2's source_only checkpoint, reused unchanged (not retrained).",
    }
    print(json.dumps(results, indent=2))
    with open(os.path.join(results_dir, "erm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results, backbone, head


if __name__ == "__main__":
    TASK2_CKPT_DIR = "/content/drive/MyDrive/atml_pa1_task2/checkpoints"
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task3/results"
    main(TASK2_CKPT_DIR, RESULTS_DIR)
