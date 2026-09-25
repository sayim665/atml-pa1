"""
Task 4 Step 6 requirement: at least 3 near-unknown and 3 far-unknown failures, each with
unknown class name, predicted known class, score, and threshold. This is a lightweight,
INFERENCE-ONLY script (no training) -- reuses the already-saved Vanilla checkpoint and the
already-computed MLS threshold, just re-extracts predictions + class names in one pass.
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from resnet_cifar import resnet18_cifar
from cifar100_unknowns import build_unknown_datasets

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def extract_with_class_names(model, loader, device):
    model.eval()
    all_logits, all_class_names = [], []
    for imgs, labels, class_names in loader:
        imgs = imgs.to(device)
        logits, _ = model(imgs)
        all_logits.append(logits.cpu().numpy())
        all_class_names.extend(class_names)
    return np.concatenate(all_logits), all_class_names


def mls_score(logits):
    return -logits.max(axis=1)


def main(data_root, checkpoint_path, tau_mls, results_dir, device_str="cuda", n_examples=5):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)

    model = resnet18_cifar(num_classes=10).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    near_ds, far_ds = build_unknown_datasets(data_root)
    near_loader = DataLoader(near_ds, batch_size=256, shuffle=False, num_workers=2)
    far_loader = DataLoader(far_ds, batch_size=256, shuffle=False, num_workers=2)

    near_logits, near_class_names = extract_with_class_names(model, near_loader, device)
    far_logits, far_class_names = extract_with_class_names(model, far_loader, device)

    near_scores = mls_score(near_logits)
    far_scores = mls_score(far_logits)
    near_preds = near_logits.argmax(axis=1)
    far_preds = far_logits.argmax(axis=1)

    def top_failures(scores, preds, class_names, k):
        accepted_idx = np.where(scores <= tau_mls)[0]
        # Most CONFIDENTLY accepted (lowest/most negative MLS score = most confident known-class prediction)
        ranked = accepted_idx[np.argsort(scores[accepted_idx])][:k]
        return [
            {
                "unknown_class": class_names[i],
                "predicted_known_class": CIFAR10_CLASSES[preds[i]],
                "mls_score": float(scores[i]),
                "threshold": float(tau_mls),
            }
            for i in ranked
        ]

    failures = {
        "threshold_tau_MLS": float(tau_mls),
        "near_incorrectly_accepted": top_failures(near_scores, near_preds, near_class_names, n_examples),
        "far_incorrectly_accepted": top_failures(far_scores, far_preds, far_class_names, n_examples),
        "near_total_incorrectly_accepted": int((near_scores <= tau_mls).sum()),
        "far_total_incorrectly_accepted": int((far_scores <= tau_mls).sum()),
    }

    print(json.dumps(failures, indent=2))
    with open(os.path.join(results_dir, "task4_failure_examples.json"), "w") as f:
        json.dump(failures, f, indent=2)
    return failures


if __name__ == "__main__":
    DATA_ROOT = "./data"
    CHECKPOINT_PATH = "checkpoints/vanilla.pth"
    TAU_MLS = -5.77525520324707  # from task4_final_comparison_table.json's Vanilla (MLS) row
    RESULTS_DIR = "results"
    main(DATA_ROOT, CHECKPOINT_PATH, TAU_MLS, RESULTS_DIR)
