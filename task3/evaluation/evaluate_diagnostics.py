"""
Task 3 diagnostics -- source-domain separability + sharpness proxy, for ERM, DAN-DG, SAM.
Deliberately does NOT touch Sketch at all: these are source-side-only diagnostics, computed
and reported BEFORE final Sketch evaluation, per spec's ordering.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "methods"))
from pacs_protocol import SEED, SOURCE_DOMAINS, N_CLASSES
from pacs import get_pacs_datasets
from backbone import build_backbone, freeze_batchnorm_running_stats
from classifier_head import ClassifierHead
from source_only import get_transforms

METHODS = ["erm", "dan_dg", "sam"]
RHO_SHARPNESS = 0.05
N_PER_DOMAIN_SHARPNESS = 32


@torch.no_grad()
def collect_features(backbone, loader, device):
    backbone.eval()
    all_feats = []
    for imgs, _ in loader:
        all_feats.append(backbone(imgs.to(device)).cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def source_domain_separability(backbone, val_loaders, device, seed=SEED):
    """Multinomial logistic regression predicting Photo/Art/Cartoon (3-way), 70/30 split,
    seed 6304, C=1. Held-out accuracy is the score; chance = 33.3%."""
    X_list, y_list = [], []
    for i, dom in enumerate(SOURCE_DOMAINS):
        feats = collect_features(backbone, val_loaders[dom], device)
        X_list.append(feats)
        y_list.append(np.full(len(feats), i))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=1.0, max_iter=2000, multi_class="multinomial", random_state=seed)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def build_fixed_sharpness_batch(datasets, seed=SEED, n_per_domain=N_PER_DOMAIN_SHARPNESS):
    """Selects the SAME 32-per-domain (96 total) fixed batch, seed 6304, reused identically
    across ERM/DAN-DG/SAM so the sharpness comparison is apples-to-apples."""
    _, eval_tf = get_transforms()
    rng = np.random.RandomState(seed)
    all_imgs, all_labels = [], []
    for dom in SOURCE_DOMAINS:
        val_ds = datasets[dom]["val"]
        idx = rng.choice(len(val_ds), min(n_per_domain, len(val_ds)), replace=False)
        subset = Subset(val_ds, idx)
        loader = DataLoader(subset, batch_size=len(idx), shuffle=False)
        imgs, labels = next(iter(loader))
        all_imgs.append(imgs)
        all_labels.append(labels)
    return torch.cat(all_imgs), torch.cat(all_labels)


def sharpness_proxy(backbone, head, fixed_imgs, fixed_labels, device, rho=RHO_SHARPNESS):
    """Delta_sharp = L(theta + epsilon) - L(theta), epsilon = rho * grad / ||grad||_2,
    on the FIXED batch, model in eval mode (BatchNorm already frozen via freeze_batchnorm)."""
    backbone.eval(); head.eval()
    imgs = fixed_imgs.to(device)
    labels = fixed_labels.to(device)

    params = [p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad]

    # L(theta)
    for p in params:
        p.requires_grad_(True)
    logits = head(backbone(imgs))
    loss_clean = F.cross_entropy(logits, labels)
    grads = torch.autograd.grad(loss_clean, params)

    grad_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), p=2)
    scale = rho / (grad_norm + 1e-12)

    # theta -> theta + epsilon
    with torch.no_grad():
        for p, g in zip(params, grads):
            p.add_(g * scale)

    with torch.no_grad():
        logits_pert = head(backbone(imgs))
        loss_pert = F.cross_entropy(logits_pert, labels)

    # revert: theta + epsilon -> theta
    with torch.no_grad():
        for p, g in zip(params, grads):
            p.sub_(g * scale)

    return float(loss_pert.item() - loss_clean.item())


def main(task3_checkpoint_dir, results_dir, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    _, eval_tf = get_transforms()
    datasets = get_pacs_datasets(transform_train=None, transform_eval=eval_tf)
    val_loaders = {
        dom: DataLoader(datasets[dom]["val"], batch_size=64, shuffle=False, num_workers=2)
        for dom in SOURCE_DOMAINS
    }

    fixed_imgs, fixed_labels = build_fixed_sharpness_batch(datasets)
    print(f"Fixed sharpness batch: {fixed_imgs.size(0)} images "
          f"({N_PER_DOMAIN_SHARPNESS} per source domain), seed {SEED}")

    diagnostics = {}
    for method in METHODS:
        # ERM's checkpoint lives in Task 2's checkpoint dir (reused unchanged, never retrained);
        # DAN-DG and SAM live in Task 3's own checkpoint dir.
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

        sep_score = source_domain_separability(backbone, val_loaders, device)
        delta_sharp = sharpness_proxy(backbone, head, fixed_imgs, fixed_labels, device)

        diagnostics[method] = {
            "source_domain_separability_score": sep_score,
            "chance_level": 1.0 / len(SOURCE_DOMAINS),
            "delta_sharp": delta_sharp,
        }
        print(f"\n=== {method} ===")
        print(json.dumps(diagnostics[method], indent=2))

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "task3_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    print("\n\n=== FULL DIAGNOSTICS ===")
    print(json.dumps(diagnostics, indent=2))
    return diagnostics


if __name__ == "__main__":
    TASK3_CKPT_DIR = "/content/drive/MyDrive/atml_pa1_task3/checkpoints"
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task3/results"
    main(TASK3_CKPT_DIR, RESULTS_DIR)
