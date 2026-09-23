"""
Post-hoc OSR scores on a FIXED (Vanilla) model. Larger u(x) = more novel/unknown.
All four scores use exactly the same saved logits/features, per spec.
"""
import torch
import torch.nn.functional as F
import numpy as np


def msp_score(logits):
    """u_MSP(x) = 1 - max_k p_k(x). Normalized confidence."""
    probs = F.softmax(logits, dim=-1)
    return (1.0 - probs.max(dim=-1).values)


def mls_score(logits):
    """u_MLS(x) = -max_k z_k(x). Retains absolute logit magnitude."""
    return -logits.max(dim=-1).values


def energy_score(logits):
    """u_Energy(x) = -log sum_k exp(z_k(x)). Uses ALL logits, not just the max."""
    return -torch.logsumexp(logits, dim=-1)


def fit_mahalanobis_params(train_feats, train_labels, n_classes, eps=1e-6):
    """Estimate class means mu_c and ONE shared diagonal covariance Sigma from
    UNAUGMENTED CIFAR-10 training features. Returns (means: (C,D), inv_diag_cov: (D,))."""
    feat_dim = train_feats.shape[1]
    means = np.zeros((n_classes, feat_dim), dtype=np.float64)
    for c in range(n_classes):
        class_feats = train_feats[train_labels == c]
        means[c] = class_feats.mean(axis=0)

    # Shared diagonal covariance: pool all classes' centered features together.
    centered = np.zeros_like(train_feats, dtype=np.float64)
    for c in range(n_classes):
        mask = train_labels == c
        centered[mask] = train_feats[mask] - means[c]
    diag_var = centered.var(axis=0) + eps  # add eps to every diagonal entry, per spec
    inv_diag_cov = 1.0 / diag_var
    return means, inv_diag_cov


def mahalanobis_score(feats, means, inv_diag_cov):
    """u_Mah(x) = min_c (f(x) - mu_c)^T Sigma^-1 (f(x) - mu_c), diagonal Sigma.
    feats: (N, D) numpy array. Returns (N,) numpy array."""
    n_classes = means.shape[0]
    dists = np.zeros((feats.shape[0], n_classes))
    for c in range(n_classes):
        diff = feats - means[c]                      # (N, D)
        dists[:, c] = np.sum(diff * diff * inv_diag_cov, axis=1)  # diagonal Mahalanobis
    return dists.min(axis=1)


@torch.no_grad()
def extract_logits_and_features(model, loader, device):
    """Runs a fixed model over a loader once; returns (logits, feats, labels_or_None) as numpy."""
    model.eval()
    all_logits, all_feats, all_labels = [], [], []
    for batch in loader:
        if len(batch) == 3:
            imgs, labels, _ = batch  # CIFAR-100 unknown loader also returns class_name
        else:
            imgs, labels = batch
        imgs = imgs.to(device)
        logits, feats = model(imgs)
        all_logits.append(logits.cpu().numpy())
        all_feats.append(feats.cpu().numpy())
        all_labels.append(np.asarray(labels))
    return np.concatenate(all_logits), np.concatenate(all_feats), np.concatenate(all_labels)


def compute_all_scores(logits, feats, mahal_means, mahal_inv_diag_cov):
    """Returns a dict of the four u(x) arrays, all computed from the SAME saved logits/feats."""
    logits_t = torch.from_numpy(logits)
    return {
        "MSP": msp_score(logits_t).numpy(),
        "MLS": mls_score(logits_t).numpy(),
        "Energy": energy_score(logits_t).numpy(),
        "Mahalanobis": mahalanobis_score(feats, mahal_means, mahal_inv_diag_cov),
    }
