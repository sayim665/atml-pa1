"""
Shared OSR evaluation: AUROC (Known vs Near / Far / All) and validation-calibrated rejection
(95th percentile of unknownness on CIFAR-10 VAL, accept x when u(x) <= tau), per spec.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def auroc_known_vs_unknown(known_scores, unknown_scores):
    """known_scores/unknown_scores: 1D arrays of u(x) (larger = more novel).
    Label knowns as 0, unknowns as 1 -- AUROC measures how well u(x) ranks unknowns above knowns."""
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_score = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(y_true, y_score))


def calibrate_threshold(val_scores, percentile=95.0):
    """95th percentile of unknownness on CIFAR-10 VALIDATION (known-only) scores.
    Accepting x when u(x) <= tau accepts ~95% of KNOWN validation examples by construction."""
    return float(np.percentile(val_scores, percentile))


def apply_threshold(scores, tau):
    """Returns the acceptance mask: True = accepted (treated as known)."""
    return scores <= tau


def evaluate_score_full(known_val_scores, known_test_scores, near_scores, far_scores):
    """Everything Section 6 / the common evaluation table needs for ONE score on ONE model."""
    all_unknown_scores = np.concatenate([near_scores, far_scores])

    aurocs = {
        "near": auroc_known_vs_unknown(known_test_scores, near_scores),
        "far": auroc_known_vs_unknown(known_test_scores, far_scores),
        "all": auroc_known_vs_unknown(known_test_scores, all_unknown_scores),
    }

    tau = calibrate_threshold(known_val_scores, percentile=95.0)

    known_test_accept = apply_threshold(known_test_scores, tau)
    near_accept = apply_threshold(near_scores, tau)
    far_accept = apply_threshold(far_scores, tau)

    result = {
        "auroc": aurocs,
        "threshold_tau": tau,
        "known_test_acceptance_rate": float(known_test_accept.mean()),
        "near_unknown_acceptance_rate": float(near_accept.mean()),   # = FPR@95TPR(near)
        "far_unknown_acceptance_rate": float(far_accept.mean()),     # = FPR@95TPR(far)
        "near_unknown_rejection_rate": float(1 - near_accept.mean()),
        "far_unknown_rejection_rate": float(1 - far_accept.mean()),
    }
    return result
