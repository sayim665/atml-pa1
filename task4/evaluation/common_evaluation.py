"""
Task 4, Step 6: Common Evaluation and Failure Analysis.
Assembles the final comparison table (Vanilla/GCSC/PROSER, MLS as common score + PROSER's
placeholder row), a compact score-distribution figure (MSP/MLS/Mahalanobis on Vanilla), and
pulls concrete near/far incorrectly-accepted failure examples using Vanilla's MLS threshold.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from cifar100_unknowns import NEAR_CLASSES, FAR_CLASSES


def build_comparison_table(results_dir):
    """Pulls together vanilla_scores_results.json, vanilla_step1_step2_results.json (GCSC,
    despite the generic filename -- see conversation note), and proser_scores_results.json
    into ONE final table: CSA + near/far/all AUROC + validation-calibrated rejection, using
    MLS as the common score (plus PROSER's placeholder score as an extra row)."""
    with open(os.path.join(results_dir, "vanilla_scores_results.json")) as f:
        vanilla = json.load(f)
    with open(os.path.join(results_dir, "vanilla_step1_step2_results.json")) as f:
        gcsc = json.load(f)  # NOTE: this file is actually GCSC's output -- see chat history
    with open(os.path.join(results_dir, "proser_scores_results.json")) as f:
        proser = json.load(f)

    table = {
        "Vanilla (MLS)": {
            "closed_set_accuracy": vanilla["clean_baseline"]["top1_acc"],
            **vanilla["post_hoc_scores"]["MLS"],
        },
        "GCSC (MLS)": {
            "closed_set_accuracy": gcsc["clean_baseline"]["top1_acc"],
            **gcsc["post_hoc_scores"]["MLS"],
        },
        "PROSER (MLS, known-only)": {
            "closed_set_accuracy": proser["closed_set_accuracy"],
            **proser["MLS_known_only"],
        },
        "PROSER (placeholder score)": {
            "closed_set_accuracy": proser["closed_set_accuracy"],
            **proser["placeholder_score"],
        },
    }

    with open(os.path.join(results_dir, "task4_final_comparison_table.json"), "w") as f:
        json.dump(table, f, indent=2)
    print(json.dumps(table, indent=2))
    return table


def plot_score_distributions(results_dir):
    """Compact multi-panel figure: MSP, MLS, Mahalanobis score distributions on Vanilla,
    known-test vs all-unknowns, per spec's Required Evidence."""
    raw = np.load(os.path.join(results_dir, "vanilla_raw_scores_backup.npz")
                   if os.path.exists(os.path.join(results_dir, "vanilla_raw_scores_backup.npz"))
                   else os.path.join(results_dir, "vanilla_raw_scores.npz"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, score_name in zip(axes, ["MSP", "MLS", "Mahalanobis"]):
        known = raw[f"test_{score_name}"]
        unknown = np.concatenate([raw[f"near_{score_name}"], raw[f"far_{score_name}"]])
        ax.hist(known, bins=40, alpha=0.6, label="known (test)", density=True)
        ax.hist(unknown, bins=40, alpha=0.6, label="unknown (near+far)", density=True)
        ax.set_title(f"Vanilla: {score_name} distribution")
        ax.legend()

    plt.tight_layout()
    png_path = os.path.join(results_dir, "score_distributions.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")
    return png_path


def find_failure_examples(results_dir, n_near=3, n_far=3):
    """Near/far unknowns that Vanilla's MLS-threshold rule incorrectly ACCEPTED (i.e. treated
    as known) -- concrete failure cases for the report, per spec's Required Evidence."""
    raw = np.load(os.path.join(results_dir, "vanilla_raw_scores_backup.npz")
                   if os.path.exists(os.path.join(results_dir, "vanilla_raw_scores_backup.npz"))
                   else os.path.join(results_dir, "vanilla_raw_scores.npz"))

    with open(os.path.join(results_dir, "vanilla_scores_results.json")) as f:
        vanilla = json.load(f)
    tau = vanilla["post_hoc_scores"]["MLS"]["threshold_tau"]

    near_scores = raw["near_MLS"]
    far_scores = raw["far_MLS"]

    near_accepted_idx = np.where(near_scores <= tau)[0]
    far_accepted_idx = np.where(far_scores <= tau)[0]

    failures = {
        "threshold_tau": float(tau),
        "near_incorrectly_accepted": [
            {"index": int(i), "mls_score": float(near_scores[i])}
            for i in near_accepted_idx[:n_near]
        ],
        "far_incorrectly_accepted": [
            {"index": int(i), "mls_score": float(far_scores[i])}
            for i in far_accepted_idx[:n_far]
        ],
        "note": ("Indices refer to position within the near/far unknown loader (NEAR_CLASSES/"
                 "FAR_CLASSES order in cifar100_unknowns.py). Re-run with dataset access to "
                 "pull the actual images/predicted-class for the report if needed."),
    }
    with open(os.path.join(results_dir, "task4_failure_examples.json"), "w") as f:
        json.dump(failures, f, indent=2)
    print(json.dumps(failures, indent=2))
    return failures


def main(results_dir):
    print("=== Building final comparison table ===")
    table = build_comparison_table(results_dir)
    print("\n=== Plotting score distributions ===")
    plot_score_distributions(results_dir)
    print("\n=== Finding failure examples ===")
    failures = find_failure_examples(results_dir)
    return table, failures


if __name__ == "__main__":
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task4/results"
    main(RESULTS_DIR)
