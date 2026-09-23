"""
SAM training script: standard, non-adaptive SAM on the ERM objective, rho=0.05 for the main
comparison. Uses the shared frozen-BatchNorm policy during BOTH SAM passes, per spec.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "methods"))
from pacs_protocol import (SEED, SOURCE_DOMAINS, N_CLASSES,
                            BATCH_SIZE_PER_SOURCE_DOMAIN, MAX_EPOCHS, LR, WEIGHT_DECAY,
                            EARLY_STOP_PATIENCE)
from pacs import get_pacs_datasets
from backbone import build_backbone, freeze_batchnorm_running_stats
from classifier_head import ClassifierHead
from source_only import set_seed, get_transforms, domain_balanced_iterator, evaluate

sys.path.insert(0, os.path.dirname(__file__))
from sam_optimizer import SAM

RHO = 0.05


def main(results_dir, checkpoint_dir, device_str="cuda", rho=RHO, tag="sam"):
    set_seed(SEED)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_tf, eval_tf = get_transforms()
    datasets = get_pacs_datasets(transform_train=train_tf, transform_eval=eval_tf)
    train_loaders = {
        dom: DataLoader(datasets[dom]["train"], batch_size=BATCH_SIZE_PER_SOURCE_DOMAIN,
                         shuffle=True, num_workers=2, drop_last=True)
        for dom in SOURCE_DOMAINS
    }
    val_loaders = {
        dom: DataLoader(datasets[dom]["val"], batch_size=64, shuffle=False, num_workers=2)
        for dom in SOURCE_DOMAINS
    }

    backbone, feat_dim = build_backbone(device)
    head = ClassifierHead(feat_dim, N_CLASSES).to(device)
    all_params = list(backbone.parameters()) + list(head.parameters())
    optimizer = SAM(all_params, base_optimizer_cls=torch.optim.AdamW, rho=rho,
                     lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = min(len(dl) for dl in train_loaders.values())
    batch_gen = domain_balanced_iterator(train_loaders)

    best_mean_val_f1, best_state, patience = -1, None, 0

    for epoch in range(MAX_EPOCHS):
        backbone.train(); head.train()
        freeze_batchnorm_running_stats(backbone)  # applies to BOTH SAM passes below

        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            batch = next(batch_gen)
            imgs = torch.cat([batch[d][0] for d in SOURCE_DOMAINS], dim=0).to(device)
            labels = torch.cat([batch[d][1] for d in SOURCE_DOMAINS], dim=0).to(device)

            # --- SAM pass 1: compute grad at theta, ascend to theta + epsilon ---
            logits = head(backbone(imgs))
            loss1 = F.cross_entropy(logits, labels)
            loss1.backward()
            optimizer.first_step(zero_grad=True)
            freeze_batchnorm_running_stats(backbone)  # re-apply: no-op for stats, but safe after any internal state touch

            # --- SAM pass 2: compute grad AT theta+epsilon, revert, take the real AdamW step ---
            logits2 = head(backbone(imgs))
            loss2 = F.cross_entropy(logits2, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            epoch_loss += loss1.item()

        val_f1s = []
        for dom in SOURCE_DOMAINS:
            _, f1 = evaluate(backbone, head, val_loaders[dom], device)
            val_f1s.append(f1)
        mean_val_f1 = float(np.mean(val_f1s))

        print(f"Epoch {epoch+1}: loss={epoch_loss/steps_per_epoch:.4f}  "
              f"mean source val macro-F1={mean_val_f1:.4f}")

        if mean_val_f1 > best_mean_val_f1:
            best_mean_val_f1 = mean_val_f1
            best_state = {
                "backbone": {k: v.clone() for k, v in backbone.state_dict().items()},
                "head": {k: v.clone() for k, v in head.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    backbone.load_state_dict(best_state["backbone"])
    head.load_state_dict(best_state["head"])
    torch.save(best_state, os.path.join(checkpoint_dir, f"{tag}.pth"))

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
        "rho": rho,
        "best_mean_source_val_f1": best_mean_val_f1,
        "note": "Sketch was NEVER loaded anywhere in this script.",
    }
    print(json.dumps(results, indent=2))
    with open(os.path.join(results_dir, f"{tag}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task3/results"
    CHECKPOINT_DIR = "/content/drive/MyDrive/atml_pa1_task3/checkpoints"
    main(RESULTS_DIR, CHECKPOINT_DIR)
