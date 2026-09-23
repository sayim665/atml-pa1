"""
DAN-DG -- pairwise MMD alignment across the THREE OBSERVED SOURCE DOMAINS ONLY.
Unlike Task 2's DAN, this method NEVER loads or touches Sketch (target) images or labels
at any point in training, checkpoint selection, or hyperparameter choice.
L_DAN-DG = L_ERM + (lambda_DG / 3) * sum over unordered source-domain pairs of MMD^2(pair)
Reuses the exact same MMD implementation/kernel construction as Task 2's DAN (task2/shared/mmd.py).
"""
import os
import sys
import json
import itertools
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
from mmd import mmd_loss
from source_only import set_seed, get_transforms, domain_balanced_iterator, evaluate

LAMBDA_DG = 1.0
SOURCE_PAIRS = list(itertools.combinations(SOURCE_DOMAINS, 2))  # 3 unordered pairs


def main(results_dir, checkpoint_dir, device_str="cuda", lambda_dg=LAMBDA_DG, tag="dan_dg"):
    set_seed(SEED)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_tf, eval_tf = get_transforms()
    # NOTE: get_pacs_datasets() only ever builds SOURCE-domain train/val + the target "all" split
    # for compatibility with Task 2's shared loader, but this script simply never constructs a
    # target loader below -- Sketch is never touched anywhere in this file.
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
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )

    steps_per_epoch = min(len(dl) for dl in train_loaders.values())
    batch_gen = domain_balanced_iterator(train_loaders)

    best_mean_val_f1, best_state, patience = -1, None, 0

    for epoch in range(MAX_EPOCHS):
        backbone.train(); head.train()
        freeze_batchnorm_running_stats(backbone)

        epoch_cls_loss, epoch_mmd_loss = 0.0, 0.0

        for _ in range(steps_per_epoch):
            batch = next(batch_gen)
            imgs_by_dom = {d: batch[d][0].to(device) for d in SOURCE_DOMAINS}
            labels_by_dom = {d: batch[d][1].to(device) for d in SOURCE_DOMAINS}

            all_imgs = torch.cat([imgs_by_dom[d] for d in SOURCE_DOMAINS], dim=0)
            all_labels = torch.cat([labels_by_dom[d] for d in SOURCE_DOMAINS], dim=0)

            optimizer.zero_grad()
            all_feats = backbone(all_imgs)
            n_per_dom = imgs_by_dom[SOURCE_DOMAINS[0]].size(0)
            feats_by_dom = {d: all_feats[i * n_per_dom:(i + 1) * n_per_dom] for i, d in enumerate(SOURCE_DOMAINS)}

            logits = head(all_feats)
            cls_loss = F.cross_entropy(logits, all_labels)

            pair_mmd = sum(mmd_loss(feats_by_dom[a], feats_by_dom[b]) for a, b in SOURCE_PAIRS) / len(SOURCE_PAIRS)
            loss = cls_loss + lambda_dg * pair_mmd

            loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_mmd_loss += pair_mmd.item()

        val_f1s = []
        for dom in SOURCE_DOMAINS:
            _, f1 = evaluate(backbone, head, val_loaders[dom], device)
            val_f1s.append(f1)
        mean_val_f1 = float(np.mean(val_f1s))

        print(f"Epoch {epoch+1}: cls_loss={epoch_cls_loss/steps_per_epoch:.4f}  "
              f"pairwise_mmd={epoch_mmd_loss/steps_per_epoch:.4f}  "
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
        "lambda_dg": lambda_dg,
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
