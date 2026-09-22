"""
DAN -- MMD alignment between pooled source features and unlabeled target features,
applied to the 512-dim feature immediately before the classifier head.
L_DAN = L_cls + lambda_MMD * MMD^2(source_feat, target_feat)
lambda_MMD = 1 for the main comparison; RBF bandwidths per mmd.py.
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from pacs_protocol import (SEED, SOURCE_DOMAINS, TARGET_DOMAIN, N_CLASSES,
                            IMG_SIZE, RESIZE_SIZE, BATCH_SIZE_PER_SOURCE_DOMAIN,
                            BATCH_SIZE_TARGET, MAX_EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE)
from pacs import get_pacs_datasets
from backbone import build_backbone, freeze_batchnorm_running_stats
from classifier_head import ClassifierHead
from mmd import mmd_loss

# Reuse set_seed / get_transforms / domain_balanced_iterator / evaluate from source_only.py
sys.path.insert(0, os.path.dirname(__file__))
from source_only import set_seed, get_transforms, domain_balanced_iterator, evaluate

LAMBDA_MMD = 1.0


def target_cycle_iterator(loader):
    it = iter(loader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(loader)
            yield next(it)


def main(results_dir, checkpoint_dir, device_str="cuda", lambda_mmd=LAMBDA_MMD, tag="dan"):
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
    # Target used WITHOUT labels during adaptation -- train-style augmentation is fine here since
    # we never touch target labels; only the images matter for MMD.
    target_train_loader = DataLoader(datasets[TARGET_DOMAIN]["all"], batch_size=BATCH_SIZE_TARGET,
                                      shuffle=True, num_workers=2, drop_last=True)
    target_eval_loader = DataLoader(datasets[TARGET_DOMAIN]["all"], batch_size=64,
                                     shuffle=False, num_workers=2)

    backbone, feat_dim = build_backbone(device)
    head = ClassifierHead(feat_dim, N_CLASSES).to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )

    steps_per_epoch = min(len(dl) for dl in train_loaders.values())
    source_batch_gen = domain_balanced_iterator(train_loaders)
    target_gen = target_cycle_iterator(target_train_loader)

    best_mean_val_f1, best_state, patience = -1, None, 0

    for epoch in range(MAX_EPOCHS):
        backbone.train(); head.train()
        freeze_batchnorm_running_stats(backbone)

        epoch_cls_loss, epoch_mmd_loss = 0.0, 0.0

        for _ in range(steps_per_epoch):
            src_batch = next(source_batch_gen)
            src_imgs = torch.cat([src_batch[d][0] for d in SOURCE_DOMAINS], dim=0).to(device)
            src_labels = torch.cat([src_batch[d][1] for d in SOURCE_DOMAINS], dim=0).to(device)

            tgt_imgs, _ = next(target_gen)  # labels discarded -- unsupervised
            tgt_imgs = tgt_imgs.to(device)

            optimizer.zero_grad()
            src_feats = backbone(src_imgs)
            tgt_feats = backbone(tgt_imgs)

            logits = head(src_feats)
            cls_loss = F.cross_entropy(logits, src_labels)
            m_loss = mmd_loss(src_feats, tgt_feats)
            loss = cls_loss + lambda_mmd * m_loss

            loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_mmd_loss += m_loss.item()

        val_f1s = []
        for dom in SOURCE_DOMAINS:
            _, f1 = evaluate(backbone, head, val_loaders[dom], device)
            val_f1s.append(f1)
        mean_val_f1 = float(np.mean(val_f1s))

        print(f"Epoch {epoch+1}: cls_loss={epoch_cls_loss/steps_per_epoch:.4f}  "
              f"mmd_loss={epoch_mmd_loss/steps_per_epoch:.4f}  "
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

    results = {"per_source_val": {}, "target": {}, "lambda_mmd": lambda_mmd}
    for dom in SOURCE_DOMAINS:
        acc, f1 = evaluate(backbone, head, val_loaders[dom], device)
        results["per_source_val"][dom] = {"acc": acc, "macro_f1": f1}
    tgt_acc, tgt_f1 = evaluate(backbone, head, target_eval_loader, device)
    results["target"] = {"acc": tgt_acc, "macro_f1": tgt_f1}
    results["best_mean_source_val_f1"] = best_mean_val_f1

    print(json.dumps(results, indent=2))
    with open(os.path.join(results_dir, f"{tag}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task2/results"
    CHECKPOINT_DIR = "/content/drive/MyDrive/atml_pa1_task2/checkpoints"
    main(RESULTS_DIR, CHECKPOINT_DIR)
