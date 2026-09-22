"""
Source-only ERM baseline for Task 2 / Task 3.
Trains ResNet-18 + linear head on domain-balanced batches from Photo, Art Painting, Cartoon.
Checkpoint saved here is reused UNCHANGED as:
  - the Source-only row in Task 2's comparison table
  - the ERM baseline in Task 3 (load, do not retrain)
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task2", "models"))
from pacs_protocol import (SEED, SOURCE_DOMAINS, TARGET_DOMAIN, CLASSES, N_CLASSES,
                            IMG_SIZE, RESIZE_SIZE, BATCH_SIZE_PER_SOURCE_DOMAIN,
                            MAX_EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE)
from pacs import get_pacs_datasets
from backbone import build_backbone, freeze_batchnorm_running_stats
from classifier_head import ClassifierHead


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_transforms():
    imagenet_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = T.Compose([
        T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        T.RandomCrop(IMG_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        imagenet_norm,
    ])
    eval_tf = T.Compose([
        T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        imagenet_norm,
    ])
    return train_tf, eval_tf


def domain_balanced_iterator(loaders):
    """Cycles each source domain's DataLoader in lockstep, yielding one
    (imgs, labels, domain_name) batch per domain per step -- caller concatenates."""
    iterators = {dom: iter(loader) for dom, loader in loaders.items()}
    while True:
        batch = {}
        for dom, loader in loaders.items():
            try:
                batch[dom] = next(iterators[dom])
            except StopIteration:
                iterators[dom] = iter(loader)
                batch[dom] = next(iterators[dom])
        yield batch


@torch.no_grad()
def evaluate(backbone, head, loader, device):
    backbone.eval(); head.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)
        logits = head(feats)
        all_preds.append(logits.argmax(-1).cpu())
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro")
    return float(acc), float(f1)


def main(results_dir, checkpoint_dir, device_str="cuda"):
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
    target_loader = DataLoader(datasets[TARGET_DOMAIN]["all"], batch_size=64, shuffle=False, num_workers=2)

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
        freeze_batchnorm_running_stats(backbone)  # re-apply every epoch: model.train() above re-unfreezes BN

        for _ in range(steps_per_epoch):
            batch = next(batch_gen)
            imgs = torch.cat([batch[d][0] for d in SOURCE_DOMAINS], dim=0).to(device)
            labels = torch.cat([batch[d][1] for d in SOURCE_DOMAINS], dim=0).to(device)

            optimizer.zero_grad()
            feats = backbone(imgs)
            logits = head(feats)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        # Checkpoint selection: mean macro-F1 across the three source validation splits
        val_f1s = []
        for dom in SOURCE_DOMAINS:
            _, f1 = evaluate(backbone, head, val_loaders[dom], device)
            val_f1s.append(f1)
        mean_val_f1 = float(np.mean(val_f1s))

        print(f"Epoch {epoch+1}: mean source val macro-F1 = {mean_val_f1:.4f}  (per-domain: {[round(f,4) for f in val_f1s]})")

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

    torch.save(best_state, os.path.join(checkpoint_dir, "source_only.pth"))

    # Final reporting: per-domain val acc/F1 + target acc/F1
    results = {"per_source_val": {}, "target": {}}
    for dom in SOURCE_DOMAINS:
        acc, f1 = evaluate(backbone, head, val_loaders[dom], device)
        results["per_source_val"][dom] = {"acc": acc, "macro_f1": f1}
    tgt_acc, tgt_f1 = evaluate(backbone, head, target_loader, device)
    results["target"] = {"acc": tgt_acc, "macro_f1": tgt_f1}
    results["best_mean_source_val_f1"] = best_mean_val_f1

    print(json.dumps(results, indent=2))
    with open(os.path.join(results_dir, "source_only_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task2/results"
    CHECKPOINT_DIR = "/content/drive/MyDrive/atml_pa1_task2/checkpoints"
    main(RESULTS_DIR, CHECKPOINT_DIR)
