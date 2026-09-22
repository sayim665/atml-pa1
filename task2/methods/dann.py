"""
DANN -- adversarial domain alignment via a domain discriminator + gradient-reversal layer (GRL).
Discriminator sees the same 512-dim feature as DAN's MMD term. Only source examples contribute
to the classification loss; BOTH source and target contribute to the domain-classification loss.
alpha(p) = 2/(1+exp(-10p)) - 1, p = training progress in [0, 1] over the WHOLE training run.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from pacs_protocol import (SEED, SOURCE_DOMAINS, TARGET_DOMAIN, N_CLASSES,
                            BATCH_SIZE_PER_SOURCE_DOMAIN, BATCH_SIZE_TARGET,
                            MAX_EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE)
from pacs import get_pacs_datasets
from backbone import build_backbone, freeze_batchnorm_running_stats
from classifier_head import ClassifierHead
from domain_discriminator import DomainDiscriminator, GradientReversalLayer, grl_alpha_schedule

sys.path.insert(0, os.path.dirname(__file__))
from source_only import set_seed, get_transforms, domain_balanced_iterator, evaluate
from dan import target_cycle_iterator

DOMAIN_LOSS_WEIGHT = 1.0  # "unit weight" per spec


def main(results_dir, checkpoint_dir, device_str="cuda", tag="dann"):
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
    target_train_loader = DataLoader(datasets[TARGET_DOMAIN]["all"], batch_size=BATCH_SIZE_TARGET,
                                      shuffle=True, num_workers=2, drop_last=True)
    target_eval_loader = DataLoader(datasets[TARGET_DOMAIN]["all"], batch_size=64,
                                     shuffle=False, num_workers=2)

    backbone, feat_dim = build_backbone(device)
    head = ClassifierHead(feat_dim, N_CLASSES).to(device)
    grl = GradientReversalLayer().to(device)
    discriminator = DomainDiscriminator(feat_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()) + list(discriminator.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = min(len(dl) for dl in train_loaders.values())
    total_steps = steps_per_epoch * MAX_EPOCHS  # denominator for the alpha(p) schedule
    source_batch_gen = domain_balanced_iterator(train_loaders)
    target_gen = target_cycle_iterator(target_train_loader)

    best_mean_val_f1, best_state, patience = -1, None, 0
    global_step = 0

    for epoch in range(MAX_EPOCHS):
        backbone.train(); head.train(); discriminator.train()
        freeze_batchnorm_running_stats(backbone)

        epoch_cls_loss, epoch_domain_loss = 0.0, 0.0

        for _ in range(steps_per_epoch):
            p = global_step / max(1, total_steps)
            grl.alpha = grl_alpha_schedule(p)

            src_batch = next(source_batch_gen)
            src_imgs = torch.cat([src_batch[d][0] for d in SOURCE_DOMAINS], dim=0).to(device)
            src_labels = torch.cat([src_batch[d][1] for d in SOURCE_DOMAINS], dim=0).to(device)
            tgt_imgs, _ = next(target_gen)
            tgt_imgs = tgt_imgs.to(device)

            optimizer.zero_grad()

            src_feats = backbone(src_imgs)
            tgt_feats = backbone(tgt_imgs)

            # Classification: SOURCE ONLY
            logits = head(src_feats)
            cls_loss = F.cross_entropy(logits, src_labels)

            # Domain classification: BOTH source and target, through the GRL
            all_feats = torch.cat([src_feats, tgt_feats], dim=0)
            domain_labels = torch.cat([
                torch.zeros(src_feats.size(0), dtype=torch.long),  # 0 = source
                torch.ones(tgt_feats.size(0), dtype=torch.long),   # 1 = target
            ]).to(device)
            reversed_feats = grl(all_feats)
            domain_logits = discriminator(reversed_feats)
            domain_loss = F.cross_entropy(domain_logits, domain_labels)

            loss = cls_loss + DOMAIN_LOSS_WEIGHT * domain_loss
            loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_domain_loss += domain_loss.item()
            global_step += 1

        val_f1s = []
        for dom in SOURCE_DOMAINS:
            _, f1 = evaluate(backbone, head, val_loaders[dom], device)
            val_f1s.append(f1)
        mean_val_f1 = float(np.mean(val_f1s))

        print(f"Epoch {epoch+1}: cls_loss={epoch_cls_loss/steps_per_epoch:.4f}  "
              f"domain_loss={epoch_domain_loss/steps_per_epoch:.4f}  "
              f"alpha={grl.alpha:.3f}  mean source val macro-F1={mean_val_f1:.4f}")

        if mean_val_f1 > best_mean_val_f1:
            best_mean_val_f1 = mean_val_f1
            best_state = {
                "backbone": {k: v.clone() for k, v in backbone.state_dict().items()},
                "head": {k: v.clone() for k, v in head.state_dict().items()},
                "discriminator": {k: v.clone() for k, v in discriminator.state_dict().items()},
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

    results = {"per_source_val": {}, "target": {}}
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
