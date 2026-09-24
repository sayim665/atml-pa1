"""
PROSER: classifier placeholders + data placeholders, fine-tuned from the Vanilla checkpoint.
- 5 randomly-initialized dummy classifiers appended to the 10-class head.
- Classifier placeholders: for a normal example, the true class should stay top-1, but among
  the REMAINING logits (true class excluded), one of the dummy classifiers should win.
- Data placeholders: manifold mixup of two DIFFERENT-class examples' features (after layer2,
  before layer3), trained toward the dummy classifiers rather than either source class.
- Each mini-batch's first half -> classifier-placeholder loss; second half -> data-placeholder loss.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from resnet_cifar import resnet18_cifar
from cifar10 import build_cifar10_datasets

SEED = 6304
N_KNOWN = 10
N_DUMMY = 5
BETA_CLS_PLACEHOLDER = 1.0
GAMMA_DATA_PLACEHOLDER = 0.1
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class ProserHead(nn.Module):
    """10 known-class outputs + N_DUMMY placeholder outputs, all from one linear layer."""
    def __init__(self, feat_dim, n_known=N_KNOWN, n_dummy=N_DUMMY):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_known + n_dummy)
        self.n_known = n_known
        self.n_dummy = n_dummy

    def forward(self, feat):
        return self.fc(feat)  # (B, n_known + n_dummy)


def classifier_placeholder_loss(logits, labels, n_known):
    """For each example: true class should be argmax overall (standard CE on the FULL logit
    vector, true label). THEN, among logits with the true class masked out, one dummy should
    be the strongest remaining response -- encouraged via a second CE term whose target is
    "any dummy index", implemented as CE against the max-dummy trick: push the max-dummy logit
    above every OTHER non-true-class logit using a max-margin-style CE over (non-true logits)."""
    B, n_total = logits.shape
    n_dummy = n_total - n_known

    # (1) Standard classification loss: true class should win overall.
    ce_main = F.cross_entropy(logits, labels)

    # (2) Placeholder loss: mask out the true class, then treat "the best dummy" as the
    # desired argmax among the remaining (known-but-wrong + dummy) logits. We implement this
    # via cross-entropy against a synthetic target built by taking, for each example, the
    # highest-scoring dummy index as the positive class in a softmax over all NON-true logits.
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)  # mask out the true-class column

    masked_logits = logits.masked_fill(~mask, float("-inf"))
    dummy_logits = masked_logits[:, n_known:]  # (B, n_dummy), true class never lives here anyway
    best_dummy_local = dummy_logits.argmax(dim=1)  # which dummy is currently strongest
    best_dummy_global = best_dummy_local + n_known  # index into the full (masked) logits

    ce_placeholder = F.cross_entropy(masked_logits, best_dummy_global)

    return ce_main + BETA_CLS_PLACEHOLDER * ce_placeholder


def manifold_mixup_pair(feat_layer2, labels, alpha_beta=2.0):
    """Pairs each example with a DIFFERENT-class example in the same batch, mixes their
    post-layer2 features with lambda ~ Beta(2,2). Returns (mixed_feat, valid_mask) -- valid_mask
    excludes pairs that accidentally share a label (can't avoid entirely without an extra pass,
    so we filter afterward)."""
    B = feat_layer2.size(0)
    perm = torch.randperm(B, device=feat_layer2.device)
    labels_perm = labels[perm]
    valid = labels != labels_perm  # spec: y_i != y_j required

    lam = float(np.random.beta(alpha_beta, alpha_beta))
    mixed = lam * feat_layer2 + (1 - lam) * feat_layer2[perm]
    return mixed, valid


def data_placeholder_loss(model, head, feat_layer2, labels, n_known, device):
    mixed_feat, valid_mask = manifold_mixup_pair(feat_layer2, labels)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    mixed_feat = mixed_feat[valid_mask]
    # Continue the mixed feature through layer3 -> layer4 -> avgpool to get the final
    # penultimate feature, then apply ProserHead (NOT model.forward_from_layer3, since that
    # uses the backbone's own bypassed fc rather than the ProserHead with its dummy classes).
    out = model.layer3(mixed_feat)
    out = model.layer4(out)
    out = model.avgpool(out)
    final_feat = out.flatten(1)
    proser_logits = head(final_feat)  # (n_valid, n_known + n_dummy)

    # Target: any dummy class. Train toward the dummy classifiers as a group, via CE against
    # the currently-strongest dummy (same trick as the classifier-placeholder loss, but here
    # there is no "true class" to exclude -- ALL dummy logits are eligible targets).
    dummy_logits = proser_logits[:, n_known:]
    best_dummy_local = dummy_logits.argmax(dim=1)
    best_dummy_global = best_dummy_local + n_known
    loss = F.cross_entropy(proser_logits, best_dummy_global)
    return loss


def train(data_root, vanilla_checkpoint_path, results_dir, checkpoint_dir, device_str="cuda"):
    set_seed(SEED)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_ds, val_ds, test_ds = build_cifar10_datasets(data_root, gcsc=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model = resnet18_cifar(num_classes=N_KNOWN).to(device)
    model.load_state_dict(torch.load(vanilla_checkpoint_path, map_location=device))

    head = ProserHead(model.feat_dim).to(device)
    # Initialize the head's KNOWN-class rows from Vanilla's trained fc weights; dummy rows random.
    with torch.no_grad():
        head.fc.weight[:N_KNOWN] = model.fc.weight.clone()
        head.fc.bias[:N_KNOWN] = model.fc.bias.clone()
    model.fc = nn.Identity()  # bypass Vanilla's own fc; ProserHead takes over entirely

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(head.parameters()),
        lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc, best_state = -1, None

    for epoch in range(EPOCHS):
        model.train(); head.train()
        epoch_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            half = imgs.size(0) // 2
            imgs_a, labels_a = imgs[:half], labels[:half]
            imgs_b, labels_b = imgs[half:], labels[half:]

            optimizer.zero_grad()

            # --- First half: classifier-placeholder loss ---
            feat_a = model.forward_features(imgs_a)
            logits_a = head(feat_a)
            loss_cls_ph = classifier_placeholder_loss(logits_a, labels_a, N_KNOWN)

            # --- Second half: data-placeholder loss (manifold mixup after layer2) ---
            feat_l2_b = model.forward_up_to_layer2(imgs_b)
            loss_data_ph = data_placeholder_loss(model, head, feat_l2_b, labels_b, N_KNOWN, device)

            loss = loss_cls_ph + GAMMA_DATA_PLACEHOLDER * loss_data_ph
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation: known-class accuracy using ONLY the known-class logits (per spec).
        model.eval(); head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = model.forward_features(imgs)
                logits = head(feat)
                known_logits = logits[:, :N_KNOWN]
                correct += (known_logits.argmax(-1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1}/{EPOCHS}: loss={epoch_loss/len(train_loader):.4f}  val_acc(known)={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model": {k: v.clone() for k, v in model.state_dict().items()},
                "head": {k: v.clone() for k, v in head.state_dict().items()},
            }

    model.load_state_dict(best_state["model"])
    head.load_state_dict(best_state["head"])
    torch.save(best_state, os.path.join(checkpoint_dir, "proser.pth"))

    results = {"best_val_acc_known_only": best_val_acc}
    print(json.dumps(results, indent=2))
    with open(os.path.join(results_dir, "proser_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, model, head


if __name__ == "__main__":
    DATA_ROOT = "./data"
    VANILLA_CKPT = "/content/drive/MyDrive/atml_pa1_task4/checkpoints/vanilla.pth"
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task4/results"
    CHECKPOINT_DIR = "/content/drive/MyDrive/atml_pa1_task4/checkpoints"
    train(DATA_ROOT, VANILLA_CKPT, RESULTS_DIR, CHECKPOINT_DIR)
