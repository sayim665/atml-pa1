"""
Vanilla closed-set baseline: ten-class CIFAR ResNet-18, cross-entropy, SGD, cosine decay,
seed 6304. This checkpoint's frozen logits/features feed ALL four post-hoc scores (MSP, MLS,
Energy, Mahalanobis) in scores/, and PROSER initializes from it (per spec Step 4).
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
BATCH_SIZE = 128
EPOCHS = 100
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _ = model(imgs)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train(data_root, results_dir, checkpoint_dir, device_str="cuda",
          gcsc=False, tag=None, epochs=EPOCHS):
    set_seed(SEED)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    tag = tag or ("gcsc" if gcsc else "vanilla")

    train_ds, val_ds, test_ds = build_cifar10_datasets(data_root, gcsc=gcsc, seed=SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = resnet18_cifar(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc, best_state = -1, None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{tag}.pth"))

    test_acc = evaluate(model, test_loader, device)
    results = {"best_val_acc": best_val_acc, "test_acc": test_acc, "gcsc": gcsc}
    print(json.dumps(results, indent=2))
    with open(os.path.join(results_dir, f"{tag}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, model


if __name__ == "__main__":
    DATA_ROOT = "./data"
    RESULTS_DIR = "/content/drive/MyDrive/atml_pa1_task4/results"
    CHECKPOINT_DIR = "/content/drive/MyDrive/atml_pa1_task4/checkpoints"
    train(DATA_ROOT, RESULTS_DIR, CHECKPOINT_DIR)
