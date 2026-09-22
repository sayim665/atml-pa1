"""
ResNet-18 backbone (ImageNet1K_V1 weights) with the spec's BatchNorm-freezing policy:
running mean/var stay fixed at pretrained ImageNet values for every method in Tasks 2 & 3;
only BatchNorm's own scale/bias (gamma/beta) remain trainable, same as every other layer.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_backbone(device):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feat_dim = model.fc.in_features  # 512
    model.fc = nn.Identity()  # expose the pooled feature; classifier head lives separately
    model.to(device)
    return model, feat_dim


def freeze_batchnorm_running_stats(model):
    """Call this EVERY time after model.train() -- that's the spec's required pattern:
    the whole model is in train() mode (so BN's gamma/beta get gradients), but each BN
    module is individually flipped back to eval() so its running_mean/running_var don't update.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
