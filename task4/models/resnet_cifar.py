"""
CIFAR-appropriate ResNet-18: replace the ImageNet 7x7 stride-2 first conv with a 3x3 stride-1
conv, and remove the initial max-pool, per spec. Operates on native 32x32 images (no resize).
Built from scratch (not torchvision's ImageNet ResNet-18) since torchvision's stem assumes
224x224 inputs; this is the standard, widely-used CIFAR ResNet-18 architecture pattern.
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        # CIFAR stem: 3x3 stride-1 conv, NO max-pool (replaces ImageNet's 7x7 stride-2 + maxpool)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.feat_dim = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.flatten(1)  # penultimate feature, f(x), shape (B, 512)

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.fc(feat)
        return logits, feat  # return BOTH -- OSR scores need penultimate features too (Mahalanobis)

    def forward_up_to_layer2(self, x):
        """For PROSER's manifold mixup, which needs the network truncated after layer2."""
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    def forward_from_layer3(self, h):
        """Continues the forward pass from a layer2 output (post-mixup) through to logits+feat."""
        out = self.layer3(h)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out.flatten(1)
        logits = self.fc(feat)
        return logits, feat


def resnet18_cifar(num_classes=10):
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
