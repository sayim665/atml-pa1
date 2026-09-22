import torch.nn as nn

class ClassifierHead(nn.Module):
    """Plain linear classifier on top of the backbone's pooled feature."""
    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, feat):
        return self.fc(feat)
