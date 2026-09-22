"""
Domain discriminator for DANN / CDAN, plus the gradient-reversal layer (GRL) both use.
Architecture fixed by spec: 256-unit hidden layer, ReLU, dropout 0.5, 2-class output.
"""
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.0  # set externally each step via the schedule below

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


def grl_alpha_schedule(p: float) -> float:
    """p in [0, 1] = training progress. Standard DANN schedule from the spec."""
    return 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))).item() - 1.0


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)
