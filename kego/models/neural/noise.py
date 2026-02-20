import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """Adds Gaussian noise during training only. Regularization for synthetic data."""

    def __init__(self, std=0.01):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x
