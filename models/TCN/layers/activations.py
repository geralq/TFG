import torch
import torch.nn as nn
import torch.nn.functional as F

class NormReLU(nn.Module):
    """ReLU normalizado: y = ReLU(x) / (max_c + eps)"""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        max_c = x.max(dim=1, keepdim=True)[0]
        return x / (max_c + self.eps)

class WavenetActivation(nn.Module):
    """f(x) = tanh(x) * sigmoid(x)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) * torch.sigmoid(x)
