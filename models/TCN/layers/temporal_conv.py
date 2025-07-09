import torch
import torch.nn as nn
from .spatial_dropout import SpatialDropout1D

class TemporalConvLinear(nn.Module):
    """
    Un bloque: Conv1d + SpatialDropout + Conv1d(kernel=1)
    """
    def __init__(self, n_nodes, conv_len, n_classes, n_feat,
                 causal: bool = False, dropout: float = 0.3):
        super().__init__()
        pad = (conv_len - 1, 0) if causal else (conv_len//2, conv_len//2)
        self.pad    = nn.ConstantPad1d(pad, 0)
        self.conv1  = nn.Conv1d(n_feat,   n_nodes,    conv_len)
        self.dropout= SpatialDropout1D(dropout)
        self.fc     = nn.Conv1d(n_nodes,  n_classes, 1)

    def forward(self, x):
        x = x.transpose(1, 2)     # (B, T, F) -> (B, F, T)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x.transpose(1, 2)  # (B, n_classes, T) -> (B, T, n_classes)
