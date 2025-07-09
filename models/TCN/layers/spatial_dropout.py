import torch
import torch.nn as nn

class SpatialDropout1D(nn.Module):
    """
    Drop entire channels (features) para inputs 1D, como Keras SpatialDropout1D.
    Input: (batch, channels, length)
    """
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.dropout2d = nn.Dropout2d(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(3)           # (B, C, L) -> (B, C, L, 1)
        x = self.dropout2d(x)
        return x.squeeze(3)          # de vuelta a (B, C, L)