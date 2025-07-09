import torch
import torch.nn as nn
from layers.spatial_dropout import SpatialDropout1D
from layers.activations    import NormReLU, WavenetActivation

class TimeDelayNeuralNetwork(nn.Module):
    """
    TDNN: sólo convoluciones dilatadas + dropout + activación + conv1x1 final.
    """
    def __init__(self,
                 n_nodes: list[int],
                 conv_len: int,
                 n_classes: int,
                 n_feat: int,
                 causal: bool = False,
                 activation: str = 'sigmoid',
                 dropout: float = 0.3):
        super().__init__()
        self.causal = causal
        self.layers = nn.ModuleList()
        in_ch = n_feat

        for i, out_ch in enumerate(n_nodes):
            rate = i + 1
            pad = (rate * (conv_len - 1), 0) if causal else ((conv_len // 2) * rate, (conv_len // 2) * rate)
            seq = []
            if causal:
                seq.append(nn.ConstantPad1d(pad, 0))
                seq.append(nn.Conv1d(in_ch, out_ch, conv_len, dilation=rate, padding=0))
            else:
                seq.append(nn.Conv1d(in_ch, out_ch, conv_len, dilation=rate, padding=(conv_len // 2) * rate))
            self.layers.append(nn.Sequential(*seq))
            in_ch = out_ch

        self.final   = nn.Conv1d(in_ch, n_classes, kernel_size=1)
        self.dropout = SpatialDropout1D(dropout)

        if activation == 'norm_relu':
            self.act = NormReLU()
        elif activation == 'wavenet':
            self.act = WavenetActivation()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = getattr(nn, activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_feat) -> (B, n_feat, T)
        x = x.transpose(1, 2)
        for conv in self.layers:
            x = conv(x)
            x = self.dropout(x)
            x = self.act(x)
        x = self.final(x)
        return x.transpose(1, 2)
