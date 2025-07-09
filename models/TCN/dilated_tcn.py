import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.spatial_dropout import SpatialDropout1D
from layers.activations    import WavenetActivation

class DilatedTCN(nn.Module):
    """
    WaveNet-like Dilated TCN con conexiones residuales y skip.
    """
    def __init__(self,
                 num_feat: int,
                 num_classes: int,
                 nb_filters: int,
                 dilation_depth: int,
                 nb_stacks: int,
                 use_skip_connections: bool = True,
                 causal: bool = False,
                 dropout: float = 0.3):
        super().__init__()
        self.causal    = causal
        self.use_skip  = use_skip_connections

        # Capa inicial
        self.initial = nn.Conv1d(num_feat, nb_filters, kernel_size=3, padding=1)

        # Bloques dilatados
        self.stacks     = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for _ in range(nb_stacks):
            for i in range(dilation_depth + 1):
                rate = 2 ** i
                # convolución dilatada
                self.stacks.append(nn.Conv1d(
                    nb_filters, nb_filters,
                    kernel_size=3,
                    dilation=rate,
                    padding=rate if not causal else 0
                ))
                # convolución 1×1 para skip
                self.skip_convs.append(nn.Conv1d(nb_filters, nb_filters, kernel_size=1))

        self.dropout  = SpatialDropout1D(dropout)
        self.act      = WavenetActivation()
        self.res_conv = nn.Conv1d(nb_filters, nb_filters, kernel_size=1)

        # Post-procesamiento
        self.post_conv1 = nn.Conv1d(nb_filters, nb_filters, kernel_size=1)
        self.post_conv2 = nn.Conv1d(nb_filters, num_classes,  kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, num_feat) -> (B, num_feat, T)
        x = x.transpose(1, 2)
        x = self.initial(x)

        skip_connections = []
        idx = 0

        for conv in self.stacks:
            orig = x
            # padding causal si toca
            if self.causal:
                pad = (conv.dilation[0], 0)
                x = F.pad(x, pad)
            x = conv(x)
            x = self.dropout(x)
            x = self.act(x)
            # residual
            res = self.res_conv(x)
            x = orig + res
            # skip
            skip = self.skip_convs[idx](x)
            skip_connections.append(skip)
            idx += 1

        # suma de skips
        if self.use_skip:
            x = sum(skip_connections)

        x = F.relu(x)
        x = self.post_conv1(x)
        x = F.relu(x)
        x = self.post_conv2(x)

        return x.transpose(1, 2)  # (B, T, num_classes)
