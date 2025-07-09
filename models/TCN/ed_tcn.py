import torch
import torch.nn as nn
from .layers.spatial_dropout import SpatialDropout1D
from .layers.activations    import NormReLU, WavenetActivation

class ED_TCN(nn.Module):
    """
    Encoder-Decoder TCN
    """
    def __init__(self,
                 n_nodes: list[int],
                 conv_len: int,
                 n_classes: int,
                 n_feat: int,
                 causal: bool = False,
                 activation: str = 'norm_relu',
                 dropout: float = 0.2):
        super().__init__()
        self.causal = causal

        # Pools & dropout
        self.pool     = nn.MaxPool1d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout  = SpatialDropout1D(dropout)

        # Encoders
        self.encoder = nn.ModuleList()
        in_ch = n_feat
        for out_ch in n_nodes:
            pad = (conv_len - 1, 0) if causal else (conv_len // 2, conv_len // 2)
            layers = []
            if causal:
                layers.append(nn.ConstantPad1d(pad, 0))
                layers.append(nn.Conv1d(in_ch, out_ch, conv_len, padding=0))
            else:
                layers.append(nn.Conv1d(in_ch, out_ch, conv_len, padding=conv_len // 2))
            self.encoder.append(nn.Sequential(*layers))
            in_ch = out_ch

        # Decoders (reverse)
        self.decoder = nn.ModuleList()
        for out_ch in reversed(n_nodes):
            pad = (conv_len - 1, 0) if causal else (conv_len // 2, conv_len // 2)
            layers = []
            if causal:
                layers.append(nn.ConstantPad1d(pad, 0))
                layers.append(nn.Conv1d(in_ch, out_ch, conv_len, padding=0))
            else:
                layers.append(nn.Conv1d(in_ch, out_ch, conv_len, padding=conv_len // 2))
            self.decoder.append(nn.Sequential(*layers))
            in_ch = out_ch

        # Final 1×1 conv

        print('DEBUG: in_ch =', in_ch, '| n_classes =', n_classes)

        self.final = nn.Conv1d(in_ch, n_classes, kernel_size=1)

        # Activation selector
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
        # Encoder
        for conv in self.encoder:
            x = conv(x)
            x = self.dropout(x)
            x = self.act(x)
            x = self.pool(x)
        # Decoder
        for conv in self.decoder:
            x = self.upsample(x)
            x = conv(x)
            x = self.dropout(x)
            x = self.act(x)
        # Final 1×1
        x = self.final(x)
        # (B, n_classes, T) -> (B, T, n_classes)
        return x.transpose(1, 2)


class ED_TCN_atrous(nn.Module):
    """
    Encoder-Decoder TCN con convoluciones dilatadas (atrous)
    """
    def __init__(self,
                 n_nodes: list[int],
                 conv_len: int,
                 n_classes: int,
                 n_feat: int,
                 causal: bool = False,
                 activation: str = 'norm_relu',
                 dropout: float = 0.3):
        super().__init__()
        self.causal = causal
        self.dropout = SpatialDropout1D(dropout)

        # Encoder dilatado
        self.encoder = nn.ModuleList()
        in_ch = n_feat
        for i, out_ch in enumerate(n_nodes):
            rate = i + 1
            pad = (rate * (conv_len - 1), 0) if causal else ((conv_len // 2) * rate, (conv_len // 2) * rate)
            layers = []
            if causal:
                layers.append(nn.ConstantPad1d(pad, 0))
                layers.append(nn.Conv1d(in_ch, out_ch, conv_len, dilation=rate, padding=0))
            else:
                layers.append(nn.Conv1d(in_ch, out_ch, conv_len,
                                        dilation=rate,
                                        padding=(conv_len // 2) * rate))
            self.encoder.append(nn.Sequential(*layers))
            in_ch = out_ch

        # Decoder (dilations invertidos)
        self.decoder = nn.ModuleList()
        L = len(n_nodes)
        for i, out_ch in enumerate(reversed(n_nodes)):
            rate = L - i
            pad = (rate * (conv_len - 1), 0) if causal else ((conv_len // 2) * rate, (conv_len // 2) * rate)
            layers = []
            if causal:
                layers.append(nn.ConstantPad1d(pad, 0))
                layers.append(nn.Conv1d(in_ch, out_ch, conv_len, dilation=rate, padding=0))
            else:
                layers.append(nn.Conv1d(in_ch, out_ch, conv_len,
                                        dilation=rate,
                                        padding=(conv_len // 2) * rate))
            self.decoder.append(nn.Sequential(*layers))
            in_ch = out_ch

        # Final 1×1
        self.final = nn.Conv1d(in_ch, n_classes, kernel_size=1)

        # Activación
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
        # Encoder
        for conv in self.encoder:
            x = conv(x)
            x = self.dropout(x)
            x = self.act(x)
        # Decoder
        for conv in self.decoder:
            x = conv(x)
            x = self.dropout(x)
            x = self.act(x)
        # Final
        x = self.final(x)
        return x.transpose(1, 2)
