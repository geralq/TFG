import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSE_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, bidirectional=True):
        super(LSE_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, (hidden, _) = self.lstm(packed_input)

        if self.bidirectional:
            hidden_last = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden_last = hidden[-1]

        hidden_last = self.dropout(hidden_last)
        return hidden_last