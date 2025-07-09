import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSE_Bidirectional_LSTM_Triplet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim, dropout=0.5, bidirectional=True):
        super(LSE_Bidirectional_LSTM_Triplet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        # Fully connected layer to embedding space
        fc_input = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_input, embedding_dim)

    def forward(self, x, lengths):
        # Pack sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        # LSTM
        _, (hidden, _) = self.lstm(packed)
        # Concatenate last hidden states
        if self.bidirectional:
            hidden_last = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden_last = hidden[-1]
        emb = self.dropout(hidden_last)
        emb = self.fc(emb)
        return emb