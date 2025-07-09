import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSE_Bidirectional_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5, bidirectional=True):
        super(LSE_Bidirectional_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)
    
    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed_input)
        
        if self.bidirectional:
            hidden_last = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden_last = hidden[-1]
        
        hidden_last = self.dropout(hidden_last)
        logits = self.fc(hidden_last)
        return logits