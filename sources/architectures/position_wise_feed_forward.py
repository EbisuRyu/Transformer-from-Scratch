import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.fc_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x