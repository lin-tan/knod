import torch.nn as nn

from conv1d import Conv1D


class MLP(nn.Module):
    def __init__(self, hidden_dim, config):
        super().__init__()
        # hidden_dim = 4 * config.hidden_dim

        self.fc = Conv1D(hidden_dim, config.hidden_dim)
        self.proj = Conv1D(config.hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        h = nn.GELU()(self.fc(x))
        output = self.dropout(self.proj(h))
        return output

