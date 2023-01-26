import torch
import torch.nn as nn
from conv1d import Conv1D


class FatherPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.in_q_proj = Conv1D(self.hidden_dim, self.hidden_dim)
        self.in_k_proj = Conv1D(self.hidden_dim, self.hidden_dim)

    def forward(self, x, attention_mask=None):
        query = self.in_q_proj(x)
        key = self.in_k_proj(x).permute(0, 2, 1)
        w = torch.matmul(query, key)

        if attention_mask is not None:
            w = w + attention_mask
        w = nn.Softmax(dim=-1)(w)
        return w


class EdgeGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc = Conv1D(config.hidden_dim, config.hidden_dim)
        self.generator = Conv1D(len(config.edge_vocabulary), config.hidden_dim)

    def forward(self, x):
        out = self.generator(
            self.fc(x), activation=False
        )
        return nn.Softmax(dim=-1)(out)


class NodeGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_gen = Conv1D(1, config.hidden_dim)
        self.fc = Conv1D(config.hidden_dim, config.hidden_dim)
        self.generator = Conv1D(len(config.node_vocabulary), config.hidden_dim)

    def forward(self, x, encoder_attn_w, nodes):
        p_gen = nn.Sigmoid()(self.fc_gen(x, activation=False))
        out = self.generator(
            self.fc(x), activation=False
        )
        out = nn.Softmax(dim=-1)(out) * p_gen
        out = out.scatter_add_(2, nodes.unsqueeze(1).repeat(1, out.size(1), 1), encoder_attn_w * (1 - p_gen))
        return out

