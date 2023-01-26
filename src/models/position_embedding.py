import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.config = config
        self.position_embedding = nn.Embedding(config.max_node_num + 2, config.hidden_dim)

    def forward(self, x):
        output = torch.arange(0, x.size(1)).unsqueeze(0).repeat(x.size(0), 1).type_as(x)
        output = self.position_embedding(output)
        return output
