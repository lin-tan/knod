import torch
import torch.nn as nn


class Conv1D(nn.Module):
    def __init__(self, output_dim, input_dim):
        super().__init__()

        self.output_dim = output_dim

        w = torch.empty(input_dim, output_dim)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, activation=True):
        size_out = x.size()[:-1] + (self.output_dim, )
        x = torch.addmm(
            self.bias,
            x.view(-1, x.size(-1)),
            self.weight,
        ).view(*size_out)
        if activation:
            x = nn.GELU()(x)
        return x
