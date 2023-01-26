import math

import torch
import torch.nn as nn

from conv1d import Conv1D


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        assert self.hidden_dim % config.num_head == 0
        self.num_head = config.num_head

        self.in_q_proj = Conv1D(self.hidden_dim, self.hidden_dim)
        self.in_k_proj = Conv1D(self.hidden_dim, self.hidden_dim)
        self.in_v_proj = Conv1D(self.hidden_dim, self.hidden_dim)
        self.out_proj = Conv1D(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def attn(self, q, k, v, attention_mask=None, return_weight=False):
        """
        :param return_weight:
        :param q: [B, num_head, L, h]
        :param k: [B, num_head, h, L]
        :param v: [B, nun_head, L, h]
        :param attention_mask: [B, num_head, L, L]
        :return: [B, num_head, L, h]
        """
        w = torch.matmul(q, k)  # [B, num_head, L, L]
        w = w / math.sqrt(v.size(-1))

        if attention_mask is not None:
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.dropout(w)

        output = torch.matmul(w, v)
        if return_weight:
            return output, w
        return output

    def merge_heads(self, x):
        """
        :param x: [B, num_head, L, h]
        :return:
        """
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, L, num_head, h]
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1), )
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        """
        :param x: [B, L, H]
        :param k: is key?
        :return:
        """

        new_x_shape = x.size()[:-1] + (self.num_head, x.size(-1) // self.num_head)
        x = x.view(*new_x_shape)
        if k:
            if x.dim() == 4:
                return x.permute(0, 2, 3, 1)  # [B, num_head, h, L]
            elif x.dim() == 5:
                return x.permute(0, 3, 4, 1, 2)  # [B, num_head, h, L, L]
        else:
            if x.dim() == 4:
                return x.permute(0, 2, 1, 3)  # [B, nun_head, L, h]
            elif x.dim() == 5:
                return x.permute(0, 3, 1, 2, 4)  # [B, num_head, L, L, h]

    def forward(self, q, k, v, attention_mask=None, return_weight=False):
        query = self.in_q_proj(q)
        key = self.in_k_proj(k)
        value = self.in_v_proj(v)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        output = self.attn(query, key, value, attention_mask, return_weight=return_weight)
        if return_weight:
            output, w = output
            w = torch.mean(w, dim=1)
        output = self.merge_heads(output)
        output = self.out_proj(output)
        output = self.dropout(output)

        if return_weight:
            return output, w
        return output


class ASTAttention(Attention):
    def __init__(self, config):
        super().__init__(config)

        self.out_proj = Conv1D(config.hidden_dim, config.hidden_dim + config.edge_dim)

    def attn(self, q, k, v, edge_embedding, edge_mask=None, return_weight=False):
        """
        :param q: [B, num_head, L, h]
        :param k: [B, num_head, h, L]
        :param v: [B, num_head, L, h]
        :param node_mask: [B, 1, 1, L]
        :param edge_mask: [B, 1, L, L]
        :param edge_embedding: [B, num_heads, L, L, h]
        :return:
        """
        w = torch.matmul(q, k)  # [B, num_head, L, L]
        w = w / math.sqrt(v.size(-1))

        if edge_mask is not None:
            w = w + edge_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.dropout(w)

        out = torch.cat([torch.matmul(w, v), torch.matmul(w.unsqueeze(-2), edge_embedding).squeeze(-2)], dim=-1)
        if return_weight:
            return out, w
        return out

    def forward(self, q, k, v, edge_embed, edge_mask=None, return_weight=False):
        query = self.in_q_proj(q)
        key = self.in_k_proj(k)
        value = self.in_v_proj(v)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        edge_embedding = self.split_heads(edge_embed)

        out = self.attn(
            query, key, value, edge_embedding, edge_mask, return_weight
        )
        if return_weight:
            out, w = out
            w = torch.mean(w, dim=1)
        out = self.merge_heads(out)
        output = self.out_proj(out)
        output = self.dropout(output)

        if return_weight:
            return output, w
        return output

