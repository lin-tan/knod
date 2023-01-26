import torch
import torch.nn as nn

from attention import ASTAttention, Attention
from mlp import MLP
from conv1d import Conv1D


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_dim

        self.encoder_self_seq_attn = Attention(config)
        self.encoder_self_ast_attn = ASTAttention(config)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(4 * hidden_dim, config)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, nodes_h, edge_embedding, node_mask=None, edge_mask=None):
        x = nodes_h

        seq_h = self.encoder_self_seq_attn(
            x, x, x, node_mask
        )
        ast_h = self.encoder_self_ast_attn(
            seq_h, seq_h, seq_h,
            edge_embedding,
            edge_mask
        )

        n = self.ln1(x + seq_h + ast_h)
        h = self.mlp(n)
        h = self.ln2(n + h)

        return h


class FatherDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_dim

        self.decoder_self_seq_attn = Attention(config)
        self.decoder_self_ast_attn = ASTAttention(config)
        self.decoder_encoder_attn = Attention(config)
        self.decoder_rem_attn = Attention(config)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(4 * hidden_dim, config)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, encoder_out, rem_tags, nodes_h, edge_embedding, node_mask=None, edge_mask=None, encoder_mask=None):
        x = nodes_h
        self_seq_h = self.decoder_self_seq_attn(
            x, x, x, node_mask
        )
        self_ast_h = self.decoder_self_ast_attn(
            self_seq_h, self_seq_h, self_seq_h,
            edge_embedding,
            edge_mask,
        )

        n = self.ln1(x + self_seq_h + self_ast_h)

        encoder_attn_h = self.decoder_encoder_attn(
            n, encoder_out, encoder_out,
            encoder_mask,
        )
        rem_mask = (rem_tags.unsqueeze(1).unsqueeze(2) == 2).type_as(n)
        rem_mask = (1.0 - rem_mask) * -1.e16
        rem_attn_h = self.decoder_rem_attn(
            n, encoder_out, encoder_out,
            rem_mask
        )

        n = self.ln2(n + encoder_attn_h + rem_attn_h)
        h = self.mlp(n)

        h = self.ln3(n + h)
        return h


class EdgeDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_dim

        self.decoder_self_seq_attn = Attention(config)
        self.decoder_self_ast_attn = ASTAttention(config)
        self.decoder_encoder_attn = Attention(config)
        self.decoder_rem_attn = Attention(config)

        self.father_proj = Conv1D(config.hidden_dim, 2 * config.hidden_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(4 * hidden_dim, config)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, encoder_out, rem_tags, nodes_h, next_father_indices,
                edge_embedding, node_mask=None, edge_mask=None, encoder_mask=None):
        x = nodes_h
        next_father_h = torch.gather(x, 1, next_father_indices.unsqueeze(-1).repeat(1, 1, x.size(-1)))
        x = self.father_proj(torch.cat([
            next_father_h, x
        ], dim=-1))

        self_seq_h = self.decoder_self_seq_attn(
            x, x, x, node_mask
        )
        self_ast_h = self.decoder_self_ast_attn(
            self_seq_h, self_seq_h, self_seq_h,
            edge_embedding,
            edge_mask,
        )

        n = self.ln1(x + self_seq_h + self_ast_h)

        encoder_attn_h = self.decoder_encoder_attn(
            n, encoder_out, encoder_out,
            encoder_mask,
        )
        rem_mask = (rem_tags.unsqueeze(1).unsqueeze(2) == 2).type_as(n)
        rem_mask = (1.0 - rem_mask) * -1.e16
        rem_attn_h = self.decoder_rem_attn(
            n, encoder_out, encoder_out,
            rem_mask
        )

        n = self.ln2(n + encoder_attn_h + rem_attn_h)
        h = self.mlp(n)
        h = self.ln3(n + h)
        return h


class NodeDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_dim

        self.decoder_self_seq_attn = Attention(config)
        self.decoder_self_ast_attn = ASTAttention(config)
        self.decoder_encoder_attn = Attention(config)
        self.decoder_rem_attn = Attention(config)

        self.father_edge_proj = Conv1D(config.hidden_dim, 2 * config.hidden_dim + config.edge_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(4 * hidden_dim, config)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, encoder_out, rem_tags, nodes_h, next_father_indices, next_edge_embedding,
                edge_embedding, node_mask=None, edge_mask=None, encoder_mask=None, return_weight=False):
        x = nodes_h
        next_father_h = torch.gather(x, 1, next_father_indices.unsqueeze(-1).repeat(1, 1, x.size(-1)))
        x = self.father_edge_proj(torch.cat([
            next_father_h, next_edge_embedding, x
        ], dim=-1))

        self_seq_h = self.decoder_self_seq_attn(
            x, x, x, node_mask, return_weight
        )
        if return_weight:
            self_seq_h, self_seq_w = self_seq_h
        self_ast_h = self.decoder_self_ast_attn(
            self_seq_h, self_seq_h, self_seq_h,
            edge_embedding, edge_mask, return_weight
        )
        if return_weight:
            self_ast_h, self_ast_w = self_ast_h

        n = self.ln1(x + self_seq_h + self_ast_h)

        encoder_attn_h = self.decoder_encoder_attn(
            n, encoder_out, encoder_out,
            encoder_mask, return_weight
        )
        rem_mask = (rem_tags.unsqueeze(1).unsqueeze(2) == 2).type_as(n)
        rem_mask = (1.0 - rem_mask) * -1.e16
        rem_attn_h = self.decoder_rem_attn(
            n, encoder_out, encoder_out,
            rem_mask, return_weight
        )
        if return_weight:
            encoder_attn_h, encoder_attn_w = encoder_attn_h
            rem_attn_h, rem_attn_w = rem_attn_h

        n = self.ln2(n + encoder_attn_h + rem_attn_h)
        h = self.mlp(n)
        h = self.ln3(n + h)

        if return_weight:
            return h, self_seq_w, self_ast_w, encoder_attn_w, rem_attn_w
        return h
