import torch
import torch.nn as nn

from block import EncoderBlock, FatherDecoderBlock, EdgeDecoderBlock, NodeDecoderBlock
from conv1d import Conv1D
from grammar_checker import GrammarChecker
from predictor import FatherPredictor, EdgeGenerator, NodeGenerator
from position_embedding import PositionEmbedding


class Config:
    def __init__(self, node_vocabulary, edge_vocabulary, hidden_dim, edge_dim, num_head,
                 num_encoder_layer, num_father_layer, num_edge_layer, num_node_layer, dropout):
        self.node_vocabulary = node_vocabulary
        self.edge_vocabulary = edge_vocabulary
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.max_node_num = 512
        self.num_head = num_head
        self.num_encoder_layer = num_encoder_layer
        self.num_father_layer = num_father_layer
        self.num_edge_layer = num_edge_layer
        self.num_node_layer = num_node_layer
        self.dropout = dropout

    def to_dict(self):
        return {
            'node_vocabulary': len(self.node_vocabulary),
            'edge_vocabulary': len(self.edge_vocabulary),
            'hidden_dim': self.hidden_dim,
            'edge_dim': self.edge_dim,
            'num_head': self.num_head,
            'num_encoder_layer': self.num_encoder_layer,
            'num_father_layer': self.num_father_layer,
            'num_edge_layer': self.num_edge_layer,
            'num_node_layer': self.num_node_layer
        }


class GraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.node_vocabulary = config.node_vocabulary
        self.edge_vocabulary = config.edge_vocabulary

        self.node_embedding = nn.Embedding(len(self.node_vocabulary), config.hidden_dim, padding_idx=0)
        self.edge_embedding = nn.Embedding(len(self.edge_vocabulary), config.edge_dim, padding_idx=0)
        self.position_embedding = PositionEmbedding(self.config)

        self.encoder = GraphTransformerEncoder(
            self.node_vocabulary, self.edge_vocabulary,
            self.node_embedding, self.edge_embedding, self.position_embedding, config
        )
        self.decoder = GraphTransformerDecoder(
            self.node_vocabulary, self.edge_vocabulary,
            self.node_embedding, self.edge_embedding, self.position_embedding, config
        )

        self.apply(init_weights)

    def get_attn_weights(self, inputs):
        encoder_out = self.encode(inputs)
        _, _, _, _, _, self_seq_w, self_ast_w, encoder_attn_w, rem_attn_w = self.decoder(encoder_out, inputs, True)
        return self_seq_w, self_ast_w, encoder_attn_w, rem_attn_w

    def encode(self, inputs):
        return self.encoder(inputs)

    def forward(self, inputs):
        encoder_out = self.encoder(inputs)
        decoder_out = self.decoder(encoder_out, inputs)
        out_fathers, masked_out_fathers, out_edges, masked_out_edges, out_nodes = decoder_out

        add_mask = inputs['target_nodes'][:, :-1].unsqueeze(-1).expand(out_fathers.size())
        out_fathers = out_fathers.masked_select(add_mask > 0).view(-1, out_fathers.size(-1))
        masked_out_fathers = masked_out_fathers.masked_select(add_mask > 0).view(-1, masked_out_fathers.size(-1))
        target_fathers = inputs['target_out_fathers'].masked_select(inputs['target_nodes'][:, :-1] > 0).view(-1)

        add_mask = inputs['target_nodes'][:, :-1].unsqueeze(-1).expand(out_edges.size())
        out_edges = out_edges.masked_select(add_mask > 0).view(-1, out_edges.size(-1))
        masked_out_edges = masked_out_edges.masked_select(add_mask > 0).view(-1, masked_out_edges.size(-1))
        target_edges = inputs['target_out_edges'].masked_select(inputs['target_nodes'][:, :-1] > 0).view(-1)

        add_mask = inputs['target_nodes'][:, :-1].unsqueeze(-1).expand(out_nodes.size())
        out_nodes = out_nodes.masked_select(add_mask > 0).view(-1, out_nodes.size(-1))
        target_nodes = inputs['target_nodes'][:, 1:].masked_select(inputs['target_nodes'][:, 1:] > 0).view(-1)

        loss_fct = nn.NLLLoss(reduction='mean')
        rule_fct = nn.KLDivLoss(reduction='batchmean')
        father_rule_loss = rule_fct(torch.log(out_fathers), masked_out_fathers)
        # father_rule_loss = loss_fct(torch.log(out_fathers), target_fathers)
        father_loss = loss_fct(torch.log(masked_out_fathers), target_fathers)
        edge_rule_loss = rule_fct(torch.log(out_edges), masked_out_edges)
        # edge_rule_loss = loss_fct(torch.log(out_edges), target_edges)
        edge_loss = loss_fct(torch.log(masked_out_edges), target_edges)
        node_loss = loss_fct(torch.log(out_nodes), target_nodes)

        return father_rule_loss, father_loss, edge_rule_loss, edge_loss, node_loss


class GraphTransformerEncoder(nn.Module):
    def __init__(self, node_vocabulary, edge_vocabulary, node_embedding, edge_embedding, position_embedding, config):
        super().__init__()

        self.config = config
        self.node_vocabulary = node_vocabulary
        self.edge_vocabulary = edge_vocabulary
        self.node_embedding = node_embedding
        self.edge_embedding = edge_embedding
        self.position_embedding = position_embedding

        rem_tag_dim = 2
        self.rem_tag_embedding = nn.Embedding(3, rem_tag_dim, padding_idx=0)
        self.rem_tag_fusion = Conv1D(config.hidden_dim, config.hidden_dim + rem_tag_dim)

        self.blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_encoder_layer)]
        )

    def forward(self, inputs):
        nodes = inputs['nodes']  # [B, L]
        node_rem_tags = inputs['rem_tags']  # [B, L]
        edges = inputs['edges']  # [B, L, L]

        edge_embedding = self.edge_embedding(edges)
        nodes_h = self.node_embedding(nodes) + self.position_embedding(edges)
        rem_tag_embedding = self.rem_tag_embedding(node_rem_tags)
        nodes_h = self.rem_tag_fusion(torch.cat([nodes_h, rem_tag_embedding], dim=-1))

        # 1 for not mask, 0 for mask
        node_mask = (nodes.unsqueeze(1).unsqueeze(2) > 0).type_as(nodes_h)
        node_mask = (1.0 - node_mask) * -1.e16
        edge_mask = (edges.unsqueeze(1) > 0).type_as(nodes_h)
        edge_mask = (1.0 - edge_mask) * -1.e16

        for block in self.blocks:
            nodes_h = block(
                nodes_h,
                edge_embedding,
                node_mask, edge_mask,
            )

        # [B, L, H]
        return nodes_h


class GraphTransformerDecoder(nn.Module):
    def __init__(self, node_vocabulary, edge_vocabulary, node_embedding, edge_embedding, position_embedding, config):
        super().__init__()

        self.config = config
        self.node_vocabulary = node_vocabulary
        self.edge_vocabulary = edge_vocabulary
        self.node_embedding = node_embedding
        self.edge_embedding = edge_embedding
        self.position_embedding = position_embedding

        self.index_blocks = nn.ModuleList(
            [FatherDecoderBlock(config) for _ in range(config.num_father_layer)]
        )
        self.father_predictor = FatherPredictor(self.config)

        self.edge_blocks = nn.ModuleList(
            [EdgeDecoderBlock(config) for _ in range(config.num_edge_layer)]
        )
        self.edge_generator = EdgeGenerator(self.config)

        self.node_blocks = nn.ModuleList(
            [NodeDecoderBlock(config) for _ in range(config.num_node_layer)]
        )
        self.node_generator = NodeGenerator(self.config)

        self.grammar_checker = GrammarChecker(self.node_vocabulary, self.edge_vocabulary)

    def forward(self, encoder_out, inputs):
        nodes = inputs['target_nodes'][:, :-1]
        edges = inputs['target_edges']
        rem_tags = inputs['rem_tags']

        edge_embedding = self.edge_embedding(inputs['target_edges'])
        nodes_h = self.node_embedding(inputs['target_nodes'][:, :-1]) + self.position_embedding(inputs['target_edges'])

        # 1 for not mask, 0 for mask
        node_mask = (torch.tril(nodes.unsqueeze(1).unsqueeze(2).repeat(1, 1, nodes.size(-1), 1)) > 0).type_as(nodes_h)
        node_mask = (1.0 - node_mask) * -1.e16
        edge_mask = (torch.tril(edges).unsqueeze(1) > 0).type_as(nodes_h)
        edge_mask = (1.0 - edge_mask) * -1.e16
        encoder_mask = (inputs['nodes'].unsqueeze(1).unsqueeze(2) > 0).type_as(nodes_h)
        encoder_mask = (1.0 - encoder_mask) * -1.e16

        fathers_h = nodes_h
        for layer in self.index_blocks:
            fathers_h = layer(
                encoder_out, rem_tags,
                fathers_h,
                edge_embedding,
                node_mask, edge_mask, encoder_mask
            )
        soft_attention = self.father_predictor(fathers_h, node_mask.squeeze(1)) + 1.e-32
        soft_attention = nn.functional.normalize(soft_attention, p=1, dim=-1)
        masked_attention = self.grammar_checker.check_valid_fathers(soft_attention, nodes, edges)
        masked_attention = nn.functional.normalize(masked_attention, p=1, dim=-1)

        next_father_indices = inputs['target_out_fathers'].masked_select(inputs['target_nodes'][:, :-1] > 0).\
            split(inputs['target_size'].tolist())
        next_father_indices = torch.cat(
            [torch.cat([tensor, torch.zeros(fathers_h.size(1) - tensor.size(0)).type_as(tensor)], dim=0).unsqueeze(0)
             for tensor in next_father_indices], dim=0
        )
        edges_h = nodes_h + fathers_h
        for layer in self.edge_blocks:
            edges_h = layer(
                encoder_out, rem_tags,
                edges_h, next_father_indices,
                edge_embedding,
                node_mask, edge_mask, encoder_mask
            )
        out_edges = self.edge_generator(edges_h)
        masked_out_edges = self.grammar_checker.check_valid_edges(out_edges, nodes, next_father_indices)
        masked_out_edges = nn.functional.normalize(masked_out_edges, p=1, dim=-1)

        next_edge = inputs['target_out_edges'].masked_select(inputs['target_nodes'][:, :-1] > 0).\
            split(inputs['target_size'].tolist())
        next_edge = torch.cat(
            [torch.cat([tensor, torch.zeros(edges_h.size(1) - tensor.size(0)).type_as(tensor)], dim=0).unsqueeze(0)
             for tensor in next_edge], dim=0
        )
        next_edge_embedding = self.edge_embedding(next_edge)
        nodes_h = nodes_h + edges_h
        encoder_attn_w = None
        for layer in self.node_blocks:
            nodes_h, self_seq_w, self_ast_w, encoder_attn_w, rem_attn_w = layer(
                encoder_out, rem_tags,
                nodes_h, next_father_indices, next_edge_embedding,
                edge_embedding,
                node_mask, edge_mask, encoder_mask, True
            )
        out_nodes = self.node_generator(nodes_h, encoder_attn_w, inputs['nodes'])
        out_nodes = nn.functional.normalize(out_nodes, p=1, dim=-1)

        return soft_attention, masked_attention, out_edges, masked_out_edges, out_nodes


def init_weights(module):
    if isinstance(module, (nn.Linear, Conv1D)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
