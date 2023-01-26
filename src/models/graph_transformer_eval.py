import torch
import torch.nn as nn
from grammar_checker_eval import GrammarCheckerEval


class GraphTransformerEval:
    def __init__(self, model, gpu1, gpu2):
        model.decoder.grammar_checker = GrammarCheckerEval(model.node_vocabulary, model.edge_vocabulary)
        self.model = model
        self.model.eval()
        self.gpu1 = gpu1
        self.gpu2 = gpu2
        self.split_size = 1000

    def encode(self, inputs):
        return self.model.encoder(inputs)

    def decode(self, encoder_out, decoder_input, edge_of_rem=None, scores=None,
               father_beam_size=3, edge_beam_size=3, beam_size=1000, identifier_semantics=None, src_len=None):
        encoder_out = encoder_out.to(self.gpu2)

        # inputs = {k: v.to(self.gpu2) if torch.is_tensor(v) else v for k, v in decoder_input.items()}
        inputs = decoder_input
        nodes = inputs['target_nodes']
        edges = inputs['target_edges']
        rem_tags = inputs['rem_tags']

        edge_embedding = self.model.decoder.edge_embedding(edges.to(self.gpu1)).to(self.gpu2)
        nodes_h = self.model.decoder.node_embedding(nodes.to(self.gpu1)) + \
            self.model.decoder.position_embedding(edges.to(self.gpu1))
        nodes_h = nodes_h.to(self.gpu2)

        # 1 for not mask, 0 for mask
        node_mask = (torch.tril(nodes.unsqueeze(1).unsqueeze(2).repeat(1, 1, nodes.size(-1), 1)) > 0).type_as(nodes_h)
        node_mask = (1.0 - node_mask) * -1.e16
        edge_mask = (torch.tril(edges).unsqueeze(1) > 0).type_as(nodes_h)
        edge_mask = (1.0 - edge_mask) * -1.e16

        fathers_h = nodes_h
        for layer in self.model.decoder.index_blocks:
            fathers_h = layer.to(self.gpu2)(
                encoder_out, rem_tags,
                fathers_h,
                edge_embedding,
                node_mask, edge_mask, encoder_mask=None
            )
        soft_attention = self.model.decoder.father_predictor.to(self.gpu2)(fathers_h, attention_mask=node_mask.squeeze(1))
        # if identifier_semantics:
        #    masked_attention = self.model.decoder.grammar_checker.check_valid_fathers(
        #        soft_attention, nodes, edges,
        #        decoder_input['next_father_indices'], decoder_input['next_edges']
        #   )
        # else:
        #    masked_attention = soft_attention
        masked_attention = soft_attention
        masked_attention = nn.functional.normalize(masked_attention, p=1, dim=-1)

        next_father_scores, next_father_indices = masked_attention.topk(k=father_beam_size, dim=-1)  # [B, L, 3]
        next_father_scores = torch.log(next_father_scores[:, -1, :].reshape(-1))
        next_father_indices = next_father_indices[:, -1, :].reshape(-1, 1)  # [B1_1, B1_2, B1_3, B2...], [3B, 1]
        final_scores = next_father_scores
        if torch.is_tensor(scores):
            final_scores = final_scores + scores.repeat_interleave(father_beam_size, dim=0).type_as(final_scores)
        sort_scores, sort_order = final_scores.sort(descending=True)
        final_scores = sort_scores[: beam_size]
        sort_order = sort_order[: beam_size]
        beam_ids = sort_order // father_beam_size
        if torch.is_tensor(inputs['next_father_indices']):
            next_father_indices = torch.cat([
                inputs['next_father_indices'].repeat_interleave(father_beam_size, dim=0),
                next_father_indices
            ], dim=1)
        next_father_indices = next_father_indices[sort_order]
        nodes_h = nodes_h.repeat_interleave(father_beam_size, dim=0)[sort_order]
        fathers_h = fathers_h.repeat_interleave(father_beam_size, dim=0)[sort_order]
        edge_embedding = edge_embedding.repeat_interleave(father_beam_size, dim=0)[sort_order]
        node_mask = node_mask.repeat_interleave(father_beam_size, dim=0)[sort_order]
        edge_mask = edge_mask.repeat_interleave(father_beam_size, dim=0)[sort_order]
        nodes = nodes.repeat_interleave(father_beam_size, dim=0)[sort_order]
        edges_h = nodes_h + fathers_h  # [B, L, H]
        for layer in self.model.decoder.edge_blocks:
            edges_h = layer.to(self.gpu2)(
                encoder_out, rem_tags,
                edges_h, next_father_indices,
                edge_embedding,
                node_mask, edge_mask, encoder_mask=None
            )
        out_edges = self.model.decoder.edge_generator.to(self.gpu2)(edges_h)
        if identifier_semantics:
            masked_out_edges = self.model.decoder.grammar_checker.check_valid_edges(
                out_edges, nodes, next_father_indices, src_len=src_len
            )   # [B, L, H]
            masked_out_edges = nn.functional.normalize(masked_out_edges, p=1, dim=-1)
        else:
            masked_out_edges = out_edges
        if edge_of_rem is not None:
            masked_out_edges[:, 0, edge_of_rem] = 1.

        next_edge_scores, next_edges = masked_out_edges.topk(k=edge_beam_size, dim=-1)     # [B, L, 3]
        next_edge_scores = torch.log(next_edge_scores[:, -1, :].reshape(-1))
        next_edges = next_edges[:, -1, :].reshape(-1, 1)    # [B1_1, B1_2, B1_3, B2...], [3B, 1]
        final_scores = final_scores.repeat_interleave(edge_beam_size, dim=0) + next_edge_scores
        sort_scores, sort_order = final_scores.sort(descending=True)
        final_scores = sort_scores[:beam_size]
        sort_order = sort_order[:beam_size]
        if torch.is_tensor(inputs['next_edges']):
            next_edges = torch.cat([
                inputs['next_edges'][beam_ids].repeat_interleave(edge_beam_size, dim=0),
                next_edges
            ], dim=-1)
        next_edges = next_edges[sort_order]
        next_edge_embedding = self.model.decoder.edge_embedding(next_edges.to(self.gpu1)).to(self.gpu2)
        beam_ids = beam_ids.repeat_interleave(edge_beam_size, dim=0)[sort_order]
        nodes_h = nodes_h.repeat_interleave(edge_beam_size, dim=0)[sort_order]
        edges_h = edges_h.repeat_interleave(edge_beam_size, dim=0)[sort_order]
        next_father_indices = next_father_indices.repeat_interleave(edge_beam_size, dim=0)[sort_order]
        edge_embedding = edge_embedding.repeat_interleave(edge_beam_size, dim=0)[sort_order]
        node_mask = node_mask.repeat_interleave(edge_beam_size, dim=0)[sort_order]
        edge_mask = edge_mask.repeat_interleave(edge_beam_size, dim=0)[sort_order]
        nodes = nodes.repeat_interleave(edge_beam_size, dim=0)[sort_order]
        nodes_h = nodes_h + edges_h     # [B, L, H]
        encoder_attn_w = None
        for layer in self.model.decoder.node_blocks:
            nodes_h, _, _, encoder_attn_w, _ = layer.to(self.gpu2)(
                encoder_out, rem_tags,
                nodes_h, next_father_indices, next_edge_embedding,
                edge_embedding,
                node_mask, edge_mask, encoder_mask=None, return_weight=True
            )
        out_nodes = self.model.decoder.node_generator.to(self.gpu2)(nodes_h, encoder_attn_w, inputs['nodes'])
        if identifier_semantics:
            out_nodes = self.model.decoder.grammar_checker.check_valid_node(
                out_nodes, nodes, next_father_indices, next_edges, identifier_semantics
            )
            out_nodes = nn.functional.normalize(out_nodes, p=1, dim=-1)

        return beam_ids, next_father_indices[:, -1], next_edges[:, -1], \
            final_scores, torch.log(out_nodes[:, -1, :])

