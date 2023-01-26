import sys
import torch
import os

MODELS_DIR = os.path.abspath(__file__)
MODELS_DIR = MODELS_DIR[: MODELS_DIR.rfind('/') + 1]
sys.path.append(MODELS_DIR + '../grammar/')

from grammar import node_to_edge


class GrammarChecker:
    def __init__(self, node_vocabulary, edge_vocabulary):
        self.node_vocabulary = node_vocabulary
        self.edge_vocabulary = edge_vocabulary

        self.node2edge_grammar = {}
        self.read_node2edge_grammar()

    def read_node2edge_grammar(self):
        for node, edges in node_to_edge.items():
            k = self.node_vocabulary.index(node)
            v = [self.edge_vocabulary.unk_index, self.edge_vocabulary.eos_index]    # <unk>, <eos>
            for edge in edges:
                edge = edge[0]
                edge = node + '->' + edge
                if self.edge_vocabulary.index(edge) != self.edge_vocabulary.unk_index:
                    v.append(self.edge_vocabulary.index(edge))
            if k != self.node_vocabulary.unk_index:
                val = []
                for i in range(len(self.edge_vocabulary)):
                    if i not in v:
                        val.append(i)
                self.node2edge_grammar[k] = torch.LongTensor(val)

    def check_valid_fathers(self, self_attention, nodes, edges):
        """
        :param self_attention: [B, L, L]
        :param nodes: [B, L]
        :param edges: [B, L, L]
        :return:
        """
        # no cross edges
        valid_mask = torch.tril(torch.ones(self_attention.size())).type_as(edges) ^ 1
        for (i, j, k) in (torch.tril(edges) > 0).nonzero():
            valid_mask[i, j:, k+1:j] |= 1
        # terminal can't be father
        for (i, j) in (nodes >= self.node_vocabulary.nonterminal_size).nonzero():
            valid_mask[i, :, j] = 1

        # for i in range(nodes.size(0)):
        #    for j in range(nodes.size(1)):

        masked_self_attention = self_attention.masked_fill(valid_mask > 0, 1.e-32)
        return masked_self_attention

    def check_valid_edges(self, out_edges, nodes, father_indices):
        """
        :param out_edges: [B, L, E]
        :param nodes: [B, L]
        :param father_indices: [B, L]
        :return:
        """
        node_indices = torch.gather(nodes, 1, father_indices)   # [B, L]
        masked_out_edges = out_edges.clone()
        for i in range(out_edges.size(0)):
            for j in range(out_edges.size(1)):
                if node_indices[i, j] == self.node_vocabulary.sos_index:
                    continue
                if int(node_indices[i, j]) in self.node2edge_grammar:
                    masked_out_edges[i, j, self.node2edge_grammar[int(node_indices[i, j])]] = 1.e-32
                if father_indices[i, j] != 0:
                    masked_out_edges[i, j, self.edge_vocabulary.eos_index] = 1.e-32

        return masked_out_edges
