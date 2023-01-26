import sys
import torch
import os

MODELS_DIR = os.path.abspath(__file__)
MODELS_DIR = MODELS_DIR[: MODELS_DIR.rfind('/') + 1]
sys.path.append(MODELS_DIR + '../grammar/')

from grammar import node_to_edge


class GrammarCheckerEval:
    def __init__(self, node_vocabulary, edge_vocabulary):
        self.node_vocabulary = node_vocabulary
        self.edge_vocabulary = edge_vocabulary

        self.node2edge_grammar = {}
        self.read_node2edge_grammar()

    def read_node2edge_grammar(self):
        for node, edges in node_to_edge.items():
            k = self.node_vocabulary.index(node)
            if k == self.node_vocabulary.unk_index:
                continue
            v = [(self.edge_vocabulary.unk_index, 0), (self.edge_vocabulary.eos_index, 0)]
            for (edge, option) in edges:
                edge = node + '->' + edge
                if self.edge_vocabulary.index(edge) != self.edge_vocabulary.unk_index:
                    v.append((self.edge_vocabulary.index(edge), option))
            self.node2edge_grammar[k] = v

    def find_unfinished_father(self, fathers, edges, nodes):
        """
        :param fathers: [B, L]
        :param edges: [B, L]
        :param nodes: [B, L + 1]
        :return:
        """
        batch_possible = []
        for i in range(nodes.size(0)):
            ast = {int(fathers.size(1)): []}
            for j in range(fathers.size(1)):
                father = int(fathers[i, j])
                edge = int(edges[i, j])
                if father not in ast:
                    ast[father] = []
                ast[father].append(edge)
            possible = []
            for j in range(nodes.size(1) - 1, -1, -1):
                if j == 0:
                    possible.append(j)
                    break
                if nodes[i, j] >= self.node_vocabulary.nonterminal_size:
                    continue
                father = int(nodes[i, j])
                requirement = {edge: option for (edge, option) in self.node2edge_grammar[father]}  \
                    if father in self.node2edge_grammar else {}
                possible.append(j)
                if j not in ast:
                    continue
                for edge in ast[j]:
                    if edge not in requirement:
                        continue
                    if requirement[edge] > 0:
                        requirement[edge] = 0
                if 1 in requirement.values() or 2 in requirement.values():
                    break
            batch_possible.append(possible)
        return batch_possible

    def check_valid_fathers(self, self_attention, nodes, edges, next_father_indices, next_edges):
        """
        :param self_attention: [B, L, L]
        :param nodes: [B, L + 1]
        :param edges: [B, L + 1, L + 1]
        :param next_father_indices: [B, L]
        :param next_edges: [B, L]
        :return:
        """
        # no cross edges
        valid_mask = torch.tril(torch.ones(self_attention.size())).type_as(edges) ^ 1
        for (i, j, k) in (torch.tril(edges) > 0).nonzero():
            valid_mask[i, j:, k+1:j] |= 1
        # terminal can't be father
        for (i, j) in (nodes >= self.node_vocabulary.nonterminal_size).nonzero():
            valid_mask[i, :, j] = 1
        
        if next_father_indices is not None:
            batch_possible = self.find_unfinished_father(next_father_indices, next_edges, nodes)
            for i in range(len(batch_possible)):
                mask = torch.ones(valid_mask.size(-1)).type_as(nodes)
                mask[torch.LongTensor(batch_possible[i]).type_as(nodes)] = 0
                valid_mask[i, -1] = valid_mask[i, -1] + mask
        masked_self_attention = self_attention.masked_fill(valid_mask > 0, 1.e-32)
        return masked_self_attention

    def check_valid_edges(self, out_edges, nodes, next_father_indices, src_len=None):
        """
        :param out_edges: [B, L, E]
        :param nodes: [B, L]
        :param next_father_indices: [B, L]
        :return:
        """
        node_indices = torch.gather(nodes, 1, next_father_indices)   # [B, L]
        masked_out_edges = out_edges.clone()
        for i in range(out_edges.size(0)):
            # for j in range(out_edges.size(1)):
            for j in (out_edges.size(1) - 1, ):
                if node_indices[i, j] == self.node_vocabulary.sos_index:
                    continue
                if int(node_indices[i, j]) in self.node2edge_grammar:
                    mask = [e for (e, o) in self.node2edge_grammar[int(node_indices[i, j])]]
                    mask = torch.LongTensor(mask).type_as(nodes)
                    masked_out_edges[i, j, mask] += 1
                    masked_out_edges[i, j] = torch.clamp(masked_out_edges[i, j] - 1, 1.e-32)
                if next_father_indices[i, j] != 0:
                    masked_out_edges[i, j, self.edge_vocabulary.eos_index] = 1.e-32

        return masked_out_edges

    def check_valid_node(self, out_nodes, nodes, next_father_indices, next_edges, identifier_semantics):
        """
        :param out_nodes: [B, L, E]
        :param nodes: [B, L]
        :param next_father_indices: [B, L]
        :param next_edges: [B, L]
        :param identifier_semantics: dict
        :return:
        """
        vars, methods, var_type, type_var, type_method, qualifier_var, qualifier_method, super_arg = identifier_semantics
        
        for i in range(out_nodes.size(0)):
            father = nodes[i, next_father_indices[i, -1]]
            edge = next_edges[i, -1]

            if father == self.node_vocabulary.index('MemberReference') and \
                edge == self.edge_vocabulary.index('MemberReference->member'):
                if next_father_indices[i, -1] == 0:
                    continue
                grandfather = nodes[i, next_father_indices[i, next_father_indices[i, -1] - 1]]
                grandedge = next_edges[i, next_father_indices[i, -1] - 1]
                possible = []
                if grandfather == self.node_vocabulary.index('SuperConstructorInvocation') and \
                        grandedge == self.edge_vocabulary.index('SuperConstructorInvocation->arguments'):
                    index_of_arg = ((next_father_indices[i] == next_father_indices[i, next_father_indices[i, -1] - 1]) &
                                    (next_edges[i] == grandedge)).sum() - 1
                    index_of_arg = int(index_of_arg)
                    if index_of_arg in super_arg:
                        possible += [self.node_vocabulary.index('VAR_<UNK>')]
                        for arg_type in super_arg[index_of_arg]:
                            if arg_type in type_var:
                                for arg in type_var[arg_type]:
                                    possible.append(self.node_vocabulary.index(arg))
                index_of_qualifier = ((next_father_indices[i] == next_father_indices[i, -1]) &
                                      (next_edges[i] == self.edge_vocabulary.index('MemberReference->qualifier'))).nonzero().squeeze(-1)
                if not possible and len(index_of_qualifier) == 1:
                    qualifier = nodes[i, index_of_qualifier[0] + 1]
                    if self.node_vocabulary[int(qualifier)] in qualifier_var:
                        possible += [self.node_vocabulary.index(token) for token in
                                     list(qualifier_var[self.node_vocabulary[int(qualifier)]])]
                        possible += [self.node_vocabulary.index('VAR_<UNK>')]
                    elif self.node_vocabulary[int(qualifier)] in var_type and \
                            list(var_type[self.node_vocabulary[int(qualifier)]])[0] in qualifier_var:
                        possible += [self.node_vocabulary.index(token) for token in
                                     list(qualifier_var[list(var_type[self.node_vocabulary[int(qualifier)]])[0]])]
                        possible += [self.node_vocabulary.index('VAR_<UNK>')]
                if not possible:
                    possible += [self.node_vocabulary.index(token) for token in vars] + [self.node_vocabulary.index('VAR_<UNK>')]
                possible = list(set(possible))
                mask = torch.LongTensor(possible).type_as(nodes)
                out_nodes[i, -1, mask] += 1
                out_nodes[i, -1] = torch.clamp(out_nodes[i, -1] - 1, 1.e-32)
            elif father == self.node_vocabulary.index('MethodInvocation') and \
                    edge == self.edge_vocabulary.index('MethodInvocation->member'):
                possible = []
                index_of_qualifier = ((next_father_indices[i] == next_father_indices[i, -1]) &
                                      (next_edges[i] == self.edge_vocabulary.index('MethodInvocation->qualifier'))).nonzero().squeeze(-1)
                if len(index_of_qualifier) == 1:
                    qualifier = nodes[i, index_of_qualifier[0] + 1]
                    if self.node_vocabulary[int(qualifier)] in qualifier_method:
                        possible += [self.node_vocabulary.index(token) for token in
                                     list(qualifier_method[self.node_vocabulary[int(qualifier)]])]
                        possible += [self.node_vocabulary.index('METHOD_<UNK>')]
                    elif self.node_vocabulary[int(qualifier)] in var_type and \
                            list(var_type[self.node_vocabulary[int(qualifier)]])[0] in qualifier_method:
                        possible += [self.node_vocabulary.index(token) for token in
                                     list(qualifier_method[list(var_type[self.node_vocabulary[int(qualifier)]])[0]])]
                        possible += [self.node_vocabulary.index('METHOD_<UNK>')]
                if not possible:
                    possible += [self.node_vocabulary.index(token) for token in methods] + [self.node_vocabulary.index('METHOD_<UNK>')]
                possible = list(set(possible))
                mask = torch.LongTensor(possible).type_as(nodes)
                out_nodes[i, -1, mask] += 1
                out_nodes[i, -1] = torch.clamp(out_nodes[i, -1] - 1, 1.e-32)
            elif father == self.node_vocabulary.index('Literal') and \
                edge == self.edge_vocabulary.index('Literal->value'):
                if next_father_indices[i, -1] == 0:
                    continue
                grandfather = nodes[i, next_father_indices[i, next_father_indices[i, -1] - 1]]
                grandedge = next_edges[i, next_father_indices[i, -1] - 1]
                if grandfather == self.node_vocabulary.index('BinaryOperation') and \
                        grandedge == self.edge_vocabulary.index('BinaryOperation->operandr'):
                    index_of_operandl = ((next_father_indices[i] == next_father_indices[i, next_father_indices[i, -1] - 1]) &
                                         (next_edges[i] == self.edge_vocabulary.index('BinaryOperation->operandl'))).nonzero().squeeze(-1)
                    if len(index_of_operandl) == 0 or index_of_operandl[0] + 2 >= nodes.size(1):
                        continue
                    if nodes[i, index_of_operandl[0] + 1] == self.node_vocabulary.index('MemberReference'):
                        name = self.node_vocabulary[int(nodes[i, index_of_operandl[0] + 2])]
                        if name in var_type:
                            if list(var_type[name]) == ['boolean']:
                                possible = [self.node_vocabulary.index("true"), self.node_vocabulary.index("false")]
                                mask = torch.LongTensor(possible).type_as(nodes)
                                out_nodes[i, -1, mask] += 1
                                out_nodes[i, -1] = torch.clamp(out_nodes[i, -1] - 1, 1.e-32)
            elif 'statements' in self.edge_vocabulary[int(edge)] or 'arguments' in self.edge_vocabulary[int(edge)]:
                impossible = [i for i in range(len(self.node_vocabulary)) if i >= self.node_vocabulary.nonterminal_size]
                out_nodes[i, -1, torch.LongTensor(impossible).type_as(nodes)] = 1.e-32
            if edge != self.edge_vocabulary.eos_index:
                out_nodes[i, -1, self.node_vocabulary.eos_index] = 1.e-32
        return out_nodes
