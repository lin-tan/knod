import copy
import json
import torch
import random
import numpy as np

from utils import batch_input
from vocabulary import NodeVocabulary, EdgeVocabulary


class GraphTransformerDataset(torch.utils.data.Dataset):
    def __init__(self, ast_files, node_vocabulary, edge_vocabulary, batch_size, shuffle=True, gpu_num=4, train=True):
        super(GraphTransformerDataset, self).__init__()

        self.ast_files = ast_files
        self.node_vocabulary = node_vocabulary
        self.edge_vocabulary = edge_vocabulary
        self.gpu_num = gpu_num
        self.train = train

        self.data = []
        self.id2data = {}
        nonterminals = set(self.node_vocabulary.symbols[: self.node_vocabulary.nonterminal_size])
        discard = [0, 0, 0, 0]
        for ast_file in ast_files:
            for data in json.load(open(ast_file, 'r')):
                if len(data['rem_roots']) != 1:
                    discard[0] += 1
                    continue
                if len(data['nodes']) >= 512 or (
                        len(data['add_roots']) > 0 and len(data['add_roots'][0]['nodes']) >= 256):
                    discard[1] += 1
                    continue
                if 0 in data['rem_roots']:
                    discard[2] += 1
                    continue
                if len(set(list(data['mappings'])) & nonterminals) > 0:
                    discard[3] += 1
                    continue

                self.data.append(data)
                self.id2data[data['id']] = data
        print('discard:', discard, ', sum:', sum(discard))
        if shuffle:
            self.shuffle()

        self.total_size = len(self.data)
        self.max_target_len = 512
        self.batch_size = batch_size

    def __len__(self):
        batched_size = self.total_size // self.batch_size
        return batched_size

    def __getitem__(self, item):
        batch = self.prepare_data(self.batch_size * item, self.batch_size * (item + 1))
        batch = batch_input(batch)

        return batch

    def shuffle(self):
        random.shuffle(self.data)

    def prepare_data(self, start, end):
        batch = []
        for data in self.data[start: end]:
            mapping = data['mappings']

            father2child = {}
            for edge in data['edges']:
                father, child, ty = edge
                if father not in father2child:
                    father2child[father] = []
                father2child[father].append(child)

            # prepare nodes, edges
            nodes = [self.node_vocabulary.index(data['nodes'][0], mapping=mapping)]
            edges = [[0]*(len(data['nodes']) + 1) for _ in range(len(data['nodes']) + 1)]
            for edge in data['edges']:
                father, child, ty = edge
                ty = self.edge_vocabulary.index(data['nodes'][father] + '->' + ty)
                edges[father][child] = edges[child][father] = ty
                nodes.append(self.node_vocabulary.index(data['nodes'][child], mapping=mapping))
            nodes.append(self.node_vocabulary.eos_index)
            edges[0][-1] = edges[-1][0] = self.edge_vocabulary.eos_index

            for i in range(len(edges)):
                edges[i][i] = self.edge_vocabulary.self_loop_index
            for father, children in father2child.items():
                for i in range(1, len(children)):
                    edges[children[i]][children[i - 1]] = \
                        edges[children[i - 1]][children[i]] = self.edge_vocabulary.sibling_index

            # prepare node_rem_tag
            rem_nodes = set(data['rem_roots'])
            rem_tags = [1] * len(nodes)
            for rem_root in rem_nodes:
                rem_tags[rem_root] = 2
            for edge in data['edges']:
                father, child, edge_type = edge
                if father in rem_nodes:
                    rem_nodes.add(child)
                    rem_tags[child] = 2

            # prepare target
            target_nodes, target_edges, target_out_fathers, target_out_edges = self.prepare_target(data)

            batch.append({
                'nodes': nodes,
                'rem_tags': rem_tags,
                'edges': edges,
                'target_nodes': target_nodes,
                'target_edges': target_edges,
                'target_out_fathers': target_out_fathers,
                'target_out_edges': target_out_edges
            })

        return batch

    def prepare_input_with_target(self, ids, target_father, target_edge, target_node):
        data = copy.deepcopy(self.id2data[ids])

        # rebuild the 'add_roots' in data
        target_father, target_edge, target_node = target_father.split(), target_edge.split(), target_node.split()
        assert len(target_father) == len(target_edge) == len(target_node), "target father/edge/node should have same length."
        assert target_node.count('<EOS>') == 1, "currently only support target_node.count(<EOS>) == 1"

        target_father, target_edge, target_node = target_father[:-1], target_edge[:-1], target_node[:-1]
        add_root = {
            'nodes': target_node,
            'edges': []
        }
        for i, father, edge in enumerate(zip(target_father[1:], target_edge[1:])):
            add_root['edges'].append([int(father), i, edge])
        data['add_roots'] = [add_root]

        mapping = data['mappings']
        father2child = {}
        for edge in data['edges']:
            father, child, ty = edge
            if father not in father2child:
                father2child[father] = []
            father2child[father].append(child)

        # prepare nodes, edges
        nodes = [self.node_vocabulary.index(data['nodes'][0], mapping=mapping)]
        edges = [[0]*(len(data['nodes']) + 1) for _ in range(len(data['nodes']) + 1)]
        for edge in data['edges']:
            father, child, ty = edge
            ty = self.edge_vocabulary.index(data['nodes'][father] + '->' + ty)
            edges[father][child] = edges[child][father] = ty
            nodes.append(self.node_vocabulary.index(data['nodes'][child], mapping=mapping))
        nodes.append(self.node_vocabulary.eos_index)
        edges[0][-1] = edges[-1][0] = self.edge_vocabulary.eos_index

        for i in range(len(edges)):
            edges[i][i] = self.edge_vocabulary.self_loop_index
        for father, children in father2child.items():
            for i in range(1, len(children)):
                edges[children[i]][children[i - 1]] = \
                    edges[children[i - 1]][children[i]] = self.edge_vocabulary.sibling_index
        
        # prepare node_rem_tag
        rem_nodes = set(data['rem_roots'])
        rem_tags = [1] * len(nodes)
        for rem_root in rem_nodes:
            rem_tags[rem_root] = 2
        for edge in data['edges']:
            father, child, edge_type = edge
            if father in rem_nodes:
                rem_nodes.add(child)
                rem_tags[child] = 2
        
        # prepare target
        target_nodes, target_edges, target_out_fathers, target_out_edges = self.prepare_target(data)

        inputs = []
        inputs.append({
            'nodes': nodes,
            'rem_tags': rem_tags,
            'edges': edges,
            'target_nodes': target_nodes,
            'target_edges': target_edges,
            'target_out_fathers': target_out_fathers,
            'target_out_edges': target_out_edges
        })

        return batch_input(inputs)

    def prepare_target(self, data):
        mapping = data['mappings']

        for edge in data['edges']:
            father, child, ty = edge
            if child == data['rem_roots'][0]:
                father_of_rem, edge_of_rem = father, ty
                edge_of_rem = self.edge_vocabulary.index(
                    data['nodes'][father_of_rem] + '->' + edge_of_rem
                )

        nodes_num = sum([len(add_root['nodes']) for add_root in data['add_roots']]) + 1
        target_nodes = [self.node_vocabulary.index(data['nodes'][father_of_rem], mapping=mapping)]
        target_edges = np.zeros((nodes_num, nodes_num), dtype='int')
        target_out_fathers = []
        target_out_edges = []

        if len(data['add_roots']) == 0 and (not self.train):
            target_nodes.append(0)
            target_out_fathers.append(0)
            target_out_edges.append(edge_of_rem)
            nodes_num += 1

        father2child = {}
        for add_root in data['add_roots']:
            target_edges[0, len(target_nodes)] = \
                target_edges[len(target_nodes), 0] = edge_of_rem
            if 0 not in father2child:
                father2child[0] = []
            father2child[0].append(len(target_nodes))
            target_size = len(target_nodes)
            target_nodes += [self.node_vocabulary.index(add_root['nodes'][0], mapping=mapping)]
            target_out_fathers += [0]
            target_out_edges += [edge_of_rem]
            for edge in add_root['edges']:
                father, child, ty = edge
                ty = self.edge_vocabulary.index(
                    add_root['nodes'][father] + '->' + ty
                )
                target_edges[target_size + father, target_size + child] = \
                    target_edges[target_size + child, target_size + father] = ty
                if target_size + father not in father2child:
                    father2child[target_size + father] = []
                father2child[target_size + father].append(target_size + child)
                target_nodes += [self.node_vocabulary.index(add_root['nodes'][child], mapping=mapping)]
                target_out_fathers += [target_size + father]
                target_out_edges += [ty]
        target_nodes += [self.node_vocabulary.eos_index]
        target_out_fathers += [0]
        target_out_edges += [self.edge_vocabulary.eos_index]

        for i in range(len(target_edges)):
            target_edges[i][i] = self.edge_vocabulary.self_loop_index
        for father, children in father2child.items():
            for i in range(1, len(children)):
                target_edges[children[i]][children[i - 1]] = \
                    target_edges[children[i - 1]][children[i]] = self.edge_vocabulary.sibling_index
        target_edges = target_edges.tolist()

        assert len(target_nodes) == len(target_out_fathers) + 1 == len(target_out_edges) + 1, \
            "{},{},{}".format(len(target_nodes), len(target_out_fathers), len(target_out_edges))
        assert len(target_nodes) == nodes_num + 1, \
            "{},{}".format(len(target_nodes), nodes_num + 1)

        return target_nodes, target_edges, target_out_fathers, target_out_edges


if __name__ == "__main__":
    node_vocabulary = NodeVocabulary(
        nonterminal_file='../../data/vocabulary/nodes_nonterminal.txt',
        terminal_file='../../data/vocabulary/nodes_terminal.txt',
        abstraction_file='../../data/vocabulary/abstractions.txt',
        idiom_file='../../data/vocabulary/idioms.txt',
        nonidentifier_file='../../data/vocabulary/nonidentifiers.txt'
    )
    edge_vocabulary = EdgeVocabulary('../../data/vocabulary/specified_edges.txt')
    print('finish loading vocabulary, node vocabulary:', len(node_vocabulary),
          ', edge vocabulary:', len(edge_vocabulary))

    dataset = GraphTransformerDataset(
        ast_files=[
            '../../data/training_ast.json',
        ],
        node_vocabulary=node_vocabulary,
        edge_vocabulary=edge_vocabulary,
        batch_size=1
    )
    print('finish loading', dataset.total_size)


