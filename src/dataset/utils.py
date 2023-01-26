import torch


def batch_input(batch):
    max_node_num = max([len(b['nodes']) for b in batch])
    max_target_num = max([len(b['target_edges']) for b in batch])

    nodes, edges, rem_tags = [], [], []
    target_nodes, target_edges = [], []
    target_out_fathers, target_out_edges = [], []
    target_size = []
    for b in batch:
        nodes.append(
            b['nodes'] + [0]*(max_node_num - len(b['nodes']))
        )
        rem_tags.append(
            b['rem_tags'] + [0]*(max_node_num - len(b['rem_tags']))
        )
        edges.append(
            [row + [0]*(max_node_num - len(row)) for row in b['edges']] +
            [[0]*max_node_num for _ in range(max_node_num - len(b['edges']))]
        )
        target_nodes.append(
            b['target_nodes'][:-1] + [0]*(max_target_num - len(b['target_nodes'][:-1])) + b['target_nodes'][-1:]
        )
        target_edges.append(
            [row + [0]*(max_target_num - len(row)) for row in b['target_edges']] +
            [[0]*max_target_num for _ in range(max_target_num - len(b['target_edges']))]
        )
        target_out_fathers.append(
            b['target_out_fathers'] + [0] * (max_target_num - len(b['target_out_fathers']))
        )
        target_out_edges.append(
            b['target_out_edges'] + [0] * (max_target_num - len(b['target_out_edges']))
        )
        target_size.append(len(b['target_nodes']) - 1)

    return {
        'nodes': torch.LongTensor(nodes),
        'rem_tags': torch.LongTensor(rem_tags),
        'edges': torch.LongTensor(edges),
        'target_nodes': torch.LongTensor(target_nodes),
        'target_edges': torch.LongTensor(target_edges),
        'target_out_fathers': torch.LongTensor(target_out_fathers),
        'target_out_edges': torch.LongTensor(target_out_edges),
        'target_size': torch.LongTensor(target_size),
    }


