import codecs
import sys
import time
import traceback
import os
import torch
from beamsearch import BeamSearch

TESTER_DIR = os.path.abspath(__file__)
TESTER_DIR = TESTER_DIR[: TESTER_DIR.rfind('/') + 1]

sys.path.append(TESTER_DIR + '../dataset/')
sys.path.append(TESTER_DIR + '../models/')
from vocabulary import NodeVocabulary, EdgeVocabulary
from graph_transformer_dataset import GraphTransformerDataset
from graph_transformer import GraphTransformer, Config
from graph_transformer_eval import GraphTransformerEval
from identifier_dataset import IdentifierDataset


def read_defects4j_meta(meta_file):
    meta = []
    fp = codecs.open(meta_file, 'r', 'utf-8')
    for l in fp.readlines():
        if len(l.strip().split()) == 8:
            proj, bug_id, path, r_s, r_e, a_s, a_e, tag = l.strip().split()
        elif len(l.strip().split()) == 6:
            proj, bug_id, path, r_s, r_e, tag = l.strip().split()
        elif len(l.strip().split()) == 5:
            proj, bug_id, path, r_s, r_e = l.strip().split()
        elif len(l.strip().split()) == 4:
            proj, bug_id, path, r_s = l.strip().split()
        meta.append([proj + '_' + bug_id, path])
    return meta


def read_quixbugs_meta(meta_file):
    meta = []
    fp = codecs.open(meta_file, 'r', 'utf-8')
    for l in fp.readlines():
        if len(l.split()) == 6:
            proj, r_s, r_e, a_s, a_e, tag = l.strip().split()
        elif len(l.split()) == 4:
            proj, r_s, r_e, tag = l.strip().split()
        meta.append([proj, proj + '.java'])
    return meta


class Generator:
    def __init__(self, model, node_vocabulary, edge_vocabulary, dataset,
                 beam_size=10, identifier_dataset=None, class_name_list=None):
        self.model = model
        self.node_vocabulary = node_vocabulary
        self.edge_vocabulary = edge_vocabulary
        self.dataset = dataset
        self.identifier_dataset = identifier_dataset
        self.class_name_list = class_name_list
        self.beamsearch = BeamSearch(model, node_vocabulary, edge_vocabulary, beam_size)
        print('beam size:', beam_size)

    def generate(self, output_path, father_beam_size=3, edge_beam_size=3):
        wp = codecs.open(output_path, 'w', 'utf-8')
        oom = []
        for i in range(self.dataset.total_size):
            inputs = self.dataset[i]
            id = self.dataset.data[i]['id']
            src_len = len(inputs['nodes'].masked_select(mask=inputs['rem_tags'] > 1))
            tgt_len = inputs['target_nodes'].size(1)
            print(i, 'src:', src_len, 'tgt:', tgt_len, 'oom:', len(oom))
            start_time = time.time()
            try:
                identifier_semantics = None
                if self.identifier_dataset is not None:
                    identifier_semantics = self.identifier_dataset.prepare(int(id), self.class_name_list[int(id) - 1])
                with torch.no_grad():
                    hypothesis = self.beamsearch.generate(
                        {k: v.to(self.model.gpu1) for k, v in inputs.items()},
                        max_step=96,
                        father_beam_size=father_beam_size, edge_beam_size=edge_beam_size,
                        identifier_semantics=identifier_semantics
                    )
            except Exception as e:
                oom.append(id)
                traceback.print_exc()
                continue

            wp.write('S-n-{}\t'.format(id))
            wp.write(self.node_vocabulary.string(inputs['nodes'].masked_select(mask=inputs['rem_tags'] > 1)) + '\n')
            wp.write('T-f-{}\t'.format(id))
            wp.write(' '.join([str(t) for t in inputs['target_out_fathers'][0].tolist()]) + '\n')
            wp.write('T-e-{}\t'.format(id))
            wp.write(self.edge_vocabulary.string(inputs['target_out_edges'][0]) + '\n')
            wp.write('T-n-{}\t'.format(id))
            wp.write(self.node_vocabulary.string(inputs['target_nodes'][0][1:]) + '\n')
            for h in hypothesis:
                wp.write('H-p-{}\t{}\n'.format(id, str(h['final_score'])))
                wp.write('H-f-{}\t'.format(id))
                wp.write(' '.join([str(t) for t in h['fathers'].tolist()]) + '\n')
                wp.write('H-e-{}\t'.format(id))
                wp.write(self.edge_vocabulary.string(h['edges']) + '\n')
                wp.write('H-n-{}\t'.format(id))
                wp.write(self.node_vocabulary.string(h['nodes']) + '\n')
            print(str(int(time.time() - start_time)) + 's')
        wp.close()
        print(len(oom), oom)


def generate_knod(model_file, input_file, identifier_file, class_name_list, devices, beam_size, output_file):
    node_vocabulary = NodeVocabulary(
        nonterminal_file=TESTER_DIR + '../../data/vocabulary/nodes_nonterminal.txt',
        terminal_file=TESTER_DIR + '../../data/vocabulary/nodes_terminal.txt',
        abstraction_file=TESTER_DIR + '../../data/vocabulary/abstractions.txt',
        idiom_file=TESTER_DIR + '../../data/vocabulary/idioms.txt',
        nonidentifier_file=TESTER_DIR + '../../data/vocabulary/nonidentifiers.txt'
    )
    edge_vocabulary = EdgeVocabulary(TESTER_DIR + '../../data/vocabulary/specified_edges.txt')
    print('finish loading vocabulary, node vocabulary:', len(node_vocabulary),
          ', edge vocabulary:', len(edge_vocabulary))

    dataset = GraphTransformerDataset(
        ast_files=[input_file],
        node_vocabulary=node_vocabulary,
        edge_vocabulary=edge_vocabulary,
        batch_size=1,
        shuffle=False,
        gpu_num=1,
        train=False
    )
    print(len(dataset), 'last id:', dataset.data[-1]['id'])

    load = torch.load(model_file, map_location='cuda:' + str(devices[0]))
    config = load['config']
    config = Config(
        node_vocabulary=node_vocabulary,
        edge_vocabulary=edge_vocabulary,
        hidden_dim=config['hidden_dim'],
        edge_dim=config['edge_dim'],
        num_head=config['num_head'],
        num_encoder_layer=config['num_encoder_layer'],
        num_father_layer=config['num_father_layer'],
        num_edge_layer=config['num_edge_layer'],
        num_node_layer=config['num_node_layer'],
        dropout=0.
    )
    model = GraphTransformer(config).to('cuda:' + str(devices[0]))
    model.load_state_dict(load['model'])
    model = GraphTransformerEval(model, devices[0], devices[1])

    identifier_dataset = IdentifierDataset(
        identifier_file, 
        TESTER_DIR + '../../data/vocabulary/idioms.txt',
        input_file
    )

    generator = Generator(
        model, node_vocabulary, edge_vocabulary, dataset, beam_size=beam_size,
        identifier_dataset=identifier_dataset, class_name_list=class_name_list
    )
    generator.generate(output_file)


if __name__ == "__main__":
    input_dir = TESTER_DIR + '../../data/defects4j_input/'
    output_dir = TESTER_DIR + '../../data/defects4j_output/'
    meta = read_defects4j_meta(input_dir + 'meta_localize.txt')
    model_file = '../../data/models/general_1.pt'
    
    input_file = input_dir + 'input_general_ast.json'
    identifier_file = input_dir + 'identifiers.json'
    output_file = output_dir + 'general_1.txt'

    class_name_list = [filename.split('/')[-1][:-5] for (bug, filename) in meta]
    devices = [0, 0]    # require two devices, but could be the same one
    beam_size = 200

    generate_knod(model_file, input_file, identifier_file, class_name_list, devices, beam_size, output_file)
