import json
import sys
import os

SRC_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]

sys.path.append(SRC_DIR + '/tester/')
sys.path.append(SRC_DIR + '/validation/')

from generator import generate_knod, read_defects4j_meta
from defects4j_rerank import rerank_patches_by_rank
from defects4j_validate import validate_defects4j


def generate_general_ast_patches(input_dir, output_dir, model_file):
    # step 1: generate the AST of patches
    meta = read_defects4j_meta(input_dir + 'meta_localize.txt')
    identifier_file = input_dir + 'identifiers.json'
    class_name_list = [filename.split('/')[-1][:-5] for (bug, filename) in meta]
    
    # use general.pt to generate general patches
    # model_file = SRC_DIR + '../data/models/general.pt'
    input_file = input_dir + 'input_general_ast.json'
    output_file = output_dir + 'general.txt'
    generate_knod(model_file, input_file, identifier_file, class_name_list, devices, beam_size, output_file)


def generate_insert_ast_patches(input_dir, output_dir, model_file):
    # step 1: generate the AST of patches
    meta = read_defects4j_meta(input_dir + 'meta_localize.txt')
    identifier_file = input_dir + 'identifiers.json'
    class_name_list = [filename.split('/')[-1][:-5] for (bug, filename) in meta]
    
    # use insert.pt to generate insertion patches
    # model_file = SRC_DIR + '../data/models/insert.pt'
    input_file = input_dir + 'input_insert_ast.json'
    output_file = output_dir + 'insert.txt'
    generate_knod(model_file, input_file, identifier_file, class_name_list, devices, beam_size, output_file)


def rerank_ast_patches(meta_file, hypo_file_list, fix_type_list, output_file, dump=True):
    # step 2: rerank the AST of patches
    reranked = rerank_patches_by_rank(meta_file, hypo_file_list, fix_type_list, dump=dump, output_file=output_file)


def validate_ast_patches(reranked_file, meta_file, identifiers_file, output_file, tmp_dir='/tmp/defects4j/'):
    # step 3: convert AST of patches to source code patches and run test cases, will take a long time
    validate_defects4j(
        hypo_path=reranked_file,
        meta_path=meta_file,
        identifiers_path=identifiers_file,
        output_path=output_file, 
        tmp_dir=tmp_dir, progress_range=None
    )


if __name__ == '__main__':
    devices = [0, 0]    # require two devices, but could be the same one
    beam_size = 200
    
    model_general, model_insert = sys.argv[1], sys.argv[2]
    
    generate_general_ast_patches(
        SRC_DIR + '../data/defects4j_input/', 
        SRC_DIR + '../data/defects4j_output/', 
        model_general
    )
    generate_insert_ast_patches(
        SRC_DIR + '../data/defects4j_input/', 
        SRC_DIR + '../data/defects4j_output/', 
        model_insert
    )
    
    rerank_ast_patches(
        SRC_DIR + '../data/defects4j_input/meta_localize.txt',
        [SRC_DIR + '../data/defects4j_output/general.txt', SRC_DIR + '../data/defects4j_output/insert.txt'],
        ['general', 'insert'],
        SRC_DIR + '../data/defects4j_output/reranked.json', True
    )
    validate_ast_patches(
        SRC_DIR + '../data/defects4j_output/reranked.json',
        SRC_DIR + '../data/defects4j_input/meta_localize.txt',
        SRC_DIR + '../data/defects4j_input/identifiers.json',
        SRC_DIR + '../data/defects4j_output/validation.json'
    )
