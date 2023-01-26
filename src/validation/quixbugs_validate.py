import codecs
import json
import os
import shutil
import time

import quixbugs_command
from reconstruct import reconstruct_ctx_wrap_ast, combine_super_methods, reconstruct_wrap_ast
from reconstruct import reconstruct_patched_ctx_wrap_ast, semantic_reconstruct, remove_patch_wrap_ast, reconstruct_ast

VALIDATION_DIR = os.path.abspath(__file__)
VALIDATION_DIR = VALIDATION_DIR[: VALIDATION_DIR.rfind('/') + 1]


def read_rem(rem_path):
    fp = codecs.open(rem_path, 'r', 'utf-8')
    rem = []
    for line in fp.readlines():
        if line.strip() == '':
            rem.append([None, None, None, None, None])
        else:
            loc, index, rem_line = line.split('\t')
            start, end = loc.split('-')
            start_row, start_col = start.split(',')
            end_row, end_col = end.split(',')
            start_row, start_col = int(start_row), int(start_col)
            end_row, end_col = int(end_row), int(end_col)
            rem.append([
                start_row, start_col, end_row, end_col, rem_line.strip()
            ])
    fp.close()
    return rem

def read_meta(meta_path):
    fp = codecs.open(meta_path, 'r', 'utf-8')
    meta = []
    for line in fp.readlines():
        if len(line.split()) == 6:
            proj, rem_start, rem_end, add_start, add_end, tag = line.strip().split()
            meta.append([proj, rem_start, rem_end, tag])
        elif len(line.split()) == 4:
            proj, rem_start, rem_end, tag = line.strip().split()
            meta.append([proj, rem_start, rem_end, tag])
        elif len(line.split()) == 3:
            proj, rem_start, rem_end = line.strip().split()
            meta.append([proj, rem_start, rem_end])
    fp.close()
    return meta

def insert_fix_quixbugs(file_path, start_row, start_col, end_row, end_col, patch, fix_type='general'):
    with open(file_path, 'r') as file:
        data = file.readlines()
    file = codecs.open(file_path, 'w', 'utf-8')
    if fix_type == 'general':
        for i in range(start_row - 1):
            file.write(data[i])
        file.write(data[start_row - 1][: start_col - 1])
        file.write(patch)
        file.write(data[end_row - 1][end_col:])
        for i in range(end_row, len(data)):
            file.write(data[i])
    else:
        for i in range(start_row - 1):
            file.write(data[i])
        file.write(patch)
        file.write(data[end_row - 1])
        for i in range(end_row, len(data)):
            file.write(data[i])
    file.close()
    return file_path


def validate_quixbugs(hypo_path, meta_path, identifiers_path, output_path, tmp_dir, progress_range=None):
    identifiers_dict = json.load(open(identifiers_path, 'r'))
    hypo = json.load(open(hypo_path, 'r'))
    
    cnt, right = 0, 0

    if not os.path.exists(tmp_dir):
        quixbugs_command.command_with_timeout(['mkdir', tmp_dir])

    meta = read_meta(meta_path)
    bugs_to_validate = list(hypo.keys())
    rem_general = read_rem(VALIDATION_DIR + '../../data/quixbugs_input/rem_general_localize.txt')
    rem_insert = read_rem(VALIDATION_DIR + '../../data/quixbugs_input/rem_insert_localize.txt')
    quixbugs_general = json.load(open(VALIDATION_DIR + '../../data/quixbugs_input/input_general_ast.json', 'r'))
    quixbugs_general = {data['id']: data for data in quixbugs_general}
    quixbugs_insert = json.load(open(VALIDATION_DIR + '../../data/quixbugs_input/input_insert_ast.json', 'r'))
    quixbugs_insert = {data['id']: data for data in quixbugs_insert}

    validated_result = {}
    for line_num in bugs_to_validate:
        line_num = str(line_num)
        if progress_range is not None and (cnt < progress_range[0] or cnt >= progress_range[1]):
            cnt += 1
            continue

        cnt += 1

        proj, rem_start, rem_end, fix_type = meta[int(line_num) - 1]
        start_row_g, start_col_g, end_row_g, end_col_g, rem_line_g = rem_general[int(line_num) - 1]
        start_row_i, start_col_i, end_row_i, end_col_i, rem_line_i = rem_insert[int(line_num) - 1]

        print(right, '/', cnt, proj)
        quixbugs_command.command_with_timeout(['rm', '-rf', tmp_dir + '/java_programs/'])
        quixbugs_command.command_with_timeout(['mkdir', tmp_dir + '/java_programs/'])

        if int(line_num) in quixbugs_general:
            data = quixbugs_general[int(line_num)]
            src2abs_g = {k: v for k, v in data['mappings'].items() if '<UNK>' not in v}
            abs2src_g = {v: k for k, v in data['mappings'].items() if '<UNK>' not in v}
            ctx_ast_nodes_g, node_num_before_g, sibling_num_before_g = reconstruct_ctx_wrap_ast(
                data['nodes'], data['edges'], data['rem_roots'][0], abs2src_g
            )
        else:
            src2abs_g, abs2src_g = None, None
            ctx_ast_nodes_g, node_num_before_g, sibling_num_before_g = None, None, None
        if int(line_num) in quixbugs_insert:
            data = quixbugs_insert[int(line_num)]
            src2abs_i = {k: v for k, v in data['mappings'].items() if '<UNK>' not in v}
            abs2src_i = {v: k for k, v in data['mappings'].items() if '<UNK>' not in v}
            ctx_ast_nodes_i, node_num_before_i, sibling_num_before_i = reconstruct_ctx_wrap_ast(
                data['nodes'], data['edges'], data['rem_roots'][0], abs2src_i
            )
        else:
            src2abs_i, abs2src_i = None, None
            ctx_ast_nodes_i, node_num_before_i, sibling_num_before_i = None, None, None

        shutil.copyfile(tmp_dir + "/java_programs_bak/" + proj + '.java',
                        tmp_dir + "/java_programs/" + proj + '.java')
        shutil.copyfile(tmp_dir + "/java_programs_bak/Node.java", tmp_dir + "/java_programs/Node.java")
        shutil.copyfile(tmp_dir + "/java_programs_bak/WeightedEdge.java", tmp_dir + "/java_programs/WeightedEdge.java")
        if str(line_num) not in identifiers_dict:
            identifiers = {}
        else:
            identifiers = identifiers_dict[str(line_num)]
        print('identifiers num:', len(identifiers))

        validated_result[proj] = {'id': int(line_num), 'patches': []}
        start_this_bug = time.time()
        current_is_correct = False
        for rank, patch_ast in enumerate(hypo[line_num]['patches']):
            if time.time() - start_this_bug > 3600 * 5:
                break
            elif rank >= 2500:
                break
            if current_is_correct:
                break

            file_path = tmp_dir + "/java_programs/" + proj + '.java'
            try:
                patch_fix_type = patch_ast['fix_type']
                if patch_fix_type == 'general':
                    abs2src, src2abs = abs2src_g, src2abs_g
                    ctx_ast_nodes, node_num_before, sibling_num_before = \
                        ctx_ast_nodes_g, node_num_before_g, sibling_num_before_g
                    start_row, start_col, end_row, end_col = start_row_g, start_col_g, end_row_g, end_col_g
                else:
                    abs2src, src2abs = abs2src_i, src2abs_i
                    ctx_ast_nodes, node_num_before, sibling_num_before = \
                        ctx_ast_nodes_i, node_num_before_i, sibling_num_before_i
                    start_row, start_col, end_row, end_col = start_row_i, start_col_i, end_row_i, end_col_i

                if abs2src is None or ctx_ast_nodes is None:
                    continue
                if patch_ast['n'].strip() == '<EOS>':
                    code_patches = ['']
                elif '_<UNK>' in patch_ast['n']:
                    if identifiers is None:
                        continue
                    fathers, edges, nodes = patch_ast['f'].split(), patch_ast['e'].split(), patch_ast['n'].split()
                    patch_ast_roots, patch_ast_nodes = reconstruct_wrap_ast(fathers, edges, nodes, abs2src)
                    ast_nodes = reconstruct_patched_ctx_wrap_ast(ctx_ast_nodes, node_num_before, sibling_num_before,
                                                                 patch_ast_roots, patch_ast_nodes)
                    reconstructed_nodes = semantic_reconstruct(patch_ast_nodes, ast_nodes, proj + '.java', src2abs, abs2src,
                                                               identifiers)
                    if patch_fix_type == 'general':
                        ctx_ast_nodes_g = remove_patch_wrap_ast(ast_nodes, node_num_before, sibling_num_before,
                                                                patch_ast_roots, patch_ast_nodes)
                    else:
                        ctx_ast_nodes_i = remove_patch_wrap_ast(ast_nodes, node_num_before, sibling_num_before,
                                                                patch_ast_roots, patch_ast_nodes)
                    if reconstructed_nodes is None or reconstructed_nodes == []:
                        continue
                    reconstruct_max = 5
                    code_patches = [
                        reconstruct_ast(
                            patch_ast['f'].split(),
                            patch_ast['e'].split(),
                            reconstructed_node,
                            abs2src)
                        for reconstructed_node in reconstructed_nodes[: reconstruct_max]
                    ]
                else:
                    code_patches = [reconstruct_ast(
                        patch_ast['f'].split(),
                        patch_ast['e'].split(),
                        patch_ast['n'].split(),
                        abs2src
                    )]
                score = patch_ast['score']
                for patch in code_patches:
                    if current_is_correct:
                        break
                    s_time = time.time()
                    patch = patch.strip()
                    patched_file = insert_fix_quixbugs(file_path, start_row, start_col, end_row, end_col,
                                                       patch, fix_type=patch_fix_type)
                    compile = quixbugs_command.compile_fix(patched_file, tmp_dir + "/java_programs/")
                    correctness = 'uncompilable'
                    if compile:
                        correctness = quixbugs_command.quixbugs_test_suite(proj, quixbugs_dir=tmp_dir)
                        if correctness == 'plausible':
                            right += 1
                            print(right, cnt, rank, "Correct patch:", patch, str(int(time.time() - s_time)) + 's')
                            current_is_correct = True
                        elif correctness == 'wrong':
                            print(right, cnt, rank, "Wrong patch:", patch, str(int(time.time() - s_time)) + 's')
                        elif correctness == 'timeout':
                            print(right, cnt, rank, "Timeout patch:", patch, str(int(time.time() - s_time)) + 's')
                    else:
                        print(right, cnt, rank, 'Uncompilable patch:', patch, str(int(time.time() - s_time)) + 's')
                    if correctness == 'plausible':
                        validated_result[proj]['patches'].append({
                            'patch': patch, 'score': score, 'correctness': correctness, 'fix_type': patch_fix_type
                        })
                    else:
                        validated_result[proj]['patches'].append({
                            'patch': patch, 'score': score, 'correctness': correctness, 'fix_type': patch_fix_type
                        })
                    shutil.copyfile(tmp_dir + "/java_programs_bak/" + proj + '.java',
                                    tmp_dir + "/java_programs/" + proj + '.java')
            except Exception as e:
                print(e)
                shutil.copyfile(tmp_dir + "/java_programs_bak/" + proj + '.java',
                                tmp_dir + "/java_programs/" + proj + '.java')
                if int(line_num) in quixbugs_general:
                    data = quixbugs_general[int(line_num)]
                    src2abs_g = {k: v for k, v in data['mappings'].items() if '<UNK>' not in v}
                    abs2src_g = {v: k for k, v in data['mappings'].items() if '<UNK>' not in v}
                    ctx_ast_nodes_g, node_num_before_g, sibling_num_before_g = reconstruct_ctx_wrap_ast(
                        data['nodes'], data['edges'], data['rem_roots'][0], abs2src_g
                    )
                else:
                    src2abs_g, abs2src_g = None, None
                    ctx_ast_nodes_g, node_num_before_g, sibling_num_before_g = None, None, None
                if int(line_num) in quixbugs_insert:
                    data = quixbugs_insert[int(line_num)]
                    src2abs_i = {k: v for k, v in data['mappings'].items() if '<UNK>' not in v}
                    abs2src_i = {v: k for k, v in data['mappings'].items() if '<UNK>' not in v}
                    ctx_ast_nodes_i, node_num_before_i, sibling_num_before_i = reconstruct_ctx_wrap_ast(
                        data['nodes'], data['edges'], data['rem_roots'][0], abs2src_i
                    )
                else:
                    src2abs_i, abs2src_i = None, None
                    ctx_ast_nodes_i, node_num_before_i, sibling_num_before_i = None, None, None
        # write after finish validating every bug, to avoid wasting time
        json.dump(validated_result, open(output_path, 'w'), indent=2)
    # write the last time after validating all
    json.dump(validated_result, open(output_path, 'w'), indent=2)


if __name__ == '__main__':
    validate_quixbugs(
        hypo_path=VALIDATION_DIR + '../../data/quixbugs_output/reranked.json',
        meta_path=VALIDATION_DIR + '../../data/quixbugs_input/meta_localize.txt',
        identifiers_path=VALIDATION_DIR + '../../data/quixbugs_input/identifiers.json',
        output_path=VALIDATION_DIR + '../../data/quixbugs_output/validated.json', 
        tmp_dir=VALIDATION_DIR + '../../QuixBugs/', progress_range=None
    )

