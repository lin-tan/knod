import codecs
import json
import os
import shutil
import sys
import time

import defects4j_command
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
        if len(line.strip().split()) == 8:
            proj, bug_id, path, rem_start, rem_end, add_start, add_end, tag = line.strip().split()
            meta.append([proj, bug_id, path, rem_start, rem_end, tag])
        elif len(line.strip().split()) == 6:
            proj, bug_id, path, rem_start, rem_end, tag = line.strip().split()
            meta.append([proj, bug_id, path, rem_start, rem_end, tag])
        elif len(line.strip().split()) == 5:
            proj, bug_id, path, line, index = line.strip().split()
            meta.append([proj, bug_id, path, line, str(int(line) + 1), 'general'])
        elif len(line.strip().split()) == 4:
            proj, bug_id, path, loc = line.strip().split()
            if ',' in loc:
                rem_start, rem_len = loc.split(',')
                rem_end = str(int(rem_start) + int(rem_len))
            else:
                rem_start = loc
                rem_end = str(int(rem_start) + 1)
            meta.append([proj, bug_id, path, rem_start, rem_end, 'general'])
    fp.close()
    return meta


def insert_fix_defects4j(file_path, start_row, start_col, end_row, end_col, patch, project_dir, fix_type='general',
                         key=None):
    file_path = project_dir + file_path
    shutil.copyfile(file_path, file_path + '.bak')

    with open(file_path, 'r') as file:
        data = file.readlines()

    if fix_type == 'general':
        with open(file_path, 'w') as file:
            for i in range(start_row - 1):
                file.write(data[i])
            file.write(data[start_row - 1][: start_col - 1] + '\n')
            # if key in ['Closure_114', 'Mockito_8']:
            #    patch = '} else if (' + patch + ') {\n'
            file.write(patch)
            file.write(data[end_row - 1][end_col:])
            for i in range(end_row, len(data)):
                file.write(data[i])
    else:
        with open(file_path, 'w') as file:
            for i in range(start_row - 1):
                file.write(data[i])
            file.write(patch)
            file.write(data[end_row - 1])
            for i in range(end_row, len(data)):
                file.write(data[i])

    return file_path + '.bak'


# cnt, right = 0, 0

def validate_defects4j(hypo_path, meta_path, identifiers_path, output_path, tmp_dir, progress_range=None):
    # global cnt, right
    identifiers_dict = json.load(open(identifiers_path, 'r'))
    hypo = json.load(open(hypo_path, 'r'))
    
    cnt, right = 0, 0

    if not os.path.exists(tmp_dir):
        defects4j_command.command_with_timeout(['mkdir', tmp_dir])

    meta = read_meta(meta_path)
    bugs_to_validate = list(hypo.keys())
    rem_general = read_rem(VALIDATION_DIR + '../../data/defects4j_input/rem_general_localize.txt')
    rem_insert = read_rem(VALIDATION_DIR + '../../data/defects4j_input/rem_insert_localize.txt')
    defects4j_general = json.load(open(VALIDATION_DIR + '../../data/defects4j_input/input_general_ast.json', 'r'))
    defects4j_general = {data['id']: data for data in defects4j_general}
    defects4j_insert = json.load(open(VALIDATION_DIR + '../../data/defects4j_input/input_insert_ast.json', 'r'))
    defects4j_insert = {data['id']: data for data in defects4j_insert}

    validated_result = {}
    test_cases_num = []
    for line_num in bugs_to_validate:
        if progress_range is not None and (cnt < progress_range[0] or cnt >= progress_range[1]):
            cnt += 1
            continue

        cnt += 1
        proj, bug_id, file_path, rem_start, rem_end, fix_type = meta[int(line_num) - 1]

        if proj + '_' + bug_id + '_' + file_path in validated_result:
            continue

        start_row_g, start_col_g, end_row_g, end_col_g, rem_line_g = rem_general[int(line_num) - 1]
        start_row_i, start_col_i, end_row_i, end_col_i, rem_line_i = rem_insert[int(line_num) - 1]

        print(right, '/', cnt, proj, bug_id)

        defects4j_command.clean_tmp_folder(tmp_dir)
        defects4j_command.checkout_defects4j_project(proj, bug_id + 'b', tmp_dir)
        if proj == "Mockito":
            print("Mockito needs separate compilation")
            defects4j_command.compile_fix(tmp_dir)

        start_time = time.time()
        init_out, init_err = defects4j_command.defects4j_test_suite(tmp_dir)
        standard_time = time.time() - start_time

        failed_test_cases = str(init_out).split(' - ')[1:]
        for i, failed_test_case in enumerate(failed_test_cases):
            failed_test_cases[i] = failed_test_case.strip()
        init_fail_num = len(failed_test_cases)
        print(init_fail_num, str(standard_time) + 's')

        trigger, err = defects4j_command.defects4j_trigger(tmp_dir)
        triggers = trigger.strip().split('\n')
        for i, trigger in enumerate(triggers):
            triggers[i] = trigger.strip()
        print('trigger number:', len(triggers))

        relevant, err = defects4j_command.defects4j_relevant(tmp_dir)
        relevants = relevant.strip().split('\n')
        for i, relevant in enumerate(relevants):
            relevants[i] = relevant.strip()
        print('relevant number:', len(relevants))

        if int(line_num) in defects4j_general:
            data = defects4j_general[int(line_num)]
            src2abs_g = {k: v for k, v in data['mappings'].items() if '<UNK>' not in v}
            abs2src_g = {v: k for k, v in data['mappings'].items() if '<UNK>' not in v}
            ctx_ast_nodes_g, node_num_before_g, sibling_num_before_g = reconstruct_ctx_wrap_ast(
                data['nodes'], data['edges'], data['rem_roots'][0], abs2src_g
            )
        else:
            src2abs_g, abs2src_g = None, None
            ctx_ast_nodes_g, node_num_before_g, sibling_num_before_g = None, None, None
        if int(line_num) in defects4j_insert:
            data = defects4j_insert[int(line_num)]
            src2abs_i = {k: v for k, v in data['mappings'].items() if '<UNK>' not in v}
            abs2src_i = {v: k for k, v in data['mappings'].items() if '<UNK>' not in v}
            ctx_ast_nodes_i, node_num_before_i, sibling_num_before_i = reconstruct_ctx_wrap_ast(
                data['nodes'], data['edges'], data['rem_roots'][0], abs2src_i
            )
        else:
            src2abs_i, abs2src_i = None, None
            ctx_ast_nodes_i, node_num_before_i, sibling_num_before_i = None, None, None

        defects4j_command.clean_tmp_folder(tmp_dir)
        defects4j_command.checkout_defects4j_project(proj, bug_id + 'b', tmp_dir)
        if str(line_num) not in identifiers_dict:
            identifiers = {}
        else:
            identifiers = identifiers_dict[str(line_num)]
        print('identifiers num:', len(identifiers))

        key = proj + '_' + bug_id + '_' + file_path
        validated_result[key] = {'id': int(line_num), 'trigger num': len(triggers),
                                 'relevant num': len(relevants), 'patches': []}
        current_is_plausible = False
        current_is_correct = False
        failed_num = None
        start_this_bug = time.time()
        for rank, patch_ast in enumerate(hypo[line_num]['patches']):
            if time.time() - start_this_bug > 3600 * 5:
                break
            if len(validated_result[key]['patches']) > 2000:
                break
            # if rank >= 2000:
            #     break
            # if current_is_correct:
            #     break
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
                    reconstructed_nodes = semantic_reconstruct(patch_ast_nodes, ast_nodes, file_path, src2abs, abs2src,
                                                               identifiers)
                    if patch_fix_type == 'general':
                        ctx_ast_nodes_g = remove_patch_wrap_ast(ast_nodes, node_num_before, sibling_num_before,
                                                                patch_ast_roots, patch_ast_nodes)
                    else:
                        ctx_ast_nodes_i = remove_patch_wrap_ast(ast_nodes, node_num_before, sibling_num_before,
                                                                patch_ast_roots, patch_ast_nodes)
                    if reconstructed_nodes is None or reconstructed_nodes == []:
                        continue
                    reconstruct_max = 10
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
                    # if current_is_correct:
                    #     break
                    patch = patch.strip()
                    patched_file = insert_fix_defects4j(file_path, start_row, start_col, end_row, end_col,
                                                        patch, tmp_dir, fix_type=patch_fix_type, key=proj + '_' + bug_id)

                    if proj == 'Mockito':
                        # Mockito needs separate compile
                        defects4j_command.compile_fix(tmp_dir)

                    outs = []
                    correctness = None
                    start_time = time.time()
                    if standard_time >= 10 and len(triggers) <= 5 and len(relevants) >= 10:
                        for i, trigger in enumerate(triggers):
                            out, err = defects4j_command.defects4j_test_one(tmp_dir, trigger, timeout=min(2*standard_time, 200))
                            if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                                test_cases_num.append(i + 1)
                                print(patch_ast['n'])
                                print(right, '/', cnt, line_num, proj + '_' + bug_id, rank, 'Time out for patch: ', patch,
                                      str(int(time.time() - start_time)) + 's',
                                      int(sum(test_cases_num)/len(test_cases_num)))
                                correctness = 'timeout'
                                break
                            elif 'FAIL' in str(err) or 'FAIL' in str(out):
                                test_cases_num.append(i + 1)
                                print(patch_ast['n'])
                                print(right, '/', cnt, line_num, proj + '_' + bug_id, rank, 'Uncompilable patch:', patch,
                                      str(int(time.time() - start_time)) + 's',
                                      int(sum(test_cases_num)/len(test_cases_num)))
                                correctness = 'uncompilable'
                                break
                            elif "Failing tests: 0" in str(out):
                                continue
                            else:
                                outs += str(out).split(' - ')[1:]
                    if len(set(outs)) >= len(triggers):
                        # does not pass any one more
                        test_cases_num.append(len(triggers))
                        print(patch_ast['n'])
                        print(right, '/', cnt, line_num, proj + '_' + bug_id, rank, 'Wrong patch:', patch,
                              str(int(time.time() - start_time)) + 's',
                              int(sum(test_cases_num)/len(test_cases_num)))
                        correctness = 'wrong'

                    if correctness is None:
                        # pass at least one more trigger case
                        # have to pass all non-trigger
                        out, err = defects4j_command.defects4j_test_suite(tmp_dir, timeout=min(2*standard_time, 200))
                        test_cases_num.append(len(relevants))

                        if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                            print(patch_ast['n'])
                            print(right, '/', cnt, line_num,proj + '_' + bug_id, rank, 'Time out for patch: ', patch,
                                  str(int(time.time() - start_time)) + 's',
                                  int(sum(test_cases_num)/len(test_cases_num)))
                            correctness = 'timeout'
                        elif 'FAIL' in str(err) or 'FAIL' in str(out):
                            print(patch_ast['n'])
                            print(right, '/', cnt, line_num,proj + '_' + bug_id, rank, 'Uncompilable patch:', patch,
                                  str(int(time.time() - start_time)) + 's',
                                  int(sum(test_cases_num)/len(test_cases_num)))
                            correctness = 'uncompilable'
                        elif "Failing tests: 0" in str(out):
                            if not current_is_correct:
                                current_is_correct = True
                                right += 1
                            print(patch_ast['n'])
                            print(right, '/', cnt, line_num, proj + '_' + bug_id, rank, 'Plausible patch:', patch,
                                  str(int(time.time() - start_time)) + 's',
                                  int(sum(test_cases_num)/len(test_cases_num)))
                            correctness = 'plausible'
                            failed_num = 0
                        elif len(str(out).split(' - ')[1:]) < init_fail_num:
                            # fail less, could be correct
                            current_failed_test_cases = str(out).split(' - ')[1:]
                            no_new_fail = True
                            for current_failed_test_case in current_failed_test_cases:
                                if current_failed_test_case.strip() not in failed_test_cases:
                                    no_new_fail = False
                                    break
                            if no_new_fail:
                                # fail less and no new fail cases, could be plausible
                                if not current_is_plausible and not current_is_correct:
                                    current_is_plausible = True
                                    right += 1
                                print(patch_ast['n'])
                                print(right, '/', cnt, line_num, proj + '_' + bug_id, rank, 'Plausible patch:', patch,
                                      str(int(time.time() - start_time)) + 's',
                                      int(sum(test_cases_num)/len(test_cases_num)))
                                correctness = 'plausible'
                                failed_num = len(current_failed_test_cases)
                            else:
                                print(patch_ast['n'])
                                print(right, '/', cnt, line_num, proj + '_' + bug_id, rank, 'Wrong patch:', patch,
                                      str(int(time.time() - start_time)) + 's',
                                      int(sum(test_cases_num)/len(test_cases_num)))
                                correctness = 'wrong'
                        else:
                            print(patch_ast['n'])
                            print(right, '/', cnt, line_num, proj + '_' + bug_id, rank, 'Wrong patch:', patch,
                                  str(int(time.time() - start_time)) + 's',
                                  int(sum(test_cases_num)/len(test_cases_num)))
                            correctness = 'wrong'

                    if correctness == 'plausible':
                        validated_result[key]['patches'].append({
                            'patch': patch, 'score': score, 'correctness': correctness, 'failed': failed_num,
                            'fix_type': patch_fix_type
                        })
                    else:
                        validated_result[key]['patches'].append({
                            'patch': patch, 'score': score, 'correctness': correctness, 'fix_type': patch_fix_type
                        })
                    shutil.copyfile(patched_file, patched_file.replace('.bak', ''))
            except Exception as e:
                print(e)
                if os.path.exists(tmp_dir + file_path + '.bak'):
                    shutil.copyfile(tmp_dir + file_path + '.bak', tmp_dir + file_path)
                # re-construct the setting.
                if int(line_num) in defects4j_general:
                    data = defects4j_general[int(line_num)]
                    src2abs_g = {k: v for k, v in data['mappings'].items() if '<UNK>' not in v}
                    abs2src_g = {v: k for k, v in data['mappings'].items() if '<UNK>' not in v}
                    ctx_ast_nodes_g, node_num_before_g, sibling_num_before_g = reconstruct_ctx_wrap_ast(
                        data['nodes'], data['edges'], data['rem_roots'][0], abs2src_g
                    )
                else:
                    src2abs_g, abs2src_g = None, None
                    ctx_ast_nodes_g, node_num_before_g, sibling_num_before_g = None, None, None
                if int(line_num) in defects4j_insert:
                    data = defects4j_insert[int(line_num)]
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
    batch_id = int(sys.argv[1])
    
    validate_defects4j(
        hypo_path=VALIDATION_DIR + '../../data/defects4j_output/reranked.json',
        meta_path=VALIDATION_DIR + '../../data/defects4j_input/meta_localize.txt',
        identifiers_path=VALIDATION_DIR + '../../data/defects4j_input/identifiers.json',
        output_path=VALIDATION_DIR + '../../data/defects4j_output/validation_' + str(batch_id) + '.json', 
        tmp_dir='/tmp/defects4j_' + str(batch_id) + '/', progress_range=[20 * (batch_id - 1), 20 * batch_id]
    )
