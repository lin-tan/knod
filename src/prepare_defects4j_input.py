import os
import codecs
import subprocess
import json
import re
import sys

SRC_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = SRC_DIR + '../javaparser/'

sys.path.append(SRC_DIR + 'parser/')
sys.path.append(SRC_DIR + '../')
sys.path.append(SRC_DIR + 'validation/')

from ast_parser import dfs
import defects4j_command
from reconstruct import extract_identifiers, combine_super_methods
from defects4j_validate import read_meta
import javalang.tokenizer
import javalang.parse
from javalang.ast import Node
from javalang.tree import BlockStatement, SwitchStatementCase


def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if err != b'':
        print(err)
    print(' '.join(cmd))
    if output != b'':
        print(output)
    return output, err


def prepare_localize_data(data_dir):
    def insert_pad_statement(rem_file, insert_loc):
        fp = codecs.open(rem_file, 'r', 'utf-8')
        rems = fp.readlines()
        fp.close()
        wp = codecs.open(rem_file, 'w', 'utf-8')
        for line in rems[: insert_loc]:
            wp.write(line)
        wp.write('PAD_STATEMENT;\n')
        for line in rems[insert_loc:]:
            wp.write(line)
        wp.close()

    tmp_dir = '/tmp/'

    meta_fp = codecs.open(data_dir + '/meta.txt', 'r', 'utf-8')
    rem_general_wp = codecs.open(data_dir + '/rem_general_localize.txt', 'w', 'utf-8')
    # add_general_wp = codecs.open(data_dir + '/add_general_localize.txt', 'w', 'utf-8')
    ctx_general_wp = codecs.open(data_dir + '/ctx_general_localize.txt', 'w', 'utf-8')
    rem_insert_wp = codecs.open(data_dir + '/rem_insert_localize.txt', 'w', 'utf-8')
    # add_insert_wp = codecs.open(data_dir + '/add_insert_localize.txt', 'w', 'utf-8')
    ctx_insert_wp = codecs.open(data_dir + '/ctx_insert_localize.txt', 'w', 'utf-8')
    meta_wp = codecs.open(data_dir + '/meta_localize.txt', 'w', 'utf-8')

    discard_cnt, success_cnt = 0, 0
    for line in meta_fp.readlines():
        proj, bug_id, path, rem_start, rem_end, add_start, add_end = line.strip().split()
        print('\n' + proj, bug_id)
        
        try:
            if os.path.exists(tmp_dir + proj + '_b'):
                command(['rm', '-rf', tmp_dir + proj + '_b'])
            command(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir + proj + '_b'])
            assert os.path.exists(tmp_dir + proj + '_b/' + path)
            if os.path.exists(tmp_dir + proj + '_f'):
                command(['rm', '-rf', tmp_dir + proj + '_f'])
            command(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'f', '-w', tmp_dir + proj + '_f'])
            assert os.path.exists(tmp_dir + proj + '_f/' + path)

            bugs = [{'rem_loc': (int(rem_start), int(rem_end)), 'add_loc': (int(add_start), int(add_end))}]
            print(bugs)
        except Exception as e:
            discard_cnt += 1
            print(e)
            continue

        general_rem, general_ctx = '', ''
        insert_rem, insert_ctx = '', ''
        meta_line, tag = '', ''
        other = 'fail'

        bug = bugs[0]
        if bug['rem_loc'][0] == bug['rem_loc'][1]:
            tag = 'insert'
            # bug['rem_loc'] = (bug['rem_loc'][0] + 1, bug['rem_loc'][0] + 1)
            print(len(codecs.open(tmp_dir + proj + '_b/' + path, 'r', 'utf-8').readlines()))
            if codecs.open(tmp_dir + proj + '_b/' + path, 'r', 'utf-8').readlines()[bug['rem_loc'][0] - 1].strip():
                out, err = command(['java', '-cp', JAVA_DIR + ':' + JAVA_DIR + '/target:' + JAVA_DIR + '/lib/*',
                                    'jiang719.BuggyASTExtractor', 'defects4j', tmp_dir,
                                    tmp_dir + proj + '_b/' + path,
                                    str(bug['rem_loc'][0]), str(bug['rem_loc'][1] + 1)])
                try:
                    result = json.load(open(tmp_dir + 'tmp.json', 'r'))
                    rem_range = result['rem_range']
                    rem_index = result['rem_index']
                    rem_code = re.sub('\\s+', ' ', result['rem_code']).strip()
                    rem_context = re.sub('\\s+', ' ', result['rem_context']).strip()
                    if rem_context != '' and rem_code != '' and rem_range != '' and rem_index != '':
                        general_rem = rem_range + '\t' + rem_index + '\t' + rem_code
                        # general_add = ''
                        general_ctx = rem_context + '\t'
                        command(['rm', '-rf', tmp_dir + 'tmp.json', tmp_dir + 'tmp_add.java',
                                 tmp_dir + 'tmp_rem_.java', tmp_dir + 'tmp_add_.java'])
                        other = 'success'
                except Exception as e:
                    pass

            insert_pad_statement(
                tmp_dir + proj + '_b/' + path, bug['rem_loc'][0] - 1
            )
            out, err = command(['java', '-cp', JAVA_DIR + ':' + JAVA_DIR + '/target:' + JAVA_DIR + '/lib/*',
                                'jiang719.BuggyASTExtractor', 'training', tmp_dir,
                                tmp_dir + proj + '_b/' + path, # tmp_dir + proj + '_f/' + path,
                                str(bug['rem_loc'][0]), str(bug['rem_loc'][1] + 1)])
            try:
                result = json.load(open(tmp_dir + 'tmp.json', 'r'))
            except Exception as e:
                discard_cnt += 1
                print('no tmp.json')
                continue
            rem_range = result['rem_range']
            rem_index = result['rem_index']
            rem_code = re.sub('\\s+', ' ', result['rem_code']).strip()
            rem_context = re.sub('\\s+', ' ', result['rem_context']).strip()
            # add_indexs = result['add_index']
            # add_codes = result['add_code']
            # add_context = re.sub('\\s+', ' ', result['add_context']).strip()

            if rem_context == '' or rem_code == '' or rem_range == '' or rem_index == '':
                discard_cnt += 1
                print('empty')
                continue

            insert_rem = rem_range + '\t' + rem_index + '\t' + rem_code
            # add_line = ''
            # for (add_index, add_code) in zip(add_indexs, add_codes):
            #     add_code = re.sub('\\s+', ' ', add_code).strip()
            #     add_line += add_index + '\t' + add_code + '\t'
            # insert_add = ''
            insert_ctx = rem_context # + '\t' + add_context
            meta_line = proj + '\t' + bug_id + '\t' + path + '\t' + \
                        str(bug['rem_loc'][0]) + '\t' + str(bug['rem_loc'][1] + 1) + '\t' + tag
                        # str(bug['add_loc'][0]) + '\t' + str(bug['add_loc'][1]) + '\t' + tag
            command(['rm', '-rf', tmp_dir + 'tmp.json', tmp_dir + 'tmp_add.java',
                     tmp_dir + 'tmp_rem_.java', tmp_dir + 'tmp_add_.java'])
            success_cnt += 1

        else:
            tag = 'general'
            out, err = command(['java', '-cp', JAVA_DIR + ':' + JAVA_DIR + '/target:' + JAVA_DIR + '/lib/*',
                                'jiang719.BuggyASTExtractor', 'defects4j', tmp_dir,
                                tmp_dir + proj + '_b/' + path, # tmp_dir + proj + '_f/' + path,
                                str(bug['rem_loc'][0]), str(bug['rem_loc'][1])])
            try:
                result = json.load(open(tmp_dir + 'tmp.json', 'r'))
            except Exception as e:
                discard_cnt += 1
                continue
            rem_range = result['rem_range']
            rem_index = result['rem_index']
            rem_code = re.sub('\\s+', ' ', result['rem_code']).strip()
            rem_context = re.sub('\\s+', ' ', result['rem_context']).strip()
            # add_indexs = result['add_index']
            # add_codes = result['add_code']
            # add_context = re.sub('\\s+', ' ', result['add_context']).strip()

            if rem_context == '' or rem_code == '' or rem_range == '' or rem_index == '':
                discard_cnt += 1
                continue

            general_rem = rem_range + '\t' + rem_index + '\t' + rem_code
            # add_line = ''
            # for (add_index, add_code) in zip(add_indexs, add_codes):
            #     add_code = re.sub('\\s+', ' ', add_code).strip()
            #     add_line += add_index + '\t' + add_code + '\t'
            # general_add = add_line[:-1]
            general_ctx = rem_context # + '\t' + add_context
            meta_line = proj + '\t' + bug_id + '\t' + path + '\t' + \
                        str(bug['rem_loc'][0]) + '\t' + str(bug['rem_loc'][1]) + '\t' + tag
                        # str(bug['add_loc'][0]) + '\t' + str(bug['add_loc'][1]) + '\t' + tag
            command(['rm', '-rf', tmp_dir + 'tmp.json', tmp_dir + 'tmp_add.java',
                     tmp_dir + 'tmp_rem_.java', tmp_dir + 'tmp_add_.java'])
            success_cnt += 1

            if bugs[0]['rem_loc'][1] - bugs[0]['rem_loc'][0] == 1:
                insert_pad_statement(
                    tmp_dir + proj + '_b/' + path, bug['rem_loc'][0] - 1
                )
                out, err = command(['java', '-cp', JAVA_DIR + ':' + JAVA_DIR + '/target:' + JAVA_DIR + '/lib/*',
                                    'jiang719.BuggyASTExtractor', 'defects4j', tmp_dir,
                                    tmp_dir + proj + '_b/' + path,
                                    str(bug['rem_loc'][0]), str(bug['rem_loc'][1])])
                try:
                    result = json.load(open(tmp_dir + 'tmp.json', 'r'))
                    rem_range = result['rem_range']
                    rem_index = result['rem_index']
                    rem_code = re.sub('\\s+', ' ', result['rem_code']).strip()
                    rem_context = re.sub('\\s+', ' ', result['rem_context']).strip()
                    if rem_context != '' and rem_code != '' and rem_range != '' and rem_index != '':
                        insert_rem = rem_range + '\t' + rem_index + '\t' + rem_code
                        # insert_add = ''
                        insert_ctx = rem_context + '\t'
                        command(['rm', '-rf', tmp_dir + 'tmp.json', tmp_dir + 'tmp_add.java',
                                 tmp_dir + 'tmp_rem_.java', tmp_dir + 'tmp_add_.java'])
                        other = 'success'
                except Exception as e:
                    pass

        rem_general_wp.write(general_rem + '\n')
        # add_general_wp.write(general_add + '\n')
        ctx_general_wp.write(general_ctx + '\n')
        rem_insert_wp.write(insert_rem + '\n')
        # add_insert_wp.write(insert_add + '\n')
        ctx_insert_wp.write(insert_ctx + '\n')
        meta_wp.write(meta_line + '\n')
        print('succeed', tag, other)
    print(discard_cnt, success_cnt)


def prepare_mapping_data(data_dir):
    out, err = command(['java', '-cp', JAVA_DIR + ':' + JAVA_DIR + '/target:' + JAVA_DIR + '/lib/*',
                        'jiang719.Abstractor', data_dir + 'ctx_general_localize.txt', data_dir + 'mapping_general_localize.txt'])
    out, err = command(['java', '-cp', JAVA_DIR + ':' + JAVA_DIR + '/target:' + JAVA_DIR + '/lib/*',
                        'jiang719.Abstractor', data_dir + 'ctx_insert_localize.txt', data_dir + 'mapping_insert_localize.txt'])


def get_ast_size(node):
    traverse, edges = dfs(node)
    return len(traverse)


def prepare_ast_data(data_dir, target_tag='general'):
    def remove_pad_statement_from_mapping(file_path):
        new_data = []
        data = json.load(open(file_path, 'r'))
        for i in range(len(data)):
            if 'PAD_STATEMENT' not in data[i]['mappings']:
                new_data.append(data[i])
                continue
            non_var = {k: v for k, v in data[i]['mappings'].items() if v[:3] != 'VAR' or 'UNK' in v}
            var = {k: int(v.split('_')[1]) for k, v in data[i]['mappings'].items() if v[:3] == 'VAR' and 'UNK' not in v}
            if 'PAD_STATEMENT' not in var:
                continue
            pad_index = var['PAD_STATEMENT']
            new_var = {}
            for k, v in var.items():
                if k == 'PAD_STATEMENT':
                    continue
                elif v < pad_index:
                    new_var[k] = 'VAR_' + str(v)
                elif v > pad_index:
                    new_var[k] = 'VAR_' + str(v - 1)
            non_var.update(new_var)
            data[i]['mappings'] = non_var
            new_data.append(data[i])
        json.dump(new_data, open(file_path, 'w'))
    
    rem_fp = codecs.open(data_dir + 'rem_' + target_tag + '_localize.txt', 'r', 'utf-8')
    # add_fp = codecs.open(data_dir + 'add_' + target_tag + '_localize.txt', 'r', 'utf-8')
    ctx_fp = codecs.open(data_dir + 'ctx_' + target_tag + '_localize.txt', 'r', 'utf-8')
    mapping_fp = codecs.open(data_dir + 'mapping_' + target_tag + '_localize.txt', 'r', 'utf-8')
    meta_fp = codecs.open(data_dir + 'meta_localize.txt', 'r', 'utf-8')

    cant_localize = 0
    cnt = 0
    input_ast = []
    for rem, ctx, mappings, meta in zip(rem_fp.readlines(), ctx_fp.readlines(), mapping_fp.readlines(), meta_fp.readlines()):
        proj, bug_id, path, rem_start, rem_end, tag = meta.strip().split()
        cnt += 1

        rem_ctx = ctx.strip()
        if rem_ctx == '':
            continue

        if tag == target_tag:
            rem = rem.split('\t')
            rem_ctx_split = []
            pre_index = 0
            loc, index, code = rem[0], int(rem[1]), rem[2]
            rem_match = {
                'location': loc,
                'index': index,
                'code': code.strip(),
                'match': None,
                'matched_cnt': 0
            }
            cur_index = [j for j in range(len(rem_ctx)) if rem_ctx.startswith(code.strip(), j)][index]
            rem_ctx_split.append(rem_ctx[pre_index: cur_index])
            pre_index = cur_index + len(code)
            rem_match['code'] = re.sub('\\s+|\\(|\\)|{|}|;|,', '', rem_match['code'])
            rem_ctx_split.append(rem_ctx[pre_index:])

            try:
                tokens = javalang.tokenizer.tokenize(rem_ctx)
                parser = javalang.parser.Parser(tokens)
                rem_ast = parser.parse_member_declaration()
            except Exception as e:
                print(cnt, "parse failed")
                continue
            traverse, rem_edges = dfs(rem_ast)
            nodes, rem_roots = [], []
            for _, node in enumerate(traverse):
                if isinstance(node, Node):
                    node_code = node.to_code()
                    nodes.append(node.__class__.__name__)
                elif type(node) in [list, set]:
                    node_code = ' '.join([s for s in list(node)])
                    nodes.append(node_code.strip())
                else:
                    node_code = str(node)
                    nodes.append(node_code.strip())

                node_code = re.sub('\\s+|\\(|\\)|{|}|;|,', '', node_code)

                is_rem_root = False
                if rem_match['code'] == node_code and \
                        not (isinstance(node, BlockStatement) and len(getattr(node, "statements")) == 1):
                    if rem_match['matched_cnt'] == rem_match['index']:
                        rem_match['match'] = node
                        is_rem_root = True
                    rem_match['matched_cnt'] += 1
                if is_rem_root:
                    rem_roots.append(_)
            rem_nodes = [rem_match['match']] if rem_match['match'] is not None else []
            if len(rem_roots) != 1 or len(rem_nodes) != 1:
                cant_localize += 1
                print(proj, bug_id, cnt, "match rem failed", rem)
                continue

            mapping_dict = {}
            mapping_rem, mapping_add = mappings.split('\t')
            mappings = mapping_rem.split(' <SEP> ')[:-1]

            nodes_set = set(nodes)
            temp = {k: [] for k in ('VAR', 'TYPE', 'METHOD', 'INT', 'STRING', 'FLOAT', 'CHAR')}
            for mapping in mappings:
                name, abstraction = mapping.strip().split('<MAP>')
                abs, _ = abstraction.split('_')
                if name in nodes_set:
                    temp[abs].append(name)
            for k, v in temp.items():
                for i, name in enumerate(v):
                    mapping_dict[name] = k + '_' + str(i + 1)

            mappings = mapping_add.split(' <SEP> ')[:-1]
            for mapping in mappings:
                name, abstraction = mapping.strip().split('<MAP>')
                if name not in mapping_dict:
                    abs, _ = abstraction.split('_')
                    mapping_dict[name] = abs + '_<UNK>'

            input_ast.append({
                'id': cnt,
                'mappings': mapping_dict,
                'nodes': nodes,
                'edges': rem_edges,
                'rem_roots': rem_roots,
                'add_roots': []
            })
        else:
            rem = rem.split('\t')
            rem_ctx_split = []
            pre_index = 0
            loc, index, code = rem[0], int(rem[1]), rem[2]
            rem_match = {
                'location': loc,
                'index': index,
                'code': code.strip(),
                'match': None,
                'matched_cnt': 0
            }
            cur_index = [j for j in range(len(rem_ctx)) if rem_ctx.startswith(code.strip(), j)][index]
            rem_ctx_split.append(rem_ctx[pre_index: cur_index])
            pre_index = cur_index + len(code)
            rem_match['code'] = re.sub('\\s+|\\(|\\)|{|}|;|,', '', rem_match['code'])
            rem_ctx_split.append(rem_ctx[pre_index:])

            try:
                tokens = javalang.tokenizer.tokenize(rem_ctx)
                parser = javalang.parser.Parser(tokens)
                rem_ast = parser.parse_member_declaration()
            except Exception as e:
                print(cnt, "parse failed")
                continue
            traverse, rem_edges = dfs(rem_ast)
            nodes, rem_roots = [], []
            for _, node in enumerate(traverse):
                if isinstance(node, Node):
                    node_code = node.to_code()
                    nodes.append(node.__class__.__name__)
                elif type(node) in [list, set]:
                    node_code = ' '.join([s for s in list(node)])
                    nodes.append(node_code.strip())
                else:
                    node_code = str(node)
                    nodes.append(node_code.strip())

                node_code = re.sub('\\s+|\\(|\\)|{|}|;|,', '', node_code)

                is_rem_root = False
                if rem_match['code'] == node_code and \
                        not (isinstance(node, BlockStatement) and len(getattr(node, "statements")) == 1):
                    if rem_match['matched_cnt'] == rem_match['index']:
                        rem_match['match'] = node
                        is_rem_root = True
                    rem_match['matched_cnt'] += 1
                if is_rem_root:
                    rem_roots.append(_)
            rem_nodes = [rem_match['match']] if rem_match['match'] is not None else []
            if len(rem_roots) != 1 or len(rem_nodes) != 1:
                cant_localize += 1
                print(cnt, "match rem failed", rem)
                continue

            mapping_dict = {}
            mapping_rem, mapping_add = mappings.split('\t')
            mappings = mapping_rem.split(' <SEP> ')[:-1]

            nodes_set = set(nodes)
            temp = {k: [] for k in ('VAR', 'TYPE', 'METHOD', 'INT', 'STRING', 'FLOAT', 'CHAR')}
            for mapping in mappings:
                name, abstraction = mapping.strip().split('<MAP>')
                abs, _ = abstraction.split('_')
                if name in nodes_set:
                    temp[abs].append(name)
            for k, v in temp.items():
                for i, name in enumerate(v):
                    mapping_dict[name] = k + '_' + str(i + 1)

            mappings = mapping_add.split(' <SEP> ')[:-1]
            for mapping in mappings:
                name, abstraction = mapping.strip().split('<MAP>')
                if name not in mapping_dict:
                    abs, _ = abstraction.split('_')
                    mapping_dict[name] = abs + '_<UNK>'

            input_ast.append({
                'id': cnt,
                'mappings': mapping_dict,
                'nodes': nodes,
                'edges': rem_edges,
                'rem_roots': rem_roots,
                'add_roots': [],
            })
    json.dump(input_ast, open(data_dir + 'input_' + target_tag + '_ast.json', 'w'))

    print("can't localize", cant_localize)
    print("localize succeed", len(input_ast))
    
    if target_tag == 'insert':
        remove_pad_statement_from_mapping(data_dir + 'input_insert_ast.json')


def prepare_identifier_data(meta_file, output_file, tmp_dir):
    meta = read_meta(meta_file)
    identifiers_list = {}
    current_bug = None
    for i, meta_line in enumerate(meta):
        if len(meta_line) == 6:
            proj, bug_id, file_path, rem_start, rem_end, tag = meta_line
        elif len(meta_line) == 5:
            proj, bug_id, file_path, rem_start, rem_end = meta_line
        if proj + '_' + bug_id != current_bug:
            current_bug = proj + '_' + bug_id
            print(proj, bug_id)
            defects4j_command.clean_tmp_folder(tmp_dir)
            defects4j_command.checkout_defects4j_project(proj, bug_id + 'b', tmp_dir)
        extract_identifiers(tmp_dir, file_path, rem_start, rem_end,
                            SRC_DIR + '../data/jdk.json')
        if os.path.exists(tmp_dir + 'identifiers.json'):
            identifiers = json.load(open(tmp_dir + 'identifiers.json', 'r'))
        else:
            identifiers = {}
        identifiers = combine_super_methods(identifiers)
        print('identifiers num:', len(identifiers))
        identifiers_list[i+1] = identifiers
    json.dump(identifiers_list, open(output_file, 'w'))
    

if __name__ == '__main__':
    prepare_localize_data(data_dir=SRC_DIR + '../data/defects4j_input/')
    prepare_mapping_data(data_dir=SRC_DIR + '../data/defects4j_input/')
    for tag in ('general', 'insert'):
        prepare_ast_data(data_dir=SRC_DIR + '../data/defects4j_input/', target_tag=tag)
    prepare_identifier_data(
        meta_file=SRC_DIR + '../data/defects4j_input/meta_localize.txt',
        output_file=SRC_DIR + '../data/defects4j_input/identifiers.json',
        tmp_dir='/tmp/d4j/'
    )
    
