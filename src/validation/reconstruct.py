import codecs
import subprocess
import sys
import os
import copy

VALIDATION_DIR = os.path.abspath(__file__)
VALIDATION_DIR = VALIDATION_DIR[: VALIDATION_DIR.rfind('/') + 1]

sys.path.append(VALIDATION_DIR + '../parser/')
sys.path.append(VALIDATION_DIR + '../../')

from ast_to_code import nonterminal_nodes
from javalang.ast import Node
from constrains import analyze_method, analyze_var, analyze_type


def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    return output, err


def extract_identifiers(proj, path, start, end, jdk_path):
    command(['java', '-cp', '/home/jiang719/java-workspace/deeprepair-javaparser:'
                            '/home/jiang719/java-workspace/deeprepair-javaparser/target:'
                            '/home/jiang719/java-workspace/deeprepair-javaparser/lib/*',
             'jiang719.ast_repair.IdentifierSemanticExtractor', proj, proj + '/' + path, str(start), str(end),
             jdk_path, proj + '/identifiers.json'])


def reconstruct_ast(fathers, edges, nodes, mapping):
    ast_nodes = []
    for father, edge, node in zip(fathers, edges, nodes):
        father = int(father)

        if father == 0 and (edge == '<EOS>' or node == '<EOS>'):
            break

        if father == 0:
            if node in nonterminal_nodes:
                ast_nodes.append(copy.deepcopy(nonterminal_nodes[node]))
            elif node in mapping:
                ast_nodes.append(mapping[node])
            else:
                ast_nodes.append(node)
        else:
            father_node = ast_nodes[father - 1]
            if node in nonterminal_nodes:
                child_node = copy.deepcopy(nonterminal_nodes[node])
            elif node in mapping:
                child_node = mapping[node]
            else:
                child_node = node
            if getattr(father_node, edge) is not None and getattr(father_node, edge) != [] and getattr(father_node, edge) != '':
                if isinstance(getattr(father_node, edge), Node) or type(getattr(father_node, edge)) == str:
                    setattr(father_node, edge, [getattr(father_node, edge)] + [child_node])
                elif type(getattr(father_node, edge)) == list:
                    setattr(father_node, edge, getattr(father_node, edge) + [child_node])
            else:
                setattr(father_node, edge, child_node)
            ast_nodes.append(child_node)

    patch = ''
    assert len(fathers) - 1 == len(ast_nodes)
    for i in range(len(ast_nodes)):
        if int(fathers[i]) == 0:
            if isinstance(ast_nodes[i], Node):
                patch += ast_nodes[i].to_code(show_plh=False)
            else:
                patch += ast_nodes[i]

    return patch


class WrappedNode:
    def __init__(self, node, father, edge):
        self.node = node
        self.father = father
        self.edge = edge
        self.children = {}

    def add_child(self, edge, child):
        if edge not in self.children:
            self.children[edge] = []
        self.children[edge].append(child)

    def get_name(self):
        if isinstance(self.node, Node):
            return type(self.node).__name__
        else:
            return self.node

    def dfs(self):
        traverse = [self]
        if len(self.children) > 0:
            for edge, children in self.children.items():
                for child in children:
                    traverse += child.dfs()
        return traverse


def type_satisfy(ty1, ty2, identifiers):
    # check if ty1 is super class of ty2
    if ty1 == ty2:
        return True
    if ty2 in identifiers:
        for identifier in identifiers[ty2]:
            if identifier['itype'] == "TYPE" and ty1 in identifier['supers']:
                return True
    return False


def combine_super_methods(identifiers):
    graph = {}
    for k, tys in identifiers.items():
        ty = tys[0]
        if ty['itype'] == 'TYPE':
            if k not in graph:
                graph[k] = {'subclass': [], 'in_degree': 0}
            for s in ty['supers']:
                if s not in graph:
                    graph[s] = {'subclass': [], 'in_degree': 0}
                graph[s]['subclass'].append(k)
                graph[k]['in_degree'] += 1

    def topological_sort(graph):
        out = []
        queue = []
        for k, v in graph.items():
            if v['in_degree'] == 0:
                queue.append(k)
        while len(queue) > 0:
            node = queue[0]
            out.append(node)
            queue = queue[1:]
            for sub in graph[node]['subclass']:
                graph[sub]['in_degree'] -= 1
                if graph[sub]['in_degree'] == 0:
                    queue.append(sub)
        return out

    out = topological_sort(graph)
    for name in out:
        if name not in identifiers or 'supers' not in identifiers[name][0]:
            continue
        for s in identifiers[name][0]['supers']:
            if s not in identifiers or 'methods' not in identifiers[s][0]:
                continue
            identifiers[name][0]['methods'] = list(set(identifiers[name][0]['methods'] + identifiers[s][0]['methods']))
    return identifiers


def read_hypo(hypo_path):
    fp = codecs.open(hypo_path, 'r', 'utf-8')
    hypo = {}
    for l in fp.readlines():
        l = l.split('\t')
        tag, flag, id = l[0].split('-')
        id = int(id)
        if id not in hypo:
            hypo[id] = {'id': id, 'target-f': '', 'target-e': '', 'target-n': '', 'patches': []}
        if tag == 'T':
            hypo[id]['target-' + flag] = l[1].strip()
        elif tag == 'S':
            continue
        elif tag == 'H':
            if flag == 'p':
                hypo[id]['patches'].append({
                    'score': float(l[1].strip()),
                    'f': '', 'e': '', 'n': ''
                })
            elif flag == 'f':
                hypo[id]['patches'][-1]['f'] = l[1].strip()
            elif flag == 'e':
                hypo[id]['patches'][-1]['e'] = l[1].strip()
            elif flag == 'n':
                hypo[id]['patches'][-1]['n'] = l[1].strip()
    fp.close()
    return hypo


def reconstruct_ctx_wrap_ast(nodes, edges, rem_root, mapping):
    ast_nodes = []
    node = nodes[0]

    rem_nodes = {rem_root}
    for (father, child, edge) in edges:
        if father in rem_nodes:
            rem_nodes.add(child)

    node_num_before, sibling_num_before = 0, 0
    remove = False
    ast_nodes.append(WrappedNode(copy.deepcopy(nonterminal_nodes[node]), None, None))
    for (father, child, edge) in edges:
        if not remove:
            father_node = ast_nodes[father]
        if child not in rem_nodes:
            if remove:
                if father > max(rem_nodes):
                    father -= len(rem_nodes)
            father_node = ast_nodes[father]
            node = nodes[child]
            if node in nonterminal_nodes:
                child_node = WrappedNode(copy.deepcopy(nonterminal_nodes[node]), father_node, edge)
            elif node in mapping:
                child_node = WrappedNode(mapping[node], father_node, edge)
            else:
                child_node = WrappedNode(node, father_node, edge)
            father_node.add_child(edge, child_node)
            ast_nodes.append(child_node)
        elif not remove:
            remove = True
            node_num_before = len(ast_nodes)
            sibling_num_before = len(father_node.children[edge]) if edge in father_node.children else 0

    return ast_nodes, node_num_before, sibling_num_before


def reconstruct_wrap_ast(fathers, edges, nodes, mapping):
    ast_roots, ast_nodes = [], []
    for father, edge, node in zip(fathers, edges, nodes):
        father = int(father)

        if father == 0 and (edge == '<EOS>' or node == '<EOS>'):
            break

        if father == 0:
            if node in nonterminal_nodes:
                ast_nodes.append(WrappedNode(copy.deepcopy(nonterminal_nodes[node]), None, edge))
            elif node in mapping:
                ast_nodes.append(WrappedNode(mapping[node], None, edge))
            else:
                ast_nodes.append(WrappedNode(node, None, edge))
            ast_roots.append(ast_nodes[-1])
        else:
            father_node = ast_nodes[father - 1]
            if node in nonterminal_nodes:
                child_node = WrappedNode(copy.deepcopy(nonterminal_nodes[node]), father_node, edge)
            elif node in mapping:
                child_node = WrappedNode(mapping[node], father_node, edge)
            else:
                child_node = WrappedNode(node, father_node, edge)
            father_node.add_child(edge, child_node)

            ast_nodes.append(child_node)

    return ast_roots, ast_nodes


def reconstruct_patched_ctx_wrap_ast(ast_nodes, node_num_before, sibling_num_before, patch_ast_roots, patch_ast_nodes):
    ast_nodes = ast_nodes[: node_num_before] + patch_ast_nodes + ast_nodes[node_num_before:]
    father_of_rem = ast_nodes[node_num_before - 1]
    for i, patch_ast_root in enumerate(patch_ast_roots):
        edge = patch_ast_root.edge
        patch_ast_root.father = father_of_rem
        if edge not in father_of_rem.children:
            father_of_rem.children[edge] = [patch_ast_root]
        else:
            father_of_rem.children[edge].insert(sibling_num_before + i, patch_ast_root)
    return ast_nodes


def remove_patch_wrap_ast(ast_nodes, node_num_before, sibling_num_before, patch_ast_roots, patch_ast_nodes):
    ast_nodes = ast_nodes[: node_num_before] + ast_nodes[node_num_before + len(patch_ast_nodes):]
    father_of_rem = ast_nodes[node_num_before - 1]
    if len(patch_ast_roots) > 0:
        edge = patch_ast_roots[0].edge
        father_of_rem.children[edge] = father_of_rem.children[edge][: sibling_num_before] + \
            father_of_rem.children[edge][sibling_num_before + len(patch_ast_roots):]
    return ast_nodes


def find_satisfy_type(ast_nodes, node, filepath, src2abs, abs2src, identifiers, constrains=None):
    if constrains is None:
        constrains = analyze_type(ast_nodes, node, abs2src, identifiers)
    candidates = {}
    for k, vs in identifiers.items():
        if k in src2abs:
            continue
        for v in vs:
            if v['itype'] == 'TYPE':
                if constrains['methods'] is not None and constrains['methods'] not in v['methods']:
                    continue
                candidates[k] = v['cnt']
                break
    candidates = sorted(candidates.items(), key=lambda e: e[1], reverse=True)
    return candidates


def find_satisfy_method(ast_nodes, node, filepath, src2abs, abs2src, identifiers, constrains=None):
    if constrains is None:
        constrains = analyze_method(ast_nodes, node, abs2src, identifiers)
    candidates = {}
    fileclass = filepath.split('/')[-1][:-5]
    for k, vs in identifiers.items():
        if k in src2abs:
            continue
        for v in vs:
            if v['itype'] == 'METHOD':
                satisfy = True
                if constrains['arguments'] is not None:
                    if len(constrains['arguments']) != len(v['params']):
                        continue
                    for arg, param in zip(constrains['arguments'], v['params']):
                        if arg is not None and (not type_satisfy(param, arg, identifiers)):
                            satisfy = False
                            break
                if not satisfy:
                    continue
                if (constrains['return_type'] is not None) and constrains['return_type'] != 'void' and \
                        (not type_satisfy(v['dtype'], constrains['return_type'], identifiers)):
                    continue
                if constrains['qualifier'] is None:
                    constrains['qualifier'] = fileclass
                if constrains['qualifier'] != 'TYPE_<UNK>' and constrains['qualifier'] != 'UNK' and \
                        (not type_satisfy(v['qualifier'], constrains['qualifier'], identifiers)):
                    continue
                if constrains['return_type_methods'] is not None:
                    return_type = v['dtype']
                    if '<' in return_type:
                        return_type = return_type[: return_type.find('<')]
                    if (return_type not in identifiers) or \
                            ('methods' not in identifiers[return_type][0] or
                             constrains['return_type_methods'] not in identifiers[return_type][0]['methods']):
                        continue
                if constrains['modifier'] is not None and constrains['modifier'] not in v['modifier']:
                    continue

                candidates[k] = v['cnt']
                break
    candidates = sorted(candidates.items(), key=lambda e: e[1], reverse=True)
    return candidates


def find_satisfy_var(ast_nodes, node, filepath, src2abs, abs2src, identifiers, constrains=None):
    if constrains is None:
        constrains = analyze_var(ast_nodes, node, abs2src, identifiers)
    candidates = {}
    fileclass = filepath.split('/')[-1][:-5]
    for k, vs in identifiers.items():
        if k in src2abs:
            continue
        for v in vs:
            if v['itype'] == 'VAR':
                if constrains['qualifier'] is None:
                    constrains['qualifier'] = fileclass
                if constrains['qualifier'] != 'TYPE_<UNK>' and \
                        (not type_satisfy(v['qualifier'], constrains['qualifier'], identifiers)):
                    continue
                if constrains['type'] is not None and (not type_satisfy(v['dtype'], constrains['type'], identifiers)):
                    continue

                candidates[k] = v['cnt']
                break
    candidates = sorted(candidates.items(), key=lambda e: e[1], reverse=True)
    return candidates


def semantic_reconstruct(patch_ast_nodes, ast_nodes, filepath, src2abs, abs2src, identifiers):
    MAX = 100
    reconstructed_nodes = []
    constrains = {}
    candidate = [None for _ in range(len(patch_ast_nodes))]
    for i, node in enumerate(patch_ast_nodes):
        if node.node == 'TYPE_<UNK>':
            constrains[i] = analyze_type(ast_nodes, node, abs2src, identifiers)
        elif node.node == 'METHOD_<UNK>':
            constrains[i] = analyze_method(ast_nodes, node, abs2src, identifiers)
        elif node.node == 'VAR_<UNK>':
            constrains[i] = analyze_var(ast_nodes, node, abs2src, identifiers)
        elif node.node == 'INT_<UNK>' or node.node == 'FLOAT_<UNK>':
            candidate[i] = "0"
        elif node.node == 'STRING_<UNK>':
            # candidate[i] = '""'
            constrains[i] = [('""', 1), ('"0"', 1), ('"\\0"', 1), ('"\\\\0"', 1), ('"\\\\000"', 1), ('">;"', 1), ('"th"', 1),  ('"null"', 1)]
        elif node.node == 'CHAR_<UNK>':
            # candidate[i] = "'.'"
            constrains[i] = [("'.'", 1), ("'\\0'", 1)]
        else:
            candidate[i] = type(node.node).__name__ if isinstance(node.node, Node) else str(node.node)
    print(constrains)
    if len(constrains) == 2:
        if list(constrains.values())[0] != list(constrains.values())[1]:
            constrains_num = []
            for i, c in constrains.items():
                num = 0
                if type(c) == dict:
                    for k, v in c.items():
                        if v:
                            num += 1
                constrains_num.append((i, num))
            constrains_num = sorted(constrains_num, key=lambda e: e[1], reverse=True)

            (i, n) = constrains_num[0]
            if patch_ast_nodes[i].node == 'TYPE_<UNK>':
                cand_1 = find_satisfy_type(ast_nodes, patch_ast_nodes[i],
                                          filepath, src2abs, abs2src, identifiers, constrains=constrains[i])
            elif patch_ast_nodes[i].node == 'METHOD_<UNK>':
                cand_1 = find_satisfy_method(ast_nodes, patch_ast_nodes[i],
                                             filepath, src2abs, abs2src, identifiers, constrains=constrains[i])
            elif patch_ast_nodes[i].node == 'VAR_<UNK>':
                cand_1 = find_satisfy_var(ast_nodes, patch_ast_nodes[i],
                                          filepath, src2abs, abs2src, identifiers, constrains=constrains[i])
            else:
                cand_1 = constrains[i]
            if not cand_1:
                return None
            cand_1 = cand_1[: MAX]
            cand_1_2 = []
            (j, n) = constrains_num[1]
            for _, (cand, cnt) in enumerate(cand_1):
                patch_ast_nodes[i].node = cand
                if patch_ast_nodes[j].node == 'TYPE_<UNK>':
                    cand_2 = find_satisfy_type(ast_nodes, patch_ast_nodes[j], filepath, src2abs, abs2src, identifiers)
                elif patch_ast_nodes[j].node == 'METHOD_<UNK>':
                    cand_2 = find_satisfy_method(ast_nodes, patch_ast_nodes[j], filepath, src2abs, abs2src, identifiers)
                elif patch_ast_nodes[j].node == 'VAR_<UNK>':
                    cand_2 = find_satisfy_var(ast_nodes, patch_ast_nodes[j], filepath, src2abs, abs2src, identifiers)
                else:
                    cand_2 = constrains[j]
                if not cand_2:
                    continue
                for k in range(len(cand_2)):
                    cand_1_2.append([cand, cand_2[k][0], cnt + cand_2[k][1]])
            cand_1_2 = sorted(cand_1_2, key=lambda e: e[2], reverse=True)[: MAX]
            if not cand_1_2:
                return None
            for (c1, c2, cnt) in cand_1_2:
                candidate[i] = c1
                candidate[j] = c2
                reconstructed_nodes.append([c for c in candidate])
        else:
            (i, c) = list(constrains.items())[0]
            (j, _) = list(constrains.items())[1]
            if patch_ast_nodes[i].node == 'TYPE_<UNK>':
                cand_1 = find_satisfy_type(ast_nodes, patch_ast_nodes[i],
                                           filepath, src2abs, abs2src, identifiers, constrains=c)
            elif patch_ast_nodes[i].node == 'METHOD_<UNK>':
                cand_1 = find_satisfy_method(ast_nodes, patch_ast_nodes[i],
                                             filepath, src2abs, abs2src, identifiers, constrains=c)
            elif patch_ast_nodes[i].node == 'VAR_<UNK>':
                cand_1 = find_satisfy_var(ast_nodes, patch_ast_nodes[i],
                                          filepath, src2abs, abs2src, identifiers, constrains=c)
            else:
                cand_1 = c
            cand_1 = sorted(cand_1, key=lambda e: e[1], reverse=True)[:MAX]
            if not cand_1:
                return None
            for (c1, cnt) in cand_1:
                candidate[i] = candidate[j] = c1
                reconstructed_nodes.append([c for c in candidate])
    elif len(constrains) == 1:
        (i, c) = list(constrains.items())[0]
        if patch_ast_nodes[i].node == 'TYPE_<UNK>':
            cand_1 = find_satisfy_type(ast_nodes, patch_ast_nodes[i],
                                      filepath, src2abs, abs2src, identifiers, constrains=c)
        elif patch_ast_nodes[i].node == 'METHOD_<UNK>':
            cand_1 = find_satisfy_method(ast_nodes, patch_ast_nodes[i],
                                         filepath, src2abs, abs2src, identifiers, constrains=c)
        elif patch_ast_nodes[i].node == 'VAR_<UNK>':
            cand_1 = find_satisfy_var(ast_nodes, patch_ast_nodes[i],
                                      filepath, src2abs, abs2src, identifiers, constrains=c)
        else:
            cand_1 = c
        cand_1 = sorted(cand_1, key=lambda e: e[1], reverse=True)[: MAX]
        if not cand_1:
            return None
        for (c1, cnt) in cand_1:
            candidate[i] = c1
            reconstructed_nodes.append([c for c in candidate])
    return reconstructed_nodes


def reconstruct_unk(fathers, edges, nodes, abs2src, src2abs,
                    ctx_ast_nodes, node_num_before, sibling_num_before,
                    file_path, identifiers):
    patch_ast_roots, patch_ast_nodes = reconstruct_wrap_ast(fathers, edges, nodes, abs2src)
    ast_nodes = reconstruct_patched_ctx_wrap_ast(ctx_ast_nodes, node_num_before, sibling_num_before,
                                                 patch_ast_roots, patch_ast_nodes)
    reconstructed_nodes = semantic_reconstruct(patch_ast_nodes, ast_nodes, file_path, src2abs, abs2src,
                                               identifiers)
    ctx_ast_nodes = remove_patch_wrap_ast(ast_nodes, node_num_before, sibling_num_before,
                                          patch_ast_roots, patch_ast_nodes)
    return reconstructed_nodes, ctx_ast_nodes


def reconstruct_ast_for_one_bug(ast_general_data, ast_insert_data, file_path, identifiers, patch_asts):
    src2abs_g = {k: v for k, v in ast_general_data['mappings'].items() if '<UNK>' not in v}
    abs2src_g = {v: k for k, v in ast_general_data['mappings'].items() if '<UNK>' not in v}
    ctx_ast_nodes_g, node_num_before_g, sibling_num_before_g = reconstruct_ctx_wrap_ast(
        ast_general_data['nodes'], ast_general_data['edges'], ast_general_data['rem_roots'][0], abs2src_g
    )
    src2abs_i = {k: v for k, v in ast_insert_data['mappings'].items() if '<UNK>' not in v}
    abs2src_i = {v: k for k, v in ast_insert_data['mappings'].items() if '<UNK>' not in v}
    ctx_ast_nodes_i, node_num_before_i, sibling_num_before_i = reconstruct_ctx_wrap_ast(
        ast_insert_data['nodes'], ast_insert_data['edges'], ast_insert_data['rem_roots'][0], abs2src_i
    )
    code_patches = []
    for patch_ast in patch_asts:
        patch_fix_type = patch_ast['fix_type']
        try:
            if patch_fix_type == 'general':
                abs2src, src2abs = abs2src_g, src2abs_g
                ctx_ast_nodes, node_num_before, sibling_num_before = \
                    ctx_ast_nodes_g, node_num_before_g, sibling_num_before_g
            else:
                abs2src, src2abs = abs2src_i, src2abs_i
                ctx_ast_nodes, node_num_before, sibling_num_before = \
                    ctx_ast_nodes_i, node_num_before_i, sibling_num_before_i
            if patch_ast['n'].strip() == '<EOS>':
                code_patches += ['']
            elif '<UNK>' in patch_ast['n']:
                fathers, edges, nodes = patch_ast['f'].split(), patch_ast['e'].split(), patch_ast['n'].split()
                reconstructed_nodes, ctx_ast_nodes = reconstruct_unk(
                    fathers, edges, nodes, abs2src, src2abs,
                    ctx_ast_nodes, node_num_before, sibling_num_before, file_path, identifiers
                )
                if patch_fix_type == 'general':
                    ctx_ast_nodes_g = ctx_ast_nodes
                else:
                    ctx_ast_nodes_i = ctx_ast_nodes
                if reconstructed_nodes is None or reconstructed_nodes == []:
                    continue
                for reconstructed_node in reconstructed_nodes[: 20]:
                    code_patches.append({
                        'patch': reconstruct_ast(patch_ast['f'].split(), patch_ast['e'].split(),
                                                 reconstructed_node.split(), abs2src).strip(),
                        'rank': patch_ast['rank'],
                        'score': patch_ast['score'],
                        'fix_type': patch_ast['fix_type']
                    })
            else:
                code_patches.append({
                    'patch': reconstruct_ast(patch_ast['f'].split(), patch_ast['e'].split(), patch_ast['n'].split(),
                                             abs2src).strip(),
                    'rank': patch_ast['rank'],
                    'score': patch_ast['score'],
                    'fix_type': patch_ast['fix_type']
                })
        except Exception as e:
            continue
    return code_patches
