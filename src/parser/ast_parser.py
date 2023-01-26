import sys
import os

PARSER_DIR = os.path.abspath(__file__)
PARSER_DIR = PARSER_DIR[: PARSER_DIR.rfind('/') + 1]
sys.path.append(PARSER_DIR + '../grammar/')
sys.path.append(PARSER_DIR + '../../')

from grammar import node_to_edge
from javalang.parser import Parser
from javalang.tokenizer import tokenize
from javalang.ast import Node


def get_type_name(obj):
    return type(obj).__name__


def get_node_attrs(node):
    name = get_type_name(node)
    return [attr[0] for attr in node_to_edge[name]]


def dfs(ast):
    stack = []
    traverse, edges = [], []
    stack.append((ast, None, None))
    while len(stack) > 0:
        (node, father_index, edge_type) = stack.pop()

        self_index = len(traverse)
        if father_index is not None:
            edges.append([father_index, self_index, edge_type])

        traverse.append(node)
        if isinstance(node, Node):
            attrs = get_node_attrs(node)
            for attr in list(reversed(attrs)):
                val = getattr(node, attr)
                if attr == 'return_type' and val is None:
                    val = "void"
                if isinstance(val, list):
                    for child in list(reversed(val)):
                        stack.append((child, self_index, attr))
                elif val not in [None, [], dict(), (), set(), '', True, False]:
                    stack.append((val, self_index, attr))
    return traverse, edges

def parse_ast():
    src = 'void add () {String s = "hello world";}'
    tokens = tokenize(src)
    parser = Parser(tokens)
    ast = parser.parse_member_declaration()
    traverse, edges = dfs(ast)

    print(len(traverse))
    for node in traverse:
       print(node.__class__.__name__, node)
       if isinstance(node, Node):
           print(node.to_code())
    print(ast.to_code())


if __name__ == "__main__":
    parse_ast()

