import sys
import os

VALIDATION_DIR = os.path.abspath(__file__)
VALIDATION_DIR = VALIDATION_DIR[: VALIDATION_DIR.rfind('/') + 1]

sys.path.append(VALIDATION_DIR + '../../')
from javalang.tree import BasicType, ReferenceType, MemberReference, MethodInvocation, Literal, This, ArraySelector
from javalang.tree import ClassCreator, Cast, BinaryOperation, IfStatement, ReturnStatement, StatementExpression


def get_type_of_member(member, qualifier, abs2src, identifiers):
    member = abs2src[member] if member in abs2src else member
    if member in identifiers:
        for identifier in identifiers[member]:
            if identifier['itype'] == 'TYPE':
                return member
            if identifier['qualifier'] != '':
                continue
            if identifier['itype'] == 'METHOD' or identifier['itype'] == 'VAR':
                return identifier['dtype']
        return None
    elif member == 'TYPE_<UNK>':
        return member
    else:
        return None


def get_type_of_member_reference_or_method_invocation(node, abs2src, identifiers):
    if 'selectors' in node.children and node.children['selectors'] > 0:
        for i in range(1, 1 + node.children['selectors']):
            selector = node.children['selectors'][-i]
            if not isinstance(selector.node, ArraySelector):
                if isinstance(selector.node, MethodInvocation) or isinstance(selector.node, MemberReference):
                    return get_type_of_member_reference_or_method_invocation(selector, abs2src, identifiers)
    else:
        if 'qualifier' not in node.children:
            return get_type_of_member(node.children['member'][0].node, '', abs2src, identifiers)
        else:
            qualifier = node.children['qualifier'][0].node
            return get_type_of_member(node.children['member'][0].node, qualifier, abs2src, identifiers)


def get_type_of_this(node, abs2src, identifiers):
    if 'selectors' in node.children and node.children['selectors'] > 0:
        for i in range(1, 1 + node.children['selectors']):
            selector = node.children['selectors'][-i]
            if not isinstance(selector.node, ArraySelector):
                if isinstance(selector.node, MethodInvocation) or isinstance(selector.node, MemberReference):
                    return get_type_of_member_reference_or_method_invocation(selector, abs2src, identifiers)
    return None


def get_type_of_literal(node):
    value = node.children['value'][0].node
    if value in ['0', '1'] or 'INT_' in value:
        return 'int'
    elif value in ['0.0', '1.0'] or 'FLOAT_' in value:
        return 'float'
    elif (value[0] == '"' and value[-1] == '"') or 'STRING_' in value:
        return 'String'
    # elif value in ['"."', '"<SPACE>"', '"0"', '"1"', '"', "'", ''] or 'STRING_' in value:
    #    return 'String'
    elif (value[0] == "'" and value[-1] == "'") or 'CHAR_' in value:
        return 'char'
    # elif value in ["'.'", "'\"'", "':'"] or 'CHAR_' in value:
    #    return 'char'
    elif value in ['true', 'false']:
        return 'boolean'
    return None


def get_type_of_class_creator_or_cast(node, abs2src, identifiers):
    if 'selectors' in node.children and node.children['selectors'] > 0:
        selector = node.children['selectors'][:-1]
        if not isinstance(selector.node, ArraySelector):
            if isinstance(selector.node, MethodInvocation) or isinstance(selector.node, MemberReference):
                return get_type_of_member_reference_or_method_invocation(selector, abs2src, identifiers)
    else:
        ty = node.children['type']
        if isinstance(ty.node, BasicType) or isinstance(ty.node, ReferenceType):
            ty = str(ty.children['name'][0].node)
            ty = abs2src[ty] if ty in abs2src else ty
            return ty
        return None


def get_type_of_binary_operation(node, abs2src, identifiers, operand=None):
    if 'operator' in node.children:
        operator = node.children['operator'][0].node
        if operator in ['&&', '||']:
            return 'boolean'
    if 'operandl' in node.children and operand is None:
        child = node.children['operandl'][0]
    elif 'operandr' in node.children and operand is None:
        child = node.children['operandr'][0]
    elif operand in node.children:
        child = node.children[operand][0]
    else:
        return None
    if not hasattr(child, 'node'):
        return None
    if isinstance(child.node, MemberReference) or isinstance(child.node, MethodInvocation):
        return get_type_of_member_reference_or_method_invocation(child, abs2src, identifiers)
    elif isinstance(child.node, Literal):
        return get_type_of_literal(child)
    elif isinstance(child.node, ClassCreator) or isinstance(child.node, Cast):
        return get_type_of_class_creator_or_cast(child, abs2src, identifiers)
    elif isinstance(child.node, This):
        return get_type_of_this(node, abs2src, identifiers)
    elif isinstance(child.node, BinaryOperation):
        return get_type_of_binary_operation(child, abs2src, identifiers)
    else:
        return None


def get_arguments(node, abs2src, identifiers):
    arguments = []
    if 'arguments' in node.children:
        for argument in node.children['arguments']:
            if isinstance(argument.node, MemberReference) or isinstance(argument.node, MethodInvocation):
                arguments.append(get_type_of_member_reference_or_method_invocation(argument, abs2src, identifiers))
            elif isinstance(argument.node, Literal):
                arguments.append(get_type_of_literal(argument))
            elif isinstance(argument.node, ClassCreator) or isinstance(argument.node, Cast):
                arguments.append(get_type_of_class_creator_or_cast(argument, abs2src, identifiers))
            elif isinstance(argument.node, This):
                arguments.append(get_type_of_this(node, abs2src, identifiers))
            elif isinstance(argument.node, BinaryOperation):
                arguments.append(get_type_of_binary_operation(node, abs2src, identifiers))
            else:
                arguments.append(None)
    return arguments


def analyze_type(ast_nodes, node, abs2src, identifiers):
    constrains = {'methods': None}
    if isinstance(node.father.node, MethodInvocation) and str(node.edge) == 'qualifier':
        if 'member' in node.father.children and type(node.father.children['member'][0].node) == str:
            member = node.father.children['member'][0].node
            if '_<UNK>' not in member:
                member = abs2src[member] if member in abs2src else member
                constrains['methods'] = member
    return constrains


def analyze_method(ast_nodes, node, abs2src, identifiers):
    constrains = {'return_type': None, 'qualifier': None, 'arguments': None,
                  'return_type_methods': None, 'modifier': None}

    if isinstance(node.father.node, MethodInvocation) and str(node.edge) == 'member':
        constrains['arguments'] = get_arguments(node.father, abs2src, identifiers)

        if 'qualifier' in node.father.children and type(node.father.children['qualifier'][0].node) == str:
            qualifier = node.father.children['qualifier'][0].node
            if 'TYPE_' in qualifier or (qualifier in identifiers and identifiers[qualifier][0]['itype'] == 'TYPE'):
                constrains['modifier'] = 'static'
            qualifier = abs2src[qualifier] if qualifier in abs2src else qualifier
            constrains['qualifier'] = get_type_of_member(qualifier, '', abs2src, identifiers)
            if constrains['qualifier'] is None:
                constrains['qualifier'] = 'UNK'
            if '.' in constrains['qualifier']:
                constrains['qualifier'] = constrains['qualifier'].split('.')[-1]

        if 'selectors' in node.father.children:
            selector = node.father.children['selectors'][0]
            if isinstance(selector.node, MethodInvocation):
                if 'member' in selector.children and type(selector.children['member'][0].node) == str:
                    member = selector.children['member'][0].node
                    if '_<UNK>' not in member:
                        member = abs2src[member] if member in abs2src else member
                        constrains['return_type_methods'] = member
        else:
            grandfather = node.father.father
            if grandfather is not None:
                if isinstance(grandfather.node, IfStatement) and str(node.father.edge) == 'condition':
                    constrains['return_type'] = 'boolean'
                elif isinstance(grandfather.node, ReturnStatement) and str(node.father.edge) == 'expression':
                    return_type = None
                    if 'return_type' in ast_nodes[0].children:
                        return_type = ast_nodes[0].children['return_type'][0]
                        if isinstance(return_type.node, BasicType) or isinstance(return_type.node, ReferenceType):
                            return_type = str(return_type.children['name'][0].node)
                            return_type = abs2src[return_type] if return_type in abs2src else return_type
                        else:
                            return_type = 'void'
                    constrains['return_type'] = return_type
                elif isinstance(grandfather.node, StatementExpression) and str(node.father.edge) == 'expression':
                    constrains['return_type'] = 'void'
                elif isinstance(grandfather.node, BinaryOperation):
                    if str(node.father.edge) == 'operandl':
                        constrains['return_type'] = get_type_of_binary_operation(grandfather, abs2src, identifiers,
                                                                                 operand='operandr')
                    else:
                        constrains['return_type'] = get_type_of_binary_operation(grandfather, abs2src, identifiers,
                                                                                 operand='operandl')
    return constrains


def analyze_var(ast_nodes, node, abs2src, identifiers):
    constrains = {'type': None, 'qualifier': None}

    if isinstance(node.father.node, MemberReference) and str(node.edge) == 'member':
        if 'qualifier' in node.father.children and type(node.father.children['qualifier'][0].node) == str:
            qualifier = node.father.children['qualifier'][0].node
            qualifier = abs2src[qualifier] if qualifier in abs2src else qualifier
            constrains['qualifier'] = get_type_of_member(qualifier, '', abs2src, identifiers)
        if 'selectors' in node.father.children:
            pass
        else:
            grandfather = node.father.father
            if grandfather is not None:
                if isinstance(grandfather.node, MethodInvocation) and str(node.father.edge) == 'arguments':
                    method = grandfather.children['member'][0].node
                    method = abs2src[method] if method in abs2src else method
                    if method in identifiers and identifiers[method][0]['itype'] == 'METHOD':
                        arguments = identifiers[method][0]['params']
                        index = grandfather.children['arguments'].index(node.father)
                        if constrains['qualifier'] is None and 0 <= index < len(arguments):
                            constrains['type'] = arguments[index]
                elif isinstance(grandfather.node, ReturnStatement) and str(node.father.edge) == 'expression':
                    return_type = None
                    if 'return_type' in ast_nodes[0].children:
                        return_type = ast_nodes[0].children['return_type'][0]
                        if isinstance(return_type.node, BasicType) or isinstance(return_type.node, ReferenceType):
                            return_type = str(return_type.children['name'][0].node)
                            return_type = abs2src[return_type] if return_type in abs2src else return_type
                        else:
                            return_type = None
                    constrains['type'] = return_type
                elif isinstance(grandfather.node, IfStatement) and str(node.father.edge) == 'condition':
                    constrains['type'] = 'boolean'

    return constrains
