import codecs
import json


class IdentifierDataset:
    def __init__(self, identifier_file, idioms_file, ast_file, ochiai=False):
        self.identifiers = {}
        self.ast = {}
        self.idioms = set()

        idioms_fp = codecs.open(idioms_file, 'r', 'utf-8')
        for l in idioms_fp.readlines():
            l = l.strip()
            self.idioms.add(l)
        idioms_fp.close()

        ast = json.load(open(ast_file, 'r'))
        for data in ast:
            self.ast[data['id']] = data

        if ochiai:
            identifiers = json.load(open(identifier_file, 'r'))
            mapping_dict = identifiers['mapping_dict']
            identifier_list = identifiers['identifiers_list']
            self.identifiers = {}
            for id, key in mapping_dict.items():
                self.identifiers[str(id)] = identifier_list[key]
        else:
            self.identifiers = json.load(open(identifier_file, 'r'))
        print('identifier dataset load finish')

    @staticmethod
    def get_token(name, mappings):
        if '<' in name:
            name = name[: name.index('<')]
        if name in mappings:
            return mappings[name]
        return name

    def prepare(self, id, class_name):
        identifier = self.identifiers[str(id)]
        if identifier == {}:
            return None
        data = self.ast[id]

        var_type = {}
        type_var, type_method = {}, {}
        qualifier_var, qualifier_method = {}, {}
        super_arg = {}
        for token in list(self.idioms) + list(data['mappings'].keys()) + ['int', 'float']:
            is_class = True if token == class_name else False

            identifier.update({
                'int': [{'itype': 'TYPE', 'fields': ['length'], 'methods': ['length'], 'supers': [], 'constructors': []}],
                'float': [{'itype': 'TYPE', 'fields': ['length'], 'methods': ['length'], 'supers': [], 'constructors': []}],
                'String': [{'itype': 'TYPE', 'fields': ['length'],
                            'methods': ['length', 'charAt', 'substring', 'valueOf', 'equals'],
                            'supers': [], 'constructors': []}],
            })
            if 'Deque' in identifier:
                identifier['Deque'][0]['methods'].append('isEmpty')
            if 'ArrayList' in identifier:
                identifier['ArrayList'][0]['methods'].append('containsAll')

            if token not in identifier:
                continue
            semantics = identifier[token]
            # print(token, semantics)
            if token in data['mappings']:
                token = data['mappings'][token]
            for semantic in semantics:
                if semantic['itype'] == 'TYPE':
                    if token not in qualifier_var:
                        qualifier_var[token] = set()
                    for field in semantic['fields']:
                        qualifier_var[token].add(self.get_token(field, data['mappings']))
                    if token not in qualifier_method:
                        qualifier_method[token] = set()
                    for method in semantic['methods']:
                        qualifier_method[token].add(self.get_token(method, data['mappings']))
                    if is_class:
                        for super_class_name in semantic['supers']:
                            if super_class_name in identifier:
                                for super_class in identifier[super_class_name]:
                                    if super_class['itype'] == 'TYPE':
                                        for constructor in super_class['constructors']:
                                            for i, arg in enumerate(constructor):
                                                if i not in super_arg:
                                                    super_arg[i] = set()
                                                super_arg[i].add(arg)
                elif semantic['itype'] == 'METHOD':
                    dtype = semantic['dtype']
                    if dtype not in type_method:
                        type_method[dtype] = set()
                    type_method[dtype].add(token)
                    
                    if '<' in dtype:
                        dtype = dtype[: dtype.index('<')]
                    if dtype in identifier:
                        for return_semantic in identifier[dtype]:
                            if return_semantic['itype'] == 'TYPE':
                                for selector in return_semantic['fields'] + return_semantic['methods']:
                                    if selector in identifier:
                                        for selector_semantic in identifier[selector]:
                                            if 'dtype' not in selector_semantic:
                                                possible_type = selector
                                            else:
                                                possible_type = selector_semantic['dtype']
                                            if possible_type not in type_method:
                                                type_method[possible_type] = set()
                                                type_method[possible_type].add(token)
                elif semantic['itype'] == 'VAR':
                    dtype = semantic['dtype']
                    if '<' in dtype:
                        dtype = dtype[: dtype.index('<')]
                    if dtype == 'Map':
                        dtype = 'HashMap'
                    if dtype == 'Set':
                        dtype = 'HashSet'

                    if dtype not in type_var:
                        type_var[dtype] = set()
                    type_var[dtype].add(token)
                    if token not in var_type:
                        var_type[token] = set()
                    var_type[token].add(self.get_token(dtype, data['mappings']))

                    if dtype in identifier:
                        for return_semantic in identifier[dtype]:
                            if return_semantic['itype'] == 'TYPE':
                                for selector in return_semantic['fields'] + return_semantic['methods']:
                                    if selector in identifier:
                                        for selector_semantic in identifier[selector]:
                                            if 'dtype' not in selector_semantic:
                                                possible_type = selector
                                            else:
                                                possible_type = selector_semantic['dtype']
                                            if possible_type not in type_method:
                                                type_var[possible_type] = set()
                                                type_var[possible_type].add(token)

        vars = list(var_type.keys())
        # vars = list(self.idioms) + [abstract for abstract in data['mappings'].values() if abstract[:4] == 'VAR_']
        methods = list(self.idioms) + [abstract for abstract in data['mappings'].values() if abstract[:7] == 'METHOD_']
        return vars, methods, var_type, type_var, type_method, qualifier_var, qualifier_method, super_arg
