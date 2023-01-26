import codecs

class Vocabulary:
    def __init__(self):
        self.dictionary = {}
        self.symbols = []

    def add_symbol(self, symbol):
        if symbol in self.dictionary:
            return self.dictionary[symbol]
        idx = len(self.dictionary)
        self.dictionary[symbol] = idx
        self.symbols.append(symbol)
        return idx

    def __getitem__(self, item):
        return self.symbols[item] if item < len(self.symbols) else self.unk_index

    def __len__(self):
        return len(self.dictionary)

    def read_vocabulary_file(self, vocabulary_file):
        fp = codecs.open(vocabulary_file, 'r', 'utf-8')
        for l in fp.readlines():
            symbol = l.strip()
            symbol = symbol.replace(" ", "<SPACE>")
            self.add_symbol(symbol)

    def index(self, symbol, mapping=None):
        if type(symbol) == list:
            return [self.index(s, mapping) for s in symbol]
        if mapping is not None and symbol in mapping:
            symbol = mapping[symbol]
        symbol = symbol.replace(" ", "<SPACE>")
        if symbol in self.dictionary:
            return self.dictionary[symbol]
        return self.unk_index


class NodeVocabulary(Vocabulary):
    def __init__(self, nonterminal_file, terminal_file, abstraction_file, idiom_file, nonidentifier_file):
        super(NodeVocabulary, self).__init__()

        self.pad_word = '<PAD>'
        self.unk_word = '<UNK>'
        self.sos_word = '<SOS>'
        self.eos_word = '<EOS>'
        self.pad_statement_word = 'PAD_STATEMENT'
        self.pad_index = self.add_symbol(self.pad_word)
        self.unk_index = self.add_symbol(self.unk_word)
        self.sos_index = self.add_symbol(self.sos_word)
        self.eos_index = self.add_symbol(self.eos_word)
        self.pad_statement_index = self.add_symbol(self.pad_statement_word)

        self.literal_string = []
        self.literal_char = []
        self.literal_number = []

        self.read_vocabulary_file(nonterminal_file)
        self.nonterminal_size = len(self.dictionary)

        self.read_vocabulary_file(abstraction_file)
        self.read_vocabulary_file(nonidentifier_file)
        self.read_vocabulary_file(idiom_file)
        self.read_vocabulary_file(terminal_file)

    def read_vocabulary_file(self, vocabulary_file):
        fp = codecs.open(vocabulary_file, 'r', 'utf-8')
        for l in fp.readlines():
            symbol = l.strip()
            symbol = symbol.replace(" ", "<SPACE>")
            idx = self.add_symbol(symbol)
            if symbol.count('\"') >= 2 or 'STRING_' in symbol:
                self.literal_string.append(idx)
            elif (symbol.count("\'") >= 2 and len(symbol) == 3) or 'CHAR_' in symbol:
                self.literal_char.append(idx)
            elif 'INT_' in symbol or 'FLOAT_' in symbol or ('0' <= symbol[0] <= '9') or \
                    (len(symbol) > 1 and symbol[0] == '-' and '0' <= symbol[1] <= '9'):
                self.literal_number.append(idx)

    def is_terminal(self, symbol, mapping=None):
        return self.index(symbol, mapping) >= self.nonterminal_size

    def string(self, tensor):
        result = ''
        for t in tensor:
            if t == self.pad_index:
                continue
            result += self[t] + ' '
        return result.strip()


class EdgeVocabulary(Vocabulary):
    def __init__(self, edge_file):
        super(EdgeVocabulary, self).__init__()

        self.pad_word = '<PAD>'
        self.unk_word = '<UNK>'
        self.eos_word = '<EOS>'
        self.self_loop_word = '<SELF>'
        self.sibling_word = '<SIBLING>'
        self.pad_index = self.add_symbol(self.pad_word)
        self.unk_index = self.add_symbol(self.unk_word)
        self.eos_index = self.add_symbol(self.eos_word)
        self.self_loop_index = self.add_symbol(self.self_loop_word)
        self.sibling_index = self.add_symbol(self.sibling_word)

        self.read_vocabulary_file(edge_file)

    def string(self, tensor):
        result = ''
        for t in tensor:
            if t == self.pad_index:
                continue
            if '->' in self[t]:
                result += self[t].split('->')[1] + ' '
            else:
                result += self[t] + ' '
        return result.strip()

