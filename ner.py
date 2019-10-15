import ahocorasick

from nlu import tokenize


class TextMatcher:
    def __init__(self):
        self.A = ahocorasick.Automaton()

    def preprocess(self, text):
        stem_text = tokenize(text, stem=True, join=True)
        return ' ' + stem_text + ' '

    def fit(self, texts):
        for text in texts:
            preprocessed = self.preprocess(text)
            if len(preprocessed.strip()) > 0:
                self.A.add_word(preprocessed, (preprocessed, text))
        self.A.make_automaton()

    def predict(self, texts):
        tokens_list, spans_list = [], []
        for text in texts:
            t, s = self.find_tokens(text)
            tokens_list.append(t)
            spans_list.append(s)
        return tokens_list, spans_list

    def find_tokens(self, text):
        tokens = tokenize(text)
        stem_text = tokenize(text, stem=True, join=True)
        spans = []

        positions = []
        idx = 0
        while idx > -1:
            idx = stem_text.find(' ', idx + 1)
            if idx != -1:
                positions.append(idx + 1)
        positions.append(len(stem_text) + 1)
        inv_positions = {p: i for i, p in enumerate(positions)}

        haystack = ' ' + stem_text + ' '
        for end_idx, (entity, original) in self.A.iter(haystack):
            e_tokens = entity.split()
            j = inv_positions[end_idx]
            spans.append([entity, j - len(e_tokens) + 1, j + 1])

        return tokens, spans

    def highlight(self, text):
        tokens, spans = self.find_tokens(text)
        for e, l, r in spans:
            tokens[l] = '(' + tokens[l]
            tokens[r-1] = tokens[r-1] + ')'
            for i in range(l, r):
                tokens[i] = tokens[i].upper()
        return ' '.join(tokens)