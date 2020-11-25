from collections import defaultdict

from nltk import CFG, wordpunct_tokenize
from nltk.parse import BottomUpLeftCornerChartParser


def try_parsing(parser, text):
    if isinstance(text, str):
        text = text.split()
    try:
        result = parser.parse_one(text)
        return result
    except ValueError:
        return None


def set_span(node, left=0):
    """ Размечаем каждый узел (проективного) дерева началом и концом интервала слов, которые этот узел накрывает. """
    if isinstance(node, str):
        return left + len(node.split())
    right = left
    for child in node:
        right = set_span(child, right)
    node.span = [left, right]
    return right


def flatten_tree(node, keys, result=None, return_span=False):
    """ Для всех ключевых узлов дерева, возвращаем токены (или их позициии), соответствующие этим узлам ы"""
    if node is None:
        return None
    if isinstance(node, str):
        return result
    if result is None:
        result = defaultdict(list)
    if node.label() in keys:
        result[node.label()].append(node.span if return_span else ' '.join(node.leaves()))
    for child in node:
        flatten_tree(child, keys, result=result, return_span=return_span)
    return result


def calculate_spans(lemmas, parser, keys, return_span=False):
    """ Пытаемся разобрать текст парсером и вернуть разбор в плоском виде"""
    tree = try_parsing(parser, lemmas)
    if not tree:
        return None
    set_span(tree)
    return flatten_tree(tree, keys=keys, return_span=return_span)


def slots_to_tags(words, slots):
    """ Превращаем словарик слот -> список интервалов в последовательность IOB меток для каждого токена. """
    tags = ['O' for w in words]
    for slot_name, slot_values in slots.items():
        for l, r in slot_values:
            tags[l] = 'B-' + slot_name
            for j in range(l+1, r):
                tags[j] = 'I-' + slot_name
    return tags


def create_grammar(filename, places, directions):
    with open(filename, 'r', encoding='utf-8') as f:
        grammar_text = f.read()
    appended = """
    PLACE -> {}
    DIRECTION_WORD -> {}
    """.format(
        ' | '.join([' '.join(["'{}'".format(w) for w in l.split()]) for l in places]),
        ' | '.join([' '.join(["'{}'".format(w) for w in l.split()]) for l in directions]),
    ).replace('ё', 'е')
    grammar = CFG.fromstring(grammar_text + appended)
    grammar_parser = BottomUpLeftCornerChartParser(grammar)
    return grammar_parser
