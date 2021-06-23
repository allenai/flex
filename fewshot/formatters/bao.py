from typing import List
from .utils import annotate_spans


def _get_annotated_tokens(d: dict) -> List[str]:
    tokens = d['tokens']
    for head_tokens in d['head']['indices']:
        tokens = annotate_spans(tokens, head_tokens, delimiter='*')
    for tail_tokens in d['tail']['indices']:
        tokens = annotate_spans(tokens, tail_tokens, delimiter='#')
    return tokens


def newsgroupbao(d): return {
    'flex.txt': d['text'],
}


def reutersbao(d): return {
    'flex.txt': d['title'] + ' ' + d['text'],
}


def huffpostbao(d): return {
    'flex.txt': d['headline'],
}


def fewrelbao(d): return {
    'flex.txt': ' '.join(_get_annotated_tokens(d)),
}


def amazon(d): return {
    'flex.txt': d['text'],
}
