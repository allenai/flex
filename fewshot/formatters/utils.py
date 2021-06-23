from typing import List, Collection

SEP = '###'


def annotate_spans(
    tokens: List[str],
    span_token_indices: List[int],
    delimiter: str = '#',
    separator: str = ' ',
) -> List[str]:
    annotated = tokens[:span_token_indices[0]]
    if len(span_token_indices) == 1:
        annotated = annotated + [delimiter + separator + tokens[span_token_indices[0]] + separator + delimiter]
    else:
        annotated = (
            annotated
            + [delimiter + separator + tokens[span_token_indices[0]]]
            + tokens[(span_token_indices[0] + 1):span_token_indices[-1]]
            + [tokens[span_token_indices[-1]] + separator + delimiter]
        )
    try:
        return annotated + tokens[(span_token_indices[-1] + 1):]
    except IndexError:
        return annotated


def snli_format(premise: str, hypothesis: str, sep: str = SEP):
    return f"{premise}{sep}{hypothesis}"


def csv_with_or(lst: Collection[str]):
    if len(lst) == 1:
        return lst[0]
    elif len(lst) == 2:
        return ' or '.join(lst)
    else:
        return ', '.join(lst[:-1]) + ', or ' + lst[-1]


def to_dict_of_lists(list_of_dicts):
    return {k: [d[k] for d in list_of_dicts] for k in list_of_dicts[0]}
