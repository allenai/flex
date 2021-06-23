from typing import Generator
from .utils import to_dict_of_lists, snli_format, annotate_spans


def snli_batched(d):
    mapping = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }
    examples = [
        {
            **{k: d[k][i] for k in d},
            'flex.txt': snli_format(premise=premise, hypothesis=hypothesis),
            'label': mapping[label],
        }
        for i, (premise, hypothesis, label) in enumerate(zip(d['premise'], d['hypothesis'], d['label']))
        if label >= 0
    ]
    if examples:
        return to_dict_of_lists(examples)
    else:
        return {k: [] for k in d}


def trec(d):
    # More readable labels based on manual templates from https://arxiv.org/pdf/2012.15723.pdf
    mapping = {
        0: 'description',
        1: 'entity',
        2: 'expression',
        3: 'human',
        4: 'number',
        5: 'location',
    }
    return {
        'flex.txt': d['text'],
        'label': mapping[d['label-coarse']],
    }


def conll_batched(d):
    names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    to_readable = {
        'PER': 'person',
        'ORG': 'organization',
        'LOC': 'location',
        'MISC': 'other',
    }

    def to_text(tokens, entity_indices) -> str:
        return ' '.join(annotate_spans(
            tokens=tokens,
            span_token_indices=entity_indices,
            delimiter='#',
            separator=' ',
        ))

    def format_example(ner_tags, tokens, entity_indices):
        label = to_readable[names[ner_tags[entity_indices[0]]].split('-')[1]]
        return {'label': label, 'flex.txt': to_text(tokens, entity_indices)}

    def extract_examples(ner_tags, tokens) -> Generator[dict, None, None]:
        entity_indices = []
        for i, t in enumerate(ner_tags):
            if names[t].startswith('I'):
                entity_indices.append(i)
            elif names[t].startswith('B'):
                if entity_indices:
                    yield format_example(ner_tags, tokens, entity_indices)
                entity_indices = [i]
            elif names[t] == 'O' and entity_indices:
                yield format_example(ner_tags, tokens, entity_indices)
                entity_indices = []
        if entity_indices:
            yield format_example(ner_tags, tokens, entity_indices)

    examples = []
    for i, (ner_tags, tokens) in enumerate(zip(d['ner_tags'], d['tokens'])):
        for ex in extract_examples(ner_tags=ner_tags, tokens=tokens):
            examples.append({
                **{k: d[k][i] for k in d},
                **ex,
            })
    if examples:
        return to_dict_of_lists(examples)
    else:
        return {k: [] for k in d}


def scitail(d):
    return {
        'flex.txt': snli_format(
            premise=d['sentence1'],
            hypothesis=d['sentence2'],
        ),
        'label': d['gold_label'],  # entailment | neutral
    }


def mr(d):
    return {'flex.txt': d['text']}


def cr_batched(d):
    def orientation(sentiments):
        n_pos = len([x for x in sentiments if x > 0])
        n_neg = len([x for x in sentiments if x < 0])
        if n_pos > n_neg:
            return 'positive'
        elif n_neg > n_pos:
            return 'negative'
        else:
            return 'neutral/unknown'
    examples = [
        {
            **{k: d[k][i] for k in d},
            'flex.txt': text,
            'label': orientation(sentiments),
        }
        for i, (text, sentiments) in enumerate(zip(d['text'], d['features_sentiment']))
    ]
    examples = [x for x in examples if 'neutral' not in x['label']]
    if examples:
        return to_dict_of_lists(examples)
    else:
        return {k: [] for k in d}


def subj(d):
    return {'flex.txt': d['text']}
