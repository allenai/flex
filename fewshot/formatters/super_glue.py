from .utils import snli_format
_boolq_labels = ['no', 'yes']
_boolq_ending = 'yes or no?'
_multirc_labels = ['false', 'true']
_multirc_ending = 'true or false?'


def boolq(d): return {
    'flex.txt': (
        f"{d['passage']}"
        f"\nquestion: {d['question']}? {_boolq_ending}"
    ),
    'label': _boolq_labels[d['label']],
}


def cb(d):
    return {
        'flex.txt': snli_format(
            premise=d['premise'],
            hypothesis=d['hypothesis'],
        ),
    }


def copa(d): return {
    'flex.txt': (
        f"{d['premise']}"
        f"\nquestion: What's the {d['question']} for this?"
        f"\n1: {d['choice1']}"
        f"\n2: {d['choice2']}"
    ),
    'label': str(d['label'] + 1),
}


def multirc(d): return {
    'flex.txt': (
        f"{d['paragraph']}"
        f"\n\n{d['question']}"
        f"\nCandidate answer: {d['answer']}. {_multirc_ending}",
    ),
    'label': _multirc_labels[d['label']],
}
