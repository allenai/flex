from .utils import snli_format, SEP


def sentence_formatter(d):
    return {'flex.txt': d['sentence']}


def mrpc_formatter(d):
    return {'flex.txt': SEP.join([d['sentence1'], d['sentence2']])}


rte_formatter = mrpc_formatter
wnli_formatter = mrpc_formatter


def qqp_formatter(d):
    return {'flex.txt': SEP.join([d['question1'], d['question2']])}


def mnli_formatter(d):
    return {
        'flex.txt': snli_format(
            premise=d['premise'],
            hypothesis=d['hypothesis'],
        )
    }


def qnli_formatter(d):
    return {'flex.txt': SEP.join([d['question'], d['sentence']])}
