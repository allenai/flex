# TODO: Check _read_words format for fewrel matches what bao et al did in paper
# TODO: Set the way
# TODO: Handle fewrel in for fewshot.eval
# TODO: Called 'flex.txt' in raw dataset, then set to 'txt' in challenge
from pathlib import Path
from torchtext.data import get_tokenizer
import pickle
import itertools
import collections
import os
import logging
import hydra
import torch
from torchtext.vocab import Vocab, Vectors
from bao.dataset.loader import _read_words, _data_to_nparray
from bao.embedding.cxtebd import CXTEBD
from bao.embedding.wordebd import WORDEBD
from bao.embedding.avg import AVG
from bao.embedding import factory as ebd
from bao.classifier import factory as clf
from bao.dataset import stats
from bao.main import set_seed
from bao.train import factory as train_utils
import fewshot

logger = logging.getLogger(__name__)
tprint = logger.info

MAX_DOC_TOKENS = 500


def _get_tmp_dir():
    p = Path('~').expanduser() / '.flex'
    if not p.exists():
        p.mkdir()
    return p


# TODO: Remove since looking at test data during training is unfair. Here because Bao et al use it for vocab during training.
test_datasets = {
    'amazon': fewshot.stores.AmazonStoreCfg(split='test'),
    'fewrel': fewshot.stores.FewrelStoreCfg(split='test'),
    'huffpost': fewshot.stores.HuffpostStoreCfg(split='test'),
    'newsgroup': fewshot.stores.NewsgroupStoreCfg(split='test'),
    'reuters': fewshot.stores.ReutersStoreCfg(split='test'),
}


def _format_fewrel(d: dict):
    def get_token_range(tokens):
        # TODO: Assert token indices are consecutive
        return [tokens[0][0], tokens[0][-1]]
    return {
        **d,
        'text': [s.lower() for s in d['tokens']],
        'head': get_token_range(d['head_tokens']),
        'tail': get_token_range(d['tail_tokens']),
    }


tokenizer = get_tokenizer('spacy', language='en')


def _tokenize_and_truncate(doc: str, max_doc_tokens=MAX_DOC_TOKENS):
    return [s for s in [s.lower().strip() for s in tokenizer(doc.replace('\n', ' '))] if s][:max_doc_tokens]


def _convert_data(datas, args):
    labels = {}
    for data in datas:
        for e in data.store:
            if not labels:
                labels[e[data.label]] = 0
            else:
                labels[e[data.label]] = labels.get(e[data.label], max(labels.values()) + 1)

    def _labels_to_int_factory(label_key):
        return lambda d: {**d, 'label': labels[d[label_key]]}

    ds = []
    for data in datas:
        d_out = data.store.map(_labels_to_int_factory(data.label))
        if 'pos' in args.auxiliary:
            d_out = d_out.map(_format_fewrel)
        else:
            # TODO: Make this easier for others.
            d_out = d_out.map(
                lambda d: {**d, 'text': _tokenize_and_truncate(d['flex.txt'])}
            )
        ds.append(d_out)
    return ds


def load_dataset(args):
    tprint('Loading data')
    challenge = fewshot.get_challenge_spec(args.challenge)
    train_data = challenge.get_stores('train')[args.dataset]
    val_data = challenge.get_stores('val')[args.dataset]
    # TODO: Remove since looking at test data during training is unfair. Here because Bao et al use it for vocab during training.
    test_data = hydra.utils.instantiate(test_datasets[args.dataset])
    train_data, val_data, test_data = _convert_data([train_data, val_data, test_data], args)
    all_data = itertools.chain(train_data, val_data, test_data)
    vocab_counter = collections.Counter(_read_words(all_data))

    tprint('Loading word vectors')
    path = os.path.join(args.wv_path, args.word_vector)
    if not os.path.exists(path):
        # Download the word vector and save it locally:
        tprint('Downloading word vectors')
        import urllib.request
        urllib.request.urlretrieve(
            'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec',
            path)

    vectors = Vectors(args.word_vector, cache=args.wv_path, max_vectors=args.max_vectors)
    vocab = Vocab(vocab_counter, vectors=vectors,
                  specials=['<pad>', '<unk>'], min_freq=5)

    # print word embedding statistics
    wv_size = vocab.vectors.size()
    tprint('Total num. of words: {}, word vector dimension: {}'.format(
        wv_size[0],
        wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(
        torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    tprint(('Num. of out-of-vocabulary words'
            '(they are initialized to zeros): {}').format(num_oov))

    # Split into meta-train, meta-val, meta-test data
    tprint('#train {}, #val {}, #test {}'.format(
        len(train_data), len(val_data), len(test_data)))

    # Convert everything into np array for fast data loading
    train_data = _data_to_nparray(train_data, vocab, args)
    val_data = _data_to_nparray(val_data, vocab, args)
    test_data = _data_to_nparray(test_data, vocab, args)

    train_data['is_train'] = True
    # this tag is used for distinguishing train/val/test when creating source pool

    stats.precompute_stats(train_data, val_data, test_data, args)

    if args.meta_w_target:
        # augment meta model by the support features
        if args.bert:
            ebd = CXTEBD(args.pretrained_bert,
                         cache_dir=args.bert_cache_dir,
                         finetune_ebd=False,
                         return_seq=True)
        else:
            ebd = WORDEBD(vocab, finetune_ebd=False)

        train_data['avg_ebd'] = AVG(ebd, args)
        if args.cuda != -1:
            train_data['avg_ebd'] = train_data['avg_ebd'].cuda(args.cuda)

        val_data['avg_ebd'] = train_data['avg_ebd']
        test_data['avg_ebd'] = train_data['avg_ebd']

    return train_data, val_data, test_data, vocab, vocab_counter


@hydra.main(config_path='conf', config_name='train')
def main(cfg):
    args = cfg
    cfg.auxiliary = cfg.auxiliary or []
    # if any('fewrel' in d['dataset'].get('name', '') for d in args['datasets_train_cfgs']):
    #     cfg.auxiliary += ['pos']
    cfg.wv_path = os.path.join(_get_tmp_dir(), 'bao')
    if not os.path.exists(cfg.wv_path):
        os.makedirs(cfg.wv_path)
    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, vocab, vocab_counter = load_dataset(args)

    # initialize model
    model = {}
    model["ebd"] = ebd.get_embedding(vocab, args)
    model["clf"] = clf.get_classifier(model["ebd"].ebd_dim, args)

    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        train_utils.train(train_data, val_data, model, args)

    elif args.mode == "finetune":
        # sample an example from each class during training
        way = args.way
        query = args.query
        shot = args.shot
        args.query = 1
        args.shot = 1
        args.way = args.n_train_class
        train_utils.train(train_data, val_data, model, args)
        # restore the original N-way K-shot setting
        args.shot = shot
        args.query = query
        args.way = way

    # testing on validation data: only for not finetune
    # In finetune, we combine all train and val classes and split it into train
    # and validation examples.
    if args.mode != "finetune":
        val_acc, val_std = train_utils.test(val_data, model, args,
                                            args.val_episodes)
    else:
        val_acc, val_std = 0, 0

    if args.save:
        out_dir = os.path.abspath(os.path.join(
            os.path.curdir,
            "saved-runs"))
        with open(os.path.join(out_dir, 'vocab_counter.pk'), "wb") as f:
            pickle.dump(vocab_counter, f, pickle.HIGHEST_PROTOCOL)
        torch.save(train_data['idf'], os.path.join(out_dir, 'idf.pt'))
        torch.save(train_data['iwf'], os.path.join(out_dir, 'iwf.pt'))
        torch.save(train_data['avg_ebd'], os.path.join(out_dir, 'avg_ebd.pt'))

    # test_acc, test_std = train_utils.test(test_data, model, args,
    #                                       args.test_episodes)

    # if args.result_path:
    #     directory = args.result_path[:args.result_path.rfind("/")]
    #     if not os.path.exists(directory):
    #         os.mkdirs(directory)

    #     result = {
    #         "test_acc": test_acc,
    #         "test_std": test_std,
    #         "val_acc": val_acc,
    #         "val_std": val_std
    #     }

    #     for attr, value in sorted(args.__dict__.items()):
    #         result[attr] = value

    #     with open(args.result_path, "wb") as f:
    #         pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
