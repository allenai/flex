from pathlib import Path
from typing import Iterable, Tuple, Sequence, Dict, Any, Optional
import pickle
import logging
import torch
from omegaconf import OmegaConf, DictConfig
from torchtext.vocab import Vocab, Vectors
from fewshot import Model
from bao.dataset.loader import _data_to_nparray
from bao.dataset.utils import to_tensor
from bao.embedding import factory as ebd
from bao.classifier import factory as clf
from bao.dataset import stats
from bao.train.regular import test_one
from train import _format_fewrel, _tokenize_and_truncate
logger = logging.getLogger(__name__)


class Bao(Model):
    def __init__(
        self,
        ebd_path: str,
        clf_path: str,
        vocab_counter_path: str,
        wv_path: str,
        word_vector: str,
        avg_ebd_path: str,
        args: DictConfig,
        idf_path: Optional[str] = None,
        iwf_path: Optional[str] = None,
        cuda: Optional[int] = None,
    ):
        self.vectors = Vectors(word_vector, cache=Path(wv_path).expanduser())
        with open(Path(vocab_counter_path).expanduser(), 'rb') as f:
            vocab_counter = pickle.load(f)
        self.vocab = Vocab(vocab_counter, vectors=self.vectors,
                           specials=['<pad>', '<unk>'], min_freq=5)
        # self.args = OmegaConf.load(Path(args_path).expanduser())
        self.args = args
        self.args.auxiliary = self.args.auxiliary or []
        if cuda is not None:
            self.args.cuda = cuda
        if cuda == -1:
            self.device = 'cpu'
        else:
            self.device = f'cuda:{self.args.cuda}'
        self.avg_ebd = torch.load(Path(avg_ebd_path).expanduser(), map_location=self.device)
        if idf_path:
            self.idf = torch.load(Path(idf_path).expanduser(), map_location=self.device)
        if iwf_path:
            self.iwf = torch.load(Path(iwf_path).expanduser(), map_location=self.device)

        self.ebd_path = ebd_path
        self.clf_path = clf_path
        self.model = {}

    def _hydrate(self):
        if 'ebd' not in self.model:
            self.model["ebd"] = ebd.get_embedding(self.vocab, self.args)
            self.model['ebd'].load_state_dict(torch.load(Path(self.ebd_path).expanduser(), map_location=self.device))
            self.model['ebd'].eval()
        if 'clf' not in self.model:
            self.model["clf"] = clf.get_classifier(self.model["ebd"].ebd_dim, self.args)
            self.model['clf'].set_way_shot(way=self.args.way, shot=self.args.shot)
            self.model['clf'].load_state_dict(torch.load(Path(self.clf_path).expanduser(), map_location=self.device))
            self.model['clf'].eval()

    def _set_way_shot(self, way: int, support_y: Iterable[str]):
        from collections import Counter
        counts = Counter(support_y)
        shot = next(iter(counts.values()))
        assert all(n == shot for n in counts.values()), 'Bao does not handle unbalanced'
        # self.args.shot = shot
        # self.args.way = way
        if (way != self.args.way or shot != self.args.shot) and 'clf' in self.model:
            raise Exception('Unable to change shot/way after hydration')
        self.args.way = way
        self.args.shot = shot

    def _get_test_epoch(self, support, query, source_classes):
        # ----- Adapted from bao.dataset.parallel_sampler.ParallelSampler.worker
        if self.args.embedding in ['idf', 'meta', 'meta_mlp']:
            # compute inverse document frequency over the meta-train set
            idf = stats.get_idf(support, source_classes)
            support['idf'] = idf
            query['idf'] = idf

        if self.args.embedding in ['iwf', 'meta', 'meta_mlp']:
            # compute SIF over the meta-train set
            iwf = stats.get_iwf(support, source_classes)
            support['iwf'] = iwf
            query['iwf'] = iwf

        # ---- Adapted from bao.dataset.parallel_sampler.ParallelSampler.get_epoch
        # convert to torch.tensor
        support = to_tensor(support, self.args.cuda, ['raw', 'vocab_size'])
        query = to_tensor(query, self.args.cuda, ['raw', 'vocab_size'])
        # support = to_tensor(support, self.args.cuda, ['raw', 'vocab_size', 'head', 'tail'])
        # query = to_tensor(query, self.args.cuda, ['raw', 'vocab_size', 'head', 'tail'])

        if self.args.meta_w_target:
            if self.args.meta_target_entropy:
                w = stats.get_w_target(
                    support, support['vocab_size'],
                    self.avg_ebd, self.args.meta_w_target_lam)
            else:  # use rr approxmation (this one is faster)
                w = stats.get_w_target_rr(
                    support, support['vocab_size'],
                    self.avg_ebd, self.args.meta_w_target_lam)
            support['w_target'] = w.detach()
            query['w_target'] = w.detach()

        support['is_support'] = True
        query['is_support'] = False

        return support, query

    def fit_and_predict(
        self,
        support_x: Iterable[str],
        support_y: Iterable[str],
        target_x: Iterable[str],
        metadata: Dict[str, Any] = None,
    ) -> Tuple[Sequence[str], Sequence[float]]:
        string_labels = metadata['labels']
        label_to_id_map = {l: i for i, l in enumerate(string_labels)}
        id_to_label_map = {v: k for k, v in label_to_id_map.items()}

        # Infer and set task type
        self._set_way_shot(way=len(metadata['labels']), support_y=support_y)
        self._hydrate()
        # if 'head_tokens' in support_x[0]:
        if False:
            self.args.auxiliary = ['pos']
            S_data = [_format_fewrel(d) for d in support_x]
            S_query = [_format_fewrel(d) for d in target_x]
        else:
            self.args.auxiliary = []
            # tokenizing text
            S_data = [{'text': _tokenize_and_truncate(doc['txt'])} for doc in support_x]
            # TODO: Check label not used
            Q_data = [{'text': _tokenize_and_truncate(doc['txt'])} for doc in target_x]
        S_data = [{**d, 'label': label_to_id_map[y]} for d, y in zip(S_data, support_y)]
        # TODO: Check label not used
        Q_data = [{**d, 'label': -999} for d in Q_data]
        S_data = _data_to_nparray(S_data, self.vocab, self.args, allow_delete=False)
        Q_data = _data_to_nparray(Q_data, self.vocab, self.args, allow_delete=False)
        for data in [S_data, Q_data]:
            try:
                data['idf'] = self.idf
            except AttributeError:
                pass
            try:
                data['iwf'] = self.iwf
            except AttributeError:
                pass

        # stats.precompute_stats(, {}, Q_data, args)
        task = self._get_test_epoch(S_data, Q_data, sorted(id_to_label_map))
        _, pred = test_one(task, self.model, None, return_pred=True)
        pred_is = torch.argmax(pred, dim=1)
        pred_labels = [id_to_label_map[i.item()] for i in pred_is]
        pred_logits = torch.gather(pred, 1, pred_is.unsqueeze(-1))
        return pred_labels, pred_logits.flatten().tolist()
