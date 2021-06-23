import itertools
from pathlib import Path
import collections
import logging
import os
import json
from abc import ABC, abstractmethod
from typing import (
    Iterable, Sequence, Iterator, List, Type, Tuple, Union, Dict, Any, Optional
)
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import fewshot.challenges.utils
from ..utils import grouper, flatten
logger = logging.getLogger(__name__)

HERE = Path(__file__).absolute().parent
_DEFAULT_BATCH_SIZE = 10


def validate(query_y, labels):
    try:
        first_bad = next(y for y in query_y if str(y) not in labels)
        raise ValueError(f"Label {first_bad} not in allowed labels {labels}")
    except StopIteration:
        pass


def _parse_fit_and_predict_result(result):
    """Parse and infer whether fit_and_predict returns y or (y, scores)."""
    if len(result) > 1 and result[1] and not isinstance(result[1], str):
        # Scores object does not resemble a label prediction (always string)
        y = result[0]
        scores = result[1]
    else:
        y = result
        scores = None
    return y, scores


def _convert_fit_and_predict_result_to_predictions(
    query_y,
    scores,
    query_question_ids
):
    def _convert_score(s):
        if s is None:
            return s
        elif isinstance(s, collections.abc.Iterable):
            return list(float(v) for v in s)
        else:
            return float(s)
    if scores is None:
        scores = itertools.repeat(None)
    return {
        id: {
            'label': y,
            'score': _convert_score(score)
        }
        for y, score, id in zip(query_y, scores, query_question_ids)
    }


class Model(ABC):
    @abstractmethod
    def fit_and_predict(
        self,
        support_x: Iterable[Any],
        support_y: Iterable[str],
        target_x: Iterable[Any],
        metadata: Dict[str, Any] = None,
    ) -> Union[Sequence[str], Tuple[Sequence[str], Sequence[float]]]:
        "Return label predictions and scores for a fewshot task"
        pass


class Evaluator:
    def __init__(
        self,
        config_name: str,
        hash: str = None,
        ignore_verification: bool = False,
    ) -> None:
        """Fewshot evaluator.

        """
        self.config_name = config_name
        self.dataset = load_dataset(
            str(HERE.parent / 'hf_datasets_scripts' / 'challenge'),
            name=config_name,
            data_dir=os.path.join(HERE, 'conf'),
            split='test',
        )
        if not ignore_verification and hash is not None:
            h = fewshot.challenges.utils.get_challenge_hash(self.dataset)
            if h != hash:
                raise ValueError(fewshot.challenges.utils.wrong_hash_message.format(
                    hash=h,
                    expected_hash=hash,
                    challenge=config_name,
                ))
        self.task_splits = pd.DataFrame({
            'task_id': self.dataset['task_id'],
            'is_train': self.dataset['is_train']
        })

    @property
    def n_tasks(self):
        return self.task_splits['task_id'].nunique()

    def _get_example_data(self, id: str) -> dict:
        """Get the data for an example id."""
        return self.tasks['data'][id]

    def _get_group_example_data(self, data_group_id: str) -> Dict[
        str, dict
    ]:
        """Get mapping from example id to data for a data group."""
        return {
            e['example_id']: self._get_example_data(e['example_id'])
            for e in self.tasks['data_groups'][data_group_id]
        }

    def get_tasks(
        self,
        start_task_index: int = 0,
        stop_task_index: Optional[int] = None,
    ) -> Iterator[Tuple[
        List[dict],
        List[str],
        List[dict],
        Dict[str, Any],
    ]]:
        for i, (split, df) in enumerate(self.task_splits.groupby(['task_id'])):
            in_start_stop_range = (
                i >= start_task_index
                and (stop_task_index is None or i < stop_task_index)
            )
            if not in_start_stop_range:
                continue
            train_indices = df[df.is_train].index.tolist()
            if train_indices:
                train = self.dataset[df[df.is_train].index.tolist()]
            else:
                train = None
            test = self.dataset[df[~df.is_train].index.tolist()]
            if train:
                support_x = [
                    {
                        'dense': dense,
                        'sparse': sparse,
                        'txt': txt,
                    }
                    for dense, sparse, txt in zip(train['dense'], train['sparse'], train['txt'])
                ]
            else:
                support_x = []
            query_x = [
                {
                    'dense': dense,
                    'sparse': sparse,
                    'txt': txt,
                }
                for dense, sparse, txt in zip(test['dense'], test['sparse'], test['txt'])
            ]
            if train:
                support_y = train['label']
            else:
                support_y = []
            metadata = {
                'labels': test['labels'][0],
                'dataset': test['dataset'][0],
                'text_labels': dict(zip(test['text_labels'][0]['keys'], test['text_labels'][0]['values'])),
                'query_question_ids': test['question_id'],
                'unlabeled_store_kwargs': test['unlabeled_store_kwargs'][0] or None,
                'used_example_ids': [
                    x for x in (
                        (train['example_id'] if train else [])
                        + test['example_id']
                    ) if x
                ],
                'majority_class': test['majority_class'][0] or None,
            }
            # if test['unlabeled_store_kwargs'][0]:
            #     unlabeled_dataset = Store(
            #         **json.loads(test['unlabeled_store_kwargs'][0])
            #     ).store
            #     if 'flex.example_id' in unlabeled_dataset.column_names:
            #         # Filter out example ids from train/test.
            #         used_example_ids = set([
            #             s for s in train['example_id'] + test['example_id']
            #             if s
            #         ])
            #         print(f'unlabeled dataset going from {len(unlabeled_dataset)} to...')
            #         unlabeled_dataset = unlabeled_dataset.filter(
            #             lambda x: x['flex.example_id'] not in used_example_ids
            #         )
            #         print(f'...{len(unlabeled_dataset)}')
            #     metadata['unlabeled'] = unlabeled_dataset
            yield support_x, support_y, query_x, metadata

    def get_model_predictions(
        self,
        model: Type[Model],
        start_task_index: int = 0,
        stop_task_index: Optional[int] = None,
        batched: bool = False,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        skip_validation: bool = False,
    ) -> Dict[str, Dict[str, Union[str, float]]]:
        """Run a model with a fit_and_predict method."""
        predictions = {}
        if not batched:
            batch_size = None
        n_tasks = (stop_task_index or self.n_tasks) - start_task_index
        with tqdm(total=n_tasks) as pbar:
            if not batched:
                for support_x, support_y, query_x, metadata in self.get_tasks(
                    start_task_index=start_task_index,
                    stop_task_index=stop_task_index,
                ):
                    query_y, scores = _parse_fit_and_predict_result(
                        model.fit_and_predict(
                            support_x=support_x,
                            support_y=support_y,
                            target_x=query_x,
                            metadata=metadata,
                        )
                    )
                    if not skip_validation:
                        validate(query_y, metadata['labels'])
                    predictions.update(
                        _convert_fit_and_predict_result_to_predictions(
                            query_y=query_y,
                            scores=scores,
                            query_question_ids=metadata['query_question_ids']
                        )
                    )
                    pbar.update(1)
            else:
                for batch in grouper(
                    batch_size,
                    self.get_tasks(
                        start_task_index=start_task_index,
                        stop_task_index=stop_task_index,
                    )
                ):
                    support_x, support_y, query_x, metadata = zip(*(b for b in batch if b is not None))
                    n_tasks_in_batch = len(support_x)
                    query_y, scores = _parse_fit_and_predict_result(
                        model.fit_and_predict(
                            support_x=support_x,
                            support_y=support_y,
                            target_x=query_x,
                            metadata=metadata,
                        )
                    )
                    try:
                        query_y = flatten(query_y)
                        scores = flatten(scores) if scores is not None else None
                    except TypeError:
                        # Already flattened
                        pass
                    query_question_ids_flat = flatten(m['query_question_ids'] for m in metadata)
                    if not skip_validation:
                        validate(query_y, metadata['labels'])
                    predictions.update(
                        _convert_fit_and_predict_result_to_predictions(
                            query_y=query_y,
                            scores=scores,
                            query_question_ids=query_question_ids_flat,
                        )
                    )
                    pbar.update(n_tasks_in_batch)
            return predictions

    def save_model_predictions(
        self,
        model: Type[Model],
        save_path: str = 'predictions.json',
        start_task_index: int = 0,
        stop_task_index: Optional[int] = None,
        batched: bool = False,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        skip_validation: bool = False,
    ) -> None:
        """Run model and save predictions."""
        predictions = self.get_model_predictions(
            model=model,
            start_task_index=start_task_index,
            stop_task_index=stop_task_index,
            batched=batched,
            batch_size=batch_size,
            skip_validation=skip_validation,
        )
        with open(save_path, 'w') as fp:
            json.dump(predictions, fp)
