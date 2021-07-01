from typing import Union, TextIO, Optional, Tuple, List, Iterable
import collections
import json
import functools
import operator
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _prod(vs: Iterable[float]) -> float:
    """Product function."""
    return functools.reduce(operator.mul, vs, 1)


def join_predictions_and_gold(
    predictions: Union[TextIO, str],
    gold_data: pd.DataFrame,
) -> pd.DataFrame:
    gold_data['n_train_by_label'] = gold_data['train_labels'].map(
        lambda vs: collections.Counter([str(v) for v in vs])
    )
    gold_data['prob_subsampled'] = gold_data['probs_subsampled'].map(_prod)

    if isinstance(predictions, str):
        predictions = open(predictions, 'r')
    preds = pd.DataFrame.from_dict(json.load(predictions), orient='index')
    predictions.close()

    # Sanity check same question ids
    assert set(gold_data['question_id']) == set(preds.index)

    data = pd.merge(
        left=gold_data,
        right=preds,
        suffixes=('', '_predicted'),
        left_on='question_id',
        right_index=True,
        how='left',
        validate='1:1',
    )
    data = data.astype({c: str for c in ('label', 'label_predicted')})
    return data


def score_joined_data(
    data: pd.DataFrame,
    distractors_sample_weight: float = 0.,
    method_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    def compute_metrics(df):
        labels = json.loads(df['labels'].iloc[0])
        n_distractors_or_not = df['is_distractor'].value_counts()
        sample_weight = df['is_distractor'].map(lambda v: (
            distractors_sample_weight / n_distractors_or_not[True] if v
            else 1 / n_distractors_or_not[False]
        ))
        result = {
            'accuracy': accuracy_score(
                df['label'],
                df['label_predicted'],
                sample_weight=sample_weight,
            ),
            'micro_f1': f1_score(
                df['label'],
                df['label_predicted'],
                labels=labels,
                average='micro',
                sample_weight=sample_weight,
            ),
            'roc_auc': roc_auc_score(
                df['label'],
                df['score'].tolist(),
                labels=labels,
                sample_weight=sample_weight,
            ) if (
                'score' in df.columns
                and not df['score'].isnull().values.any()
                and len(labels) == 2
            ) else None,
        }
        return pd.Series(result, name='metrics')

    # Serialize task label info and compute grouped metrics
    task_stats = data.assign(
        n_train_by_label=lambda x: x['n_train_by_label'].map(
            lambda d: json.dumps({k: d[k] for k in sorted(d)})
        ),
        labels=lambda x: x['labels'].map(lambda lst: json.dumps(sorted(lst))),
    ).groupby(
        by=[
            'dataset',
            'labels',
            'n_train_by_label',
            'prob_subsampled',
            'task_id',
            'node',
        ]
    ).apply(
        compute_metrics
    )
    metrics = list(task_stats.columns)
    # Deserialize task label info
    task_stats = task_stats.reset_index().assign(
        labels=lambda x: x['labels'].map(json.loads),
    )

    task_stats['method'] = method_name
    task_stats['n_train'] = task_stats['n_train_by_label'].map(
        lambda d: sum(json.loads(d).values())
    )
    task_stats['balanced_train'] = task_stats['n_train_by_label'].map(
        lambda d: len(set(json.loads(d).values())) <= 1
    )
    task_stats['way'] = task_stats['labels'].map(len)

    return task_stats, metrics
