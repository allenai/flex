import os
import collections
import multiprocessing as mp
import json
import functools
import operator
from typing import Sequence, Tuple, List, Collection, Iterable, Optional, TextIO, Union
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import click
from omegaconf import OmegaConf
import logging
from fewshot.challenges.utils import get_gold_dataset
from .plot_calibration_curve import plot_calibration_curve as plot_calibration_curve_f
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _prod(vs: Iterable[float]) -> float:
    """Product function."""
    return functools.reduce(operator.mul, vs, 1)


def _weighted_mean(x, **kws):
    """"Weighted mean estimator.

    Input values v are augmented as tuples (v, w) where w is the weight.

    From https://github.com/mwaskom/seaborn/issues/722

    """
    val, weight = map(np.asarray, zip(*x))
    return (val * weight).sum() / weight.sum()


@click.command()
@click.argument(
    'data_dirs',
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.argument(
    'plot_dir',
    type=click.Path(exists=False, file_okay=False),
)
@click.option(
    '--distractors-sample-weights',
    '-w',
    default=[0, 1],
    multiple=True,
    type=click.FLOAT,
)
@click.option(
    '--plot_col_wrap',
    default=4,
    type=click.INT,
)
@click.option(
    '--weight-prob-subsampled/--no-weight-prob-subsampled',
    default=True,
    help=(
        'Correct for sampling bias due to nodes with more examples sampling'
        ' a specific combination less frequently.'
    ),
)
@click.option(
    '--plot-calibration-curve/--no-plot-calibration-curve',
    default=False,
    help='Plot calibration curve for binary tasks.',
)
@click.option(
    '--challenge_name',
    help='Challenge name',
    default=None,
    type=click.STRING,
)
@click.option(
    '--method_names',
    multiple=True,
    help='Optional list of method names corresponding to data directories',
    default=None,
    type=click.STRING,
)
def plot(
    data_dirs: Sequence[str],
    plot_dir: str,
    distractors_sample_weights: Sequence[float],
    plot_col_wrap: int,
    weight_prob_subsampled: bool,
    plot_calibration_curve: bool,
    challenge_name: Optional[str] = None,
    method_names: Sequence[str] = (),
) -> None:
    """Score and plot the results of a set of runs.

    Args:
        data_dirs: Path(s) to hydra output directories to score and plot.
        plot_dir: Directory to save results. Created if it does not exist.
        distractors_sample_weight: Weight w for how to weight distractors vs
            the actual examples. Overall, the weight these 2 categories
            will be distributed w:1.
        weight_prob_subsampled: Correct for sampling bias due to nodes with
            more examples sampling a specific combination less frequently.

    """
    for w in distractors_sample_weights:
        logger.info(
            '---- '
            f'Starting distractors_sample_weight={w}'
            f', weight_prob_subsampled={weight_prob_subsampled}'
            ' ----'
        )
        _score_one_sample_weight(
            data_dirs=data_dirs,
            plot_dir=os.path.join(
                plot_dir,
                (
                    f'distractors_weight_{w:.2f}'
                    + (
                        '-weight_prob_subsampled'
                        if weight_prob_subsampled else ''
                    )
                )
            ),
            distractors_sample_weight=w,
            weight_prob_subsampled=weight_prob_subsampled,
            plot_col_wrap=plot_col_wrap,
            plot_calibration_curve=plot_calibration_curve,
            challenge_name=challenge_name,
            method_names=method_names,
        )


def _score_one_sample_weight(
    data_dirs: Sequence[str],
    plot_dir: str,
    distractors_sample_weight: float,
    weight_prob_subsampled: bool,
    plot_col_wrap: int,
    plot_calibration_curve: bool,
    challenge_name: Optional[str] = None,
    method_names: Optional[List[str]] = None,
):
    """Helper function for .score for one sample weight."""
    os.makedirs(plot_dir, exist_ok=True)

    def is_model_output_dir(filenames: Collection[str]):
        return 'predictions.json' in filenames

    # Get all directories including subdirectories for multiruns
    dirs = []
    for d in data_dirs:
        contents = set(os.listdir(d))
        if is_model_output_dir(contents):
            dirs.append(d)
        else:
            for e in contents:
                subdir = os.path.join(d, e)
                if os.path.isdir(subdir):
                    subcontents = set(os.listdir(subdir))
                    if is_model_output_dir(subcontents):
                        dirs.append(subdir)

    # Possibly infer method names / challenge name
    if not method_names:
        method_names_out = []
        for d in dirs:
            overrides_path = os.path.join(d, '.hydra', 'overrides.yaml')
            if os.path.exists(overrides_path) and method_names is None:
                # Infer from hydra overrides
                method_names_out.append(','.join(OmegaConf.load(overrides_path)))
            else:
                # Use directory path
                method_names_out.append(d)
    else:
        method_names_out = method_names
    if challenge_name is None:
        challenge_name_out = None
        for d in dirs:
            _challenge_name = OmegaConf.load(
                os.path.join(d, '.hydra', 'config.yaml')
            ).challenge
            assert challenge_name_out is None or _challenge_name == challenge_name_out
            challenge_name_out = _challenge_name
    else:
        challenge_name_out = challenge_name
    method_names = method_names_out
    challenge_name = challenge_name_out

    gold_data = pd.DataFrame(get_gold_dataset(challenge_name))

    # Score directories
    logger.info('Loading and scoring results...')
    with mp.Pool() as pool:
        # TODO: Check that all golds are the same?
        dfs, metrics_lst = zip(*pool.starmap(
            _score_one_method,
            (
                (
                    os.path.join(d, 'predictions.json'),
                    gold_data,
                    distractors_sample_weight,
                    plot_dir,
                    m,
                    plot_calibration_curve
                ) for d, m in zip(dirs, method_names)
            )
        ))
    metrics = metrics_lst[0]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    metrics.append('way')

    # Add subsample probabilities to metrics for weighted statistic est.
    augmented_suffix = '_with_inverse_prob_subsampled'
    if weight_prob_subsampled:
        df = df.assign(**{
            f'{m}{augmented_suffix}': lambda x, m=m: x.apply(
                lambda row: (row[m], 1 / row['prob_subsampled']),
                axis=1,
            ) for m in metrics
        })

    # Make plots
    for balanced in [None, True, False]:
        title_attributes = [
            ('distractors_weight', f'{distractors_sample_weight:.2f}'),
            ('correct_for_subsampling', f'{weight_prob_subsampled}'),
        ]
        if balanced is None:
            df_ = df
            plot_subdir = plot_dir
        else:
            df_ = df[df.balanced_train == balanced]
            title_attributes.append(('balanced', f'{balanced}'))
            plot_subdir = os.path.join(plot_dir, f'balanced={balanced}')
            os.makedirs(plot_subdir, exist_ok=True)
        plot_title_ending = ', '.join(f'{k}={v}' for k, v in title_attributes)
        if balanced is not None and df_.size == 0:
            logger.warning(f'No data points for balanced={balanced}')
            continue
        for metric in metrics:
            grid = sns.catplot(
                data=df_,
                x='n_train',
                y=f'{metric}{augmented_suffix}' if weight_prob_subsampled else metric,
                kind='point',
                hue='method',
                hue_order=sorted(df_['method'].unique()),
                col='dataset',
                col_order=sorted(df_['dataset'].unique()),
                col_wrap=min(plot_col_wrap, len(df_['dataset'].unique())),
                dodge=True,
                margin_titles=False,
                estimator=_weighted_mean if weight_prob_subsampled else np.mean,
                orient='v',
            )
            grid.set_xlabels('Number of training examples')
            grid.set_ylabels(f'{metric.capitalize()} (mean)')
            # Add super title and make space for it
            grid.fig.suptitle(f'{metric.capitalize()} ({plot_title_ending})')
            plt.subplots_adjust(top=0.9)
            outp = os.path.join(plot_subdir, f'{metric}.png')
            logger.info(f'Saving {outp}')
            grid.savefig(outp)
            plt.close(grid.fig)

        # Save count statistics
        df_[df_.method == df_['method'].unique()[0]].groupby(
            ['dataset', 'n_train', 'n_train_by_label']
        )['node'].nunique().reset_index().to_csv(
            os.path.join(plot_subdir, 'num_per_node.csv'),
            index=False,
        )

    # Special plots for when n_labels=2
    df_binary = df[df.labels.map(len) == 2].assign(
        n_train_0=lambda x: x['n_train_by_label'].map(
            lambda v: json.loads(v).get('0', 0)
        ),
        n_train_1=lambda x: x['n_train_by_label'].map(
            lambda v: json.loads(v).get('1', 0)
        ),
    )
    if len(df_binary) > 0:
        for metric in metrics:
            for dataset, df_ in df_binary.groupby('dataset'):
                grid = sns.catplot(
                    data=df_,
                    x='n_train_1',
                    y=f'{metric}{augmented_suffix}' if weight_prob_subsampled else metric,
                    kind='point',
                    hue='method',
                    hue_order=sorted(df_['method'].unique()),
                    col='n_train_0',
                    col_wrap=min(
                        plot_col_wrap, len(df_['n_train_0'].unique())
                    ),
                    dodge=True,
                    margin_titles=False,
                    estimator=(
                        _weighted_mean if weight_prob_subsampled
                        else np.mean
                    ),
                    orient='v',
                )
                grid.set_titles(
                    col_template='Negative training examples = {col_name}',
                )
                grid.set_xlabels('Number of positive training examples')
                grid.set_ylabels(f'{metric.capitalize()} (mean)')
                # Add super title and make space for it
                grid.fig.suptitle(metric.capitalize() + plot_title_ending)
                plt.subplots_adjust(top=0.9)
                outp = os.path.join(plot_dir, f'binary-{dataset}-{metric}.png')
                logger.info(f'Saving {outp}')
                grid.savefig(outp)
                plt.close(grid.fig)

    # Save summary statistics
    stats = df.groupby(by=['dataset', 'method', 'n_train', 'way']).agg(
        ['mean', 'sem', 'count']
    ).reset_index()
    logger.info(stats)
    # Flatten nested metrics
    stats.columns = [
        ':'.join(c for c in col if c) for col in stats.columns.to_flat_index()
    ]
    outp = os.path.join(plot_dir, 'stats.csv')
    logger.info(f'Saving {outp}')
    stats.to_csv(outp, index=False)


def _score_one_method(
    predictions: Union[TextIO, str],
    gold_data: pd.DataFrame,
    distractors_sample_weight: float,
    plot_dir: Optional[str] = None,
    method_name: Optional[str] = None,
    plot_calibration_curve: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Get metrics by dataset and training set size.

    Args:
        distractors_sample_weight: Sample weight of one distractor, for
            computing weighted roc_auc during scoring.

    Returns:
        data: Dataframe of scored tasks.
        metrics: List of metric names.

    """
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

    if plot_calibration_curve:
        plot_calibration_curve_f(data, plot_dir, method_name)

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


if __name__ == '__main__':
    plot()
