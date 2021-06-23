import logging
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
logger = logging.getLogger(__name__)


def plot_calibration_curve(
    data: pd.DataFrame,
    plot_dir: str,
    method_name: str,
    include_distractors: bool = False,
):
    """Plot calibration curves for tasks with binary questions.

    Note: Aggregates results across datasets.
    
    """
    # Binary questions only
    df = data[
        data.labels.map(lambda lst: (set(lst) == set(['0', '1'])))
        & data.score_predicted.notna()
    ]
    if not include_distractors:
        df = df[~df.is_distractor]
    if len(df) == 0:
        return
    df['n_neg'] = df.n_train_by_label.map(lambda d: d.get('0', 0))
    df['n_pos'] = df.n_train_by_label.map(lambda d: d.get('1', 0))

    def compute_calibration(
        df: pd.DataFrame,
        rescale: bool = True,
    ):
        prob_true, prob_pred = calibration_curve(
            y_true=df.label.map(int),
            y_prob=df.score_predicted,
            normalize=True,
            strategy='uniform',
            # strategy='quantile',
        )

        # Convert back to scores
        if rescale:
            logger.info(f'Rescaling {prob_pred} on [0, 1] to [{df.score_predicted.min()}, {df.score_predicted.max()}]...')
            prob_pred = np.interp(
                prob_pred,
                [0, 1],
                [df.score_predicted.min(), df.score_predicted.max()]
            )
            logger.info(f'...{prob_pred}')

        return pd.DataFrame({
            'prob_true': prob_true,
            'prob_pred': prob_pred,
        })

    probs = compute_calibration(df).assign(method=method_name)
    # Plot all
    grid = sns.relplot(
        x='prob_pred',
        y='prob_true',
        data=probs,
        kind='line',
        hue='method',
        markers=True,
    )
    subdir = os.path.join(plot_dir, 'calibration_curves')
    os.makedirs(subdir, exist_ok=True)
    grid.savefig(os.path.join(subdir, f'{method_name}-all.png'))
    probs.to_csv(os.path.join(subdir, f'{method_name}-all.csv'))
    plt.clf()
    sns.distplot(
        probs['prob_pred']
    ).get_figure().savefig(os.path.join(subdir, f'{method_name}-all-dist.png'))

    # Group by pos/neg
    probs = df.groupby(
        by=[
            'n_pos',
            'n_neg',
        ]
    ).apply(
        compute_calibration
    ).reset_index()
    probs['method'] = method_name
    grid = sns.relplot(
        x='prob_pred',
        y='prob_true',
        data=probs,
        kind='line',
        hue='method',
        col='n_pos',
        row='n_neg',
        markers=True,
    )
    subdir = os.path.join(plot_dir, 'calibration_curves')
    os.makedirs(subdir, exist_ok=True)
    grid.savefig(os.path.join(subdir, f'{method_name}-by-detailed.png'))
    probs.to_csv(os.path.join(subdir, f'{method_name}-by-detailed.csv'))
    plt.clf()
    sns.distplot(
        probs['prob_pred']
    ).get_figure().savefig(os.path.join(subdir, f'{method_name}-by-detailed-dist.png'))

    # Group by n training
    probs = df.assign(
        n_train_by_label=lambda x: x['n_train_by_label'].map(
            lambda d: sum(d.values())
        ),
    ).groupby(
        by=[
            'n_train_by_label',
        ]
    ).apply(
        compute_calibration
    ).reset_index()
    probs['method'] = method_name
    grid = sns.relplot(
        x='prob_pred',
        y='prob_true',
        data=probs,
        kind='line',
        hue='method',
        col='n_train_by_label',
        col_wrap=3,
        markers=True,
    )
    subdir = os.path.join(plot_dir, 'calibration_curves')
    os.makedirs(subdir, exist_ok=True)
    grid.savefig(os.path.join(subdir, f'{method_name}-by-n_train.png'))
    probs.to_csv(os.path.join(subdir, f'{method_name}-by-n_train.csv'))
    plt.clf()
    sns.distplot(
        probs['prob_pred']
    ).get_figure().savefig(os.path.join(subdir, f'{method_name}-by-n_train-dist.png'))
