import json
from typing import TextIO
from functools import partial
import click
import numpy as np
from scipy.stats import sem
import pandas as pd
from fewshot.bootstrap import bootstrap
from fewshot.bootstrap import ci
from fewshot.challenges.utils import get_gold_dataset
from . import score_utils as su


def statistics(a, estimator=np.mean, conf_interval=95, n_boot=1000, seed=0):
    """With 95% CI"""
    [ci_lower, ci_upper] = ci(
        bootstrap(
            a,
            func=estimator,
            n_boot=n_boot,
            seed=seed,
        ),
        conf_interval
    )
    stat = estimator(a)
    return {
        'stat': stat,
        'stat_ci_lower': stat - ci_lower,
        'stat_ci_upper': ci_upper - stat,
        'stat_ci_sem': sem(a, ddof=1) * 1.96,
        'std': np.std(a),
        'n': len(a),
    }


@click.command()
@click.option('--challenge_name', type=click.STRING, required=True)
@click.option(
    '--predictions',
    type=click.File('r'),
    help='Path to the file containing system predictions',
    required=True,
)
@click.option(
    '--output',
    '-o',
    type=click.File('w'),
    help='Output results to this file.',
)
@click.option('--by_way_shot', is_flag=True, default=False)
@click.option('--by_few', is_flag=True, default=False)
@click.option('--for_leaderboard', is_flag=True, default=False)
def score(
    challenge_name: str,
    predictions: TextIO,
    output: TextIO,
    by_way_shot: bool,
    by_few: bool,
    for_leaderboard: bool,
):
    """Score a predictions.json file."""
    gold_data = pd.DataFrame(get_gold_dataset(challenge_name))
    joined_data = su.join_predictions_and_gold(
        predictions=predictions,
        gold_data=gold_data,
    )
    df, metrics = su.score_joined_data(data=joined_data)
    if by_way_shot:
        df['shot'] = df.apply(lambda row: str(int(row['n_train'] / row['way']))
                              if row['balanced_train'] else '', axis=1)
        grouped = df.groupby(by=['dataset', 'way', 'shot'])['accuracy'].apply(partial(statistics, estimator=np.mean))
        grouped.index = grouped.index.set_names('stat', level=3)
        res = grouped
    elif by_few or for_leaderboard:
        df['few'] = df['n_train'].map(lambda v: v > 0)
        grouped = df.groupby(by=['dataset', 'few'])['accuracy'].apply(partial(statistics, estimator=np.mean))
        grouped.index = grouped.index.set_names('stat', level=2)
        ways = df.groupby(by=['dataset', 'few'])['way'].apply(lambda x: '/'.join(str(i) for i in sorted(x.unique())))
        res = pd.merge(
            grouped.reset_index(),
            ways.reset_index(),
            on=['dataset', 'few']
        ).set_index(['dataset', 'way', 'few', 'stat'])
    else:
        grouped = df.groupby(by=['dataset'])['accuracy'].apply(partial(statistics, estimator=np.mean))

        means = grouped.xs('stat', level=1)
        stds = grouped.xs('std', level=1)
        cis_upper = grouped.xs('stat_ci_upper', level=1)
        cis_lower = grouped.xs('stat_ci_lower', level=1)

        cis_lower.index = cis_lower.index + '_acc_ci_lower'
        cis_upper.index = cis_upper.index + '_acc_ci_upper'
        means.index = means.index + '_acc'
        stds.index = stds.index + '_acc_std'

        res = pd.concat([means, cis_upper, cis_lower, stds], axis=0)
        res.loc['overall_acc'] = means.mean()
        res.loc['overall_acc_std'] = stds.mean()
    if for_leaderboard:
        res = res.reset_index()
        res['few_string'] = res['few'].map(lambda v: 'few' if v else '0')
        res['name'] = res['dataset'] + '-' + res['few_string']
        accuracies = res[res.stat == 'stat']
        accuracies = accuracies.append([
            {'name': 'overall-0', 'accuracy': accuracies[~accuracies.few]['accuracy'].mean()},
            {'name': 'overall-few', 'accuracy': accuracies[accuracies.few]['accuracy'].mean()}
        ])
        uppers = res[res.stat == 'stat_ci_upper']
        uppers = uppers.assign(name=lambda x: x['name'] + '_ci_upper')
        lowers = res[res.stat == 'stat_ci_lower']
        lowers = lowers.assign(name=lambda x: x['name'] + '_ci_lower')
        res = pd.concat([accuracies, uppers, lowers], axis=0)
        res = res[['name', 'accuracy']].set_index('name')
        res = res['accuracy']
        print(type(res))
    if output:
        if for_leaderboard:
            # Add episode-level accuracy values under 'episode_accuracies' key
            res = json.loads(res.to_json())
            grouped = (
                df.groupby(by=['few', 'dataset'])[['task_id', 'accuracy']]
                .apply(lambda x: x.sort_values('task_id')['accuracy'].tolist())
                .reset_index(name='accuracies')
            )
            grouped['few_string'] = grouped['few'].map(lambda v: 'few' if v else '0')
            grouped['name'] = grouped['dataset'] + '-' + grouped['few_string']
            res['episode_accuracies'] = grouped.set_index('name')[['accuracies']].to_dict()['accuracies']
            json.dump(res, output)
        elif output.name.endswith('.json'):
            res.to_json(output)
        else:
            res.to_csv(output)
    else:
        pd.set_option("display.max_rows", None)
        print(res.sort_index())
