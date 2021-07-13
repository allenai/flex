"""

Simulates bootstrap CIs to compute coverage & width for sample size simulations.

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from fewshot.bootstrap import bootstrap, ci
import random

import multiprocessing


def get_test_examples_given_budget_hours(budget_hours: float, n_episodes: int) -> int:
    return int((budget_hours - 1166.0522286 * n_episodes) / (7.5251996952 * n_episodes))


def statistics(a, estimator=np.mean, conf_interval=95, n_boot=1000):
    """With 95% CI"""
    boot = bootstrap(a, func=estimator, n_boot=n_boot)
    [ci_lower, ci_upper] = ci(boot, conf_interval)
    return {
        'mean': np.mean(boot),
        'lower': ci_lower,
        'upper': ci_upper,
        'moe': abs((ci_upper - ci_lower)) / 2,  # sem(a, ddof=1) * 1.96,
        'std': np.std(boot),
        'n': len(a)
    }


def simulate_leaderboard_submission(n_episodes: int, n_test_examples: int, acc: float) -> dict:
    episode_accuracies = []
    for episode_id in range(n_episodes):
        episode_acc = acc + np.random.normal(loc=0, scale=EPISODE_NOISE)
        episode_outcomes = [1 if random.uniform(0, 1) <= episode_acc else 0 for i in range(n_test_examples)]
        episode_accuracies.append(np.mean(episode_outcomes))

    stats = statistics(a=episode_accuracies)
    stats['bias'] = abs(stats['mean'] - acc)
    return stats


############################################################


BUDGET_HOURS = [24, 36, 48, 60, 72, 84]
EPISODES = [5, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
PROBS_OF_CORRECT_ANSWER = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95]
EPISODE_NOISE = 0.05
NUM_REPS = 1000
SEED = 0
random.seed(SEED)

batches = []
for acc in tqdm(PROBS_OF_CORRECT_ANSWER):
    for n_episodes in tqdm(EPISODES):
        for budget_hours in tqdm(BUDGET_HOURS, leave=False):
            budget_in_seconds = budget_hours * 60 * 60
            n_test_examples = get_test_examples_given_budget_hours(
                budget_hours=budget_in_seconds, n_episodes=n_episodes)

            # given a small enough budget, basically Dtest < 1.  In this case, don't even bother w/ simulation.
            if n_test_examples < 1:
                continue

            batches.append((acc, budget_hours, n_episodes, n_test_examples))


def do_work(batch):
    acc, budget_hours, n_episodes, n_test_examples = batch

    sim_results = []
    for _ in tqdm(range(NUM_REPS)):
        stats = simulate_leaderboard_submission(n_episodes=n_episodes, n_test_examples=n_test_examples, acc=acc)
        sim_results.append(stats)

    # calculate noise in Boot() procedure for this config
    ci_correctness = np.mean([stats['lower'] <= acc <= stats['upper'] for stats in sim_results])
    typical_ci_width = np.mean([stats['upper'] - stats['lower'] for stats in sim_results])
    mean_bias = np.mean([stats['bias'] for stats in sim_results])
    mean2_bias = np.mean([stats['bias'] ** 2 for stats in sim_results])

    return (acc, budget_hours, n_episodes, n_test_examples, ci_correctness, typical_ci_width, mean_bias, mean2_bias)


with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as mp:
    plot_results = mp.map(do_work, batches)


all_df = pd.DataFrame(plot_results, columns=['acc', 'budget_hours', 'n_episodes',
                      'n_test_examples', 'ci_correctness', 'ci_width', 'mean_bias', 'mean2_bias'])
all_df.to_csv(f'episodes_vs_test_examples_1000_noise_05.csv', index=False)
