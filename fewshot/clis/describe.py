from typing import Optional
import logging
from hydra.utils import instantiate
import pandas as pd
import click
from fewshot.challenges.utils import get_gold_dataset
from fewshot.challenges import registry


def convert_dataset_name(store_name):
    """Parse {dataset}-{split} dataset name format."""
    dataset, *split = store_name.rsplit('-')
    if split:
        split = split[0]
    else:
        split = None
    return dataset, split


@click.command()
@click.option('--challenge_name', type=click.STRING, required=True)
@click.option('--output', '-o', type=click.STRING)
def describe(challenge_name: str, output: Optional[str]):
    # Pretty print
    logging.disable(logging.WARNING)
    pd.set_option('display.max_colwidth', 0)
    pd.set_option('display.max_columns', None)

    # Statistics about source datasets used in the challenge
    challenge_spec = registry.get_spec(challenge_name)
    stores = {}
    stores['test'] = [
        instantiate(cfg).labeled_store
        for cfg in challenge_spec.metadatasampler.datasets
    ]
    stores['train'] = [
        instantiate(cfg) for cfg in (challenge_spec.train_stores or {}).values()
    ]
    stores['val'] = [
        instantiate(cfg) for cfg in (challenge_spec.val_stores or {}).values()
    ]
    dataset_stats = []
    for challenge_split in stores:
        for store in stores[challenge_split]:
            label_col = store.label
            labels = sorted(set(store.store[label_col]))
            dataset_name, dataset_split = convert_dataset_name(store.name)
            dataset_stats.append({
                'challenge_split': challenge_split,
                'dataset': dataset_name,
                'dataset_split': dataset_split,
                'n_labels': len(labels),
                'n_examples': len(store.store),
                'labels': labels,
                'example': store.store[0]['flex.txt'],
            })
    df = pd.DataFrame(dataset_stats)
    df = df.set_index(['challenge_split', 'dataset'])
    print(df)
    if output:
        df.to_csv(f'{output}.datasets.csv')

    # Statistics about sampled tasks used in the challenge
    gold_data = pd.DataFrame(get_gold_dataset(challenge_name))
    gold_data = gold_data.assign(
        split=lambda x: x['dataset'].map(lambda s: convert_dataset_name(s)[1]),
        dataset=lambda x: x['dataset'].map(lambda s: convert_dataset_name(s)[0])
    )
    task_counts = gold_data.groupby(['dataset'])['task_id'].nunique()
    task_counts.name = 'Number of challenge tasks by dataset'
    print(task_counts)
    if output:
        task_counts.to_csv(f'{output}.tasks.csv')

    print(f"Total tasks in challenge: {gold_data['task_id'].nunique()}")
    print(f"Mean episode test set size: {gold_data.groupby(['task_id']).size().mean():.2f}")
