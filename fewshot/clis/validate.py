"""Validate correctness of a predictions file."""
from typing import TextIO
import json
import click
import pandas as pd
import fewshot


@click.command()
@click.option('--challenge_name', type=click.STRING, required=True)
@click.option(
    '--predictions',
    type=click.File('r'),
    help='Path to the file containing system predictions',
    required=True,
)
def validate(challenge_name: str, predictions: TextIO):
    """Score a predictions.json file."""
    preds = pd.DataFrame.from_dict(json.load(predictions), orient='index')
    c = fewshot.make_challenge(challenge_name)
    r = c.dataset.filter(lambda d: not d, input_columns='is_train')
    r.map(lambda row: fewshot.challenges.eval.validate([preds.loc[row['question_id']]['label']], row['labels']))
    print('Success')