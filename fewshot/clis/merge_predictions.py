from typing import Iterable
import os
from io import StringIO
import json
import click


@click.command()
@click.argument(
    'predictions',
    type=click.File('r'),
    required=True,
    nargs=-1,
)
@click.argument(
    'output',
    type=click.Path(dir_okay=False),
    nargs=1,
)
def merge(predictions: Iterable[StringIO], output: StringIO):
    """Merge predictions files into a single predictions.json file."""
    if os.path.exists(output):
        raise click.BadParameter('OUTPUT cannot already exist.')
    result = {}
    for f in predictions:
        result.update({
            k: v for k, v in json.load(f).items()
            if v.get('label', '') not in ['', -1]
        })
    with open(output, 'w') as f:
        json.dump(result, f)
