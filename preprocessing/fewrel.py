from collections import defaultdict
import json
import os
from pathlib import Path
from datasets import load_dataset

# Download and unzip BAO_DATA_DIR from https://people.csail.mit.edu/yujia/files/distributional-signatures/data.zip
_DIR = os.environ.get('BAO_DATA_DIR', Path('~') / 'Downloads' / 'data')
dataset = load_dataset(
    'few_rel',
    split='train_wiki+val_wiki'
)

text_to_properties = {
    (
        ' '.join(d['tokens']),
        ' '.join(d['tokens'][i].lower() for i in d['head']['indices'][0]),
        ' '.join(d['tokens'][i].lower() for i in d['tail']['indices'][0]),
    ): d['relation'] for d in dataset
}

mapping = defaultdict(set)
with open((Path(_DIR) / 'fewrel.json').expanduser(), 'r') as f:
    for line in f:
        d = json.loads(line)
        try:
            wikidata_property = text_to_properties[(
                d['raw'],
                ' '.join(d['text'][d['head'][0]:d['head'][1]+1]),
                ' '.join(d['text'][d['tail'][0]:d['tail'][1]+1]),
            )]
            mapping[d['label']].add(wikidata_property)
        except KeyError:
            pass
# Infer label to wikidata property from output
print(mapping)
