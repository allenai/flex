import json
from pathlib import Path
from typing import List, Container
import datasets

# From https://github.com/YujiaBao/Distributional-Signatures/blob/master/src/dataset/loader.py
_URL = 'https://people.csail.mit.edu/yujia/files/distributional-signatures/data.zip'
_labels = [
    'Amazon_Instant_Video',
    'Apps_for_Android',
    'Automotive',
    'Baby',
    'Beauty',
    'Books',
    'CDs_and_Vinyl',
    'Cell_Phones_and_Accessories',
    'Clothing_Shoes_and_Jewelry',
    'Digital_Music',
    'Electronics',
    'Grocery_and_Gourmet_Food',
    'Health_and_Personal_Care',
    'Home_and_Kitchen',
    'Kindle_Store',
    'Movies_and_TV',
    'Musical_Instruments',
    'Office_Products',
    'Patio_Lawn_and_Garden',
    'Pet_Supplies',
    'Sports_and_Outdoors',
    'Tools_and_Home_Improvement',
    'Toys_and_Games',
    'Video_Games'
]
_classes = {
    'train': [2, 3, 4, 7, 11, 12, 13, 18, 19, 20],
    'val': [1, 22, 23, 6, 9],
    'test': [0, 5, 14, 15, 8, 10, 16, 17, 21],
}


class Amazon(datasets.GeneratorBasedBuilder):
    VERSION = datasets.utils.Version('0.0.1')

    def _info(self):
        return datasets.DatasetInfo(
            description='TODO',
            features=datasets.Features({
                'text': datasets.Value('string'),
                'label': datasets.Value('string'),
            }),
        )

    def _split_generators(self, download_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        extracted_dir = download_manager.download_and_extract(_URL)
        filepath = Path(extracted_dir) / 'data' / 'amazon.json'
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                'filepath': filepath,
                'label_indices': set(_classes['train']),
            }),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                'filepath': filepath,
                'label_indices': set(_classes['val']),
            }),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                'filepath': filepath,
                'label_indices': set(_classes['test']),
            }),
        ]

    def _generate_examples(self, filepath: str, label_indices: Container[int]):
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                if d['label'] in label_indices:
                    yield i, {
                        'text': d['raw'],
                        'label': _labels[d['label']],
                    }
