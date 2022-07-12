from typing import List
import json
import datasets


_URL = 'https://huggingface.co/datasets/Fraser/news-category-dataset/resolve/78519a9e29fbcd20102a23c95e6d92e51b0c24ba/News_Category_Dataset_v2.json'


class HuffPost(datasets.GeneratorBasedBuilder):
    VERSION = datasets.utils.Version('0.0.2')

    def _info(self):
        return datasets.DatasetInfo(
            description='TODO',
            features=datasets.Features({
                'category': datasets.Value('string'),
                'headline': datasets.Value('string'),
                'short_description': datasets.Value('string'),
                'link': datasets.Value('string'),
                'authors': datasets.Value('string'),
                'date': datasets.Value('string'),
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        path = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                'filepath': path,
            })
        ]

    def _generate_examples(self, filepath: str):
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                yield i, json.loads(line)
