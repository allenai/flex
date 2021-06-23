from typing import List
import json
import datasets


_URL = 'https://www.researchgate.net/profile/Rishabh-Misra/publication/332141218_News_Category_Dataset/data/5ca2da43a6fdccab2f67c89b/News-Category-Dataset-v2.json'


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
