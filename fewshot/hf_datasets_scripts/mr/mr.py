"""Bo Pang and Lillian Lee, Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales, Proceedings of ACL 2005."""
from typing import List, Dict
from pathlib import Path
import datasets

_URL = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'


class MR(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description='TODO',
            features=datasets.Features({
                'text': datasets.Value('string'),
                'label': datasets.features.ClassLabel(names=['negative', 'positive']),
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloaded_dir = dl_manager.extract(_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                'label_filepaths': {
                    'positive': Path(downloaded_dir) / 'rt-polaritydata' / 'rt-polarity.pos',
                    'negative': Path(downloaded_dir) / 'rt-polaritydata' / 'rt-polarity.neg',
                },
            })
        ]

    def _generate_examples(self, label_filepaths: Dict[str, str]):
        for label in label_filepaths:
            with open(label_filepaths[label], 'r', encoding='latin-1') as f:
                for i, line in enumerate(f):
                    assert line.endswith('\n')
                    yield f'{label}-{i}', {'text': line[:-1], 'label': label}
