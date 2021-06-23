"""Bo Pang and Lillian Lee, A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts, Proceedings of ACL 2004."""
from typing import List, Dict
from pathlib import Path
import datasets

_URL = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz'


class Subj(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description='TODO',
            features=datasets.Features({
                'text': datasets.Value('string'),
                'label': datasets.features.ClassLabel(names=['subjective', 'objective']),
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloaded_dir = dl_manager.extract(_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                'label_filepaths': {
                    'subjective': Path(downloaded_dir) / 'quote.tok.gt9.5000',
                    'objective': Path(downloaded_dir) / 'plot.tok.gt9.5000',
                },
            })
        ]

    def _generate_examples(self, label_filepaths: Dict[str, str]):
        for label in label_filepaths:
            with open(label_filepaths[label], 'r', encoding='latin-1') as f:
                for i, line in enumerate(f):
                    assert line.endswith('\n')
                    yield f'{label}-{i}', {'text': line[:-1], 'label': label}
