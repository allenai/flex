"""Hu, M., & Liu, B. (2004). Mining and summarizing customer reviews. KDD '04."""
from typing import List
from pathlib import Path
import datasets

_URL = 'http://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip'

_products = [
    'Apex AD2600 Progressive-scan DVD player',
    'Creative Labs Nomad Jukebox Zen Xtra 40GB',
    'Nokia 6610',
    'Canon G3',
    'Nikon coolpix 4300',
]


def parse_sentiment(s: str):
    if s == '+':
        return 99
    else:
        return int(s)


class CR(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description='TODO',
            features=datasets.Features({
                'product': datasets.Value('string'),
                'text': datasets.Value('string'),
                'title': datasets.Value('string'),
                'features_sentiment': datasets.Sequence(datasets.Value('int8')),
                'features_text': datasets.Sequence(datasets.Value('string')),
                'features_appeared_pronoun': datasets.Sequence(datasets.Value('string')),
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloaded_dir = dl_manager.extract(_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                'data_dir': Path(downloaded_dir) / 'customer review data',
            })
        ]

    def _generate_examples(self, data_dir: str):
        for product in _products:
            with open(Path(data_dir) / f'{product}.txt', 'r', encoding='utf-8') as f:
                title = None
                for i, line in enumerate(f):
                    if line.startswith('[t]'):
                        title = line[3:-1]
                    elif line.startswith('***[t]'):
                        title = line[6:-1]
                    else:
                        if line.startswith('run[+3], dvd media[+2]#apex'):
                            splits = line.split('#')
                        else:
                            splits = line.split('##')
                        if len(splits) == 1 or not splits[1].strip():
                            # Not a dataset sentence
                            continue
                        features = splits[0].split(',')
                        features_text = []
                        features_sentiment = []
                        features_appeared_pronoun = []
                        for f in features:
                            text, *sentiment_and_extra = f.split('[')
                            if not text.strip(' '):
                                continue
                            features_text.append(text.strip(' '))
                            if not sentiment_and_extra:
                                features_sentiment.append(0)
                                features_appeared_pronoun.append('')
                            else:
                                features_sentiment.append(parse_sentiment(sentiment_and_extra[0].rstrip(']}')))
                                if len(sentiment_and_extra) > 1:
                                    features_appeared_pronoun.append(''.join(
                                        s.rstrip(']}') for s in sentiment_and_extra[1:]
                                    ))
                                else:
                                    features_appeared_pronoun.append('')
                        yield f'{product}{i}', {
                            'product': product,
                            'text': splits[1][:-1],
                            'title': title,
                            'features_sentiment': features_sentiment,
                            'features_text': features_text,
                            'features_appeared_pronoun': features_appeared_pronoun,
                        }
