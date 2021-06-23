import os
import collections
from typing import List, Iterable, Optional
import datasets
from .flex_utils import (
    _get_reuters_classes, _get_20newsgroup_classes, _get_huffpost_classes,
    _get_fewrel_classes
)
from fewshot.stores.base import HF_SCRIPTS_VERSION

_VERSION = '0.0.1'


class FlexConfig(datasets.BuilderConfig):
    def __init__(
        self,
        # label_column,
        # description,
        # citation,
        # text_features,
        dataset_splits: Iterable[str],
        label_field: str = 'label',
        label_splits: Optional[Iterable[Iterable[str]]] = None,
        load_dataset_path: Optional[str] = None,
        load_dataset_name: Optional[str] = None,
        load_dataset_script_version: Optional[str] = HF_SCRIPTS_VERSION,
        load_dataset_version: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(version=datasets.Version(_VERSION), **kwargs)
        # self.label_column = label_column
        # self.description = description
        # self.citation = citation
        # self.text_features = text_features
        self.dataset_splits = dataset_splits
        self.label_field = label_field
        self.label_splits = label_splits
        self.load_dataset_path = load_dataset_path
        self.load_dataset_name = load_dataset_name
        self.load_dataset_version = load_dataset_version
        self.load_dataset_script_version = load_dataset_script_version


class Flex(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        FlexConfig(
            name='newsgroupbao',
            dataset_splits=['train'],
            label_splits=_get_20newsgroup_classes(),
            load_dataset_path='newsgroup',
            load_dataset_version='3.0.0',
        ),
        FlexConfig(
            name='reutersbao',
            dataset_splits=['train', 'test'],
            label_splits=_get_reuters_classes(),
            load_dataset_path='reuters21578',
            load_dataset_name='ModApte',
        ),
        FlexConfig(
            name='huffpostbao',
            dataset_splits=['train'],
            label_splits=_get_huffpost_classes(),
            load_dataset_path='huffpost',
            label_field='category',
        ),
        FlexConfig(
            name='fewrelbao',
            dataset_splits=['train_wiki', 'val_wiki'],
            label_splits=_get_fewrel_classes(),
            load_dataset_path='few_rel',
            label_field='relation',
        )
    ]

    def _info(self):
        if 'reuters' in self.config.name:
            features = {
                'text': datasets.Value('string'),
                'title': datasets.Value('string'),
                'label': datasets.Value('string'),
            }
        elif 'huffpost' in self.config.name:
            features = {
                'category': datasets.Value('string'),
                'headline': datasets.Value('string'),
                'short_description': datasets.Value('string'),
                'link': datasets.Value('string'),
                'authors': datasets.Value('string'),
                'date': datasets.Value('string'),
            }
        elif 'fewrel' in self.config.name:
            features = {
                "relation": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "head": {
                    "text": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "indices": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                },
                "tail": {
                    "text": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "indices": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                },
                "names": datasets.Sequence(datasets.Value("string")),
                # Features match HF few_rel except this one, to make automated label selection easier:
                "wikidata_property_name": datasets.Value("string"),
            }
        else:
            features = {
                'text': datasets.Value('string'),
                'label': datasets.Value('string'),
            }
        return datasets.DatasetInfo(
            features=datasets.Features(features),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        if dl_manager.manual_dir:
            dataset_path = os.path.join(dl_manager.manual_dir, self.config.load_dataset_path)
        else:
            dataset_path = self.config.load_dataset_path
        return [
            datasets.SplitGenerator(
                split,
                gen_kwargs={
                    'dataset_path': dataset_path,
                    'labels': labels,
                },
            )
            for split, labels in zip(
                [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST],
                self.config.label_splits,
            )
        ]

    def _generate_examples(
        self,
        dataset_path: str,
        labels: Iterable[str],
    ):
        if 'newsgroup' in self.config.name:
            for label in labels:
                for split in self.config.dataset_splits:
                    load_dataset_kwargs = dict(
                        path=dataset_path,
                        name=f'18828_{label}',
                        split=split,
                        script_version=self.config.load_dataset_script_version,
                    )
                    if self.config.load_dataset_version:
                        load_dataset_kwargs['version'] = self.config.load_dataset_version
                    dataset = datasets.load_dataset(**load_dataset_kwargs)
                    for i, e in enumerate(dataset):
                        yield f'{label}_{split}_{i}', {
                            **e,
                            'label': label,
                        }
            return
        n_by_label = collections.defaultdict(int)
        for split in self.config.dataset_splits:
            load_dataset_kwargs = dict(
                path=dataset_path,
                name=self.config.load_dataset_name,
                split=split,
                script_version=self.config.load_dataset_script_version,
            )
            if self.config.load_dataset_version:
                load_dataset_kwargs['version'] = self.config.load_dataset_version
            dataset = datasets.load_dataset(**load_dataset_kwargs)
            for i, e in enumerate(dataset):
                if 'reuters' in self.config.name:
                    if len(e['topics']) == 1:
                        label = e['topics'][0]
                        if (
                            e['text_type'] not in ['"BRIEF"', '"NORM"']
                            or label not in labels
                        ):
                            continue
                        n_by_label[label] += 1
                        yield f'{split}_{i}', {
                            **{k: e[k] for k in e if k in self.info.features},
                            'label': label,
                        }
                else:
                    if e[self.config.label_field] in labels:
                        if 'fewrel' in self.config.name:
                            yield f'{split}_{i}', {**e, 'wikidata_property_name': e['names'][0]}
                        else:
                            yield f'{split}_{i}', e
        # if 'reuters' in self.config.name:
        #     for label in labels:
        #         print(f'{label}: {n_by_label[label]}')
