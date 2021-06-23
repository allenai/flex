"""Huggingface datasets FLEET challenge dataset script."""
import logging
import json
import numpy as np
from hydra.utils import instantiate
from fewshot.utils import get_hash
import datasets
from fewshot.challenges import registry
from fewshot.utils import ExampleId
logger = logging.getLogger('datasets.challenge')


_VERSION = '0.0.1'
_SEED = 0
_OOV_DELIMITER = '|'


class FlexChallengeConfig(datasets.BuilderConfig):
    def __init__(
        self,
        config_name: str,
        answer_key: bool = False,
        **kwargs
    ):
        super().__init__(version=datasets.Version(_VERSION), **kwargs)
        self.config_name = config_name
        self.answer_key = answer_key


_builder_configs = []
for name in registry.specs:
    _builder_configs += [
        FlexChallengeConfig(
            name=name,
            config_name=name,
        ),
        FlexChallengeConfig(
            name=f'{name}-answers',
            config_name=name,
            answer_key=True,
        ),
    ]


class FlexChallenge(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIGS = _builder_configs

    def _info(self):
        if self.config.answer_key:
            features = {
                'task_id': datasets.Value('int64'),
                'hashed_id': datasets.Value('string'),
                'question_id': datasets.Value('string'),
                'train_labels': [datasets.Value('string')],
                'labels': [datasets.Value('string')],
                'text_labels': {
                    'keys': [datasets.Value('string')],
                    'values': [datasets.Value('int64')]
                },
                'label': datasets.Value('string'),
                'majority_class': datasets.Value('string'),
                'dataset': datasets.Value('string'),
                'probs_subsampled': [datasets.Value('float32')],
                'node': datasets.Value('string'),
                'is_distractor': datasets.Value('bool'),
            }
        else:
            features = {
                'task_id': datasets.Value('int64'),
                'is_train': datasets.Value('bool'),
                'hashed_id': datasets.Value('string'),
                'example_id': datasets.Value('string'),
                'question_id': datasets.Value('string'),
                'labels': [datasets.Value('string')],
                'text_labels': {
                    'keys': [datasets.Value('string')],
                    'values': [datasets.Value('int64')]
                },
                'majority_class': datasets.Value('string'),
                'dataset': datasets.Value('string'),
                'label': datasets.Value('string'),
                'txt': datasets.Value('string'),
                'dense': [datasets.Value('float32')],
                'sparse': {
                    'i': [datasets.Value('int64')],
                    'd': [datasets.Value('float32')],
                    'dim': datasets.Value('int64'),
                },
                'unlabeled_store_kwargs': datasets.Value('string')
            }
        return datasets.DatasetInfo(features=datasets.Features(features))

    def _split_generators(self, _):
        return [datasets.SplitGenerator(datasets.Split.TEST)]

    def _generate_examples(self):
        """Generate examples.

        Outputs the following mapping from data.Dataset.store examples x:
        {
            'txt': x['flex.txt'],
            'dense': x['flex.dense'],
            'sparse': x['flex.sparse'],
        }

        """
        challenge = registry.get_spec(self.config.config_name)
        sampler = instantiate(challenge.metadatasampler)
        rng = np.random.RandomState(_SEED)
        for task_i, (
            _,
            _,
            support_y,
            query_y,
            metadata,
        ) in zip(
            range(challenge.num_tasks),
            sampler
        ):
            logger.info(f'Task {task_i}/{challenge.num_tasks}')
            common = {
                'task_id': task_i,
                'majority_class': '',
                'labels': sorted([str(i) for i in metadata['labels']]),
                'text_labels': {'keys': list(metadata['text_labels'].keys()),
                                'values': list(metadata['text_labels'].values()),
                                },
                'hashed_id': '',
                'question_id': '',
                'label': '',
                'dataset': metadata['dataset'].name,
            }
            if (
                challenge.show_majority_class
                and metadata['dataset'].majority_class is not None
            ):
                common['majority_class'] = str(
                    metadata['dataset'].majority_class
                )
            if (
                challenge.include_unlabeled
                and metadata['dataset'].unlabeled_store is not None
            ):
                unlabeled_store_kwargs = json.dumps(
                    metadata['dataset'].unlabeled_store.kwargs
                )
            else:
                unlabeled_store_kwargs = ''
            train_inds = rng.permutation(len(metadata['support_ids']))
            test_inds = rng.permutation(len(metadata['query_ids']))

            # Yield training set
            train_labels = []
            train_hashed_ids = []
            for i in train_inds:
                support_exampleid = metadata['support_ids'][i]
                ex = metadata['dataset'].get_example_info(
                    id=support_exampleid.id,
                    unlabeled=support_exampleid.unlabeled
                )
                example_id = ex.get('flex.example_id', repr(support_exampleid))
                label = str(int(support_y[i]))
                hashed_id = get_hash(_OOV_DELIMITER.join([
                    'train',
                    str(task_i),
                    str(i),
                    example_id,
                    label,
                ]))
                train_labels.append(label)
                train_hashed_ids.append(hashed_id)
                if not self.config.answer_key:
                    yield hashed_id, {
                        **common,
                        'unlabeled_store_kwargs': unlabeled_store_kwargs,
                        'is_train': True,
                        'label': label,
                        'example_id': example_id,
                        'txt': ex.get('flex.txt', ''),
                        'dense': ex.get('flex.dense', []),
                        'sparse': ex.get('flex.sparse', {'i': [], 'd': [], 'dim': 0}),
                    }

            # Accumulate test set
            common_answer_key = {
                'train_labels': train_labels,
                'probs_subsampled': metadata['probs_subsampled'],
                'node': metadata['node'],
            }
            test_set = []
            for i in test_inds:
                query_exampleid = metadata['query_ids'][i]
                ex = metadata['dataset'].get_example_info(
                    id=query_exampleid.id,
                    unlabeled=query_exampleid.unlabeled
                )
                example_id = ex.get('flex.example_id', repr(query_exampleid))
                hashed_id = get_hash(_OOV_DELIMITER.join([
                    'test',
                    str(task_i),
                    str(i),
                    example_id,
                ]))
                hashed_id_with_train = get_hash(_OOV_DELIMITER.join([
                    'test',
                    str(task_i),
                    str(i),
                    example_id,
                    *train_hashed_ids,
                ]))
                if not self.config.answer_key:
                    test_set.append((
                        hashed_id,
                        {
                            **common,
                            'unlabeled_store_kwargs': unlabeled_store_kwargs,
                            'is_train': False,
                            'hashed_id': hashed_id_with_train,
                            'question_id': hashed_id,
                            'example_id': example_id,
                            'txt': ex.get('flex.txt', ''),
                            'dense': ex.get('flex.dense', []),
                            'sparse': ex.get('flex.sparse', {'i': [], 'd': [], 'dim': 0}),
                        }
                    ))
                else:
                    test_set.append((
                        hashed_id,
                        {
                            **common,
                            **common_answer_key,
                            'hashed_id': hashed_id_with_train,
                            'question_id': hashed_id,
                            'is_distractor': False,
                            'label': str(int(query_y[i])),
                        }
                    ))

            # Accumulate distractors
            if (
                challenge.num_distractors
                and metadata['dataset'].unlabeled_store is not None
                and metadata['dataset'].majority_class is not None
            ):
                unlabeled_ids_used = set([
                    id.id for id in metadata['support_ids'] + metadata['query_ids']
                    if id.unlabeled
                ])
                candidate_distractors = [
                    i for i in metadata['dataset'].unlabeled_store
                    if i not in unlabeled_ids_used
                ]
                distractor_ids = rng.choice(
                    candidate_distractors,
                    size=min(
                        challenge.num_distractors,
                        len(candidate_distractors),
                    ),
                    replace=False,
                )
                for i, id in enumerate(distractor_ids):
                    distractor_exampleid = ExampleId(id=id, unlabeled=True)
                    ex = metadata['dataset'].get_example_info(
                        id=distractor_exampleid.id,
                        unlabeled=distractor_exampleid.unlabeled
                    )
                    example_id = ex.get('flex.example_id', repr(distractor_exampleid))
                    hashed_id = get_hash(_OOV_DELIMITER.join([
                        'distractor',
                        str(task_i),
                        str(i),
                        example_id,
                    ]))
                    hashed_id_with_train = get_hash(_OOV_DELIMITER.join([
                        'distractor',
                        str(task_i),
                        str(i),
                        example_id,
                        *train_hashed_ids,
                    ]))
                    if not self.config.answer_key:
                        test_set.append((
                            hashed_id,
                            {
                                **common,
                                'unlabeled_store_kwargs': unlabeled_store_kwargs,
                                'is_train': False,
                                'hashed_id': hashed_id_with_train,
                                'question_id': hashed_id,
                                'example_id': ex.get('flex.example_id', ''),
                                'txt': ex.get('flex.txt', ''),
                                'dense': ex.get('flex.dense', []),
                                'sparse': ex.get('flex.sparse', {'i': [], 'd': [], 'dim': 0}),
                            }
                        ))
                    else:
                        test_set.append((
                            hashed_id,
                            {
                                **common,
                                **common_answer_key,
                                'hashed_id': hashed_id_with_train,
                                'question_id': hashed_id,
                                'is_distractor': True,
                                'label': str(metadata['dataset'].majority_class),
                            }
                        ))

            # Yield shuffled test set
            rng.shuffle(test_set)
            for tup in test_set:
                yield tup
