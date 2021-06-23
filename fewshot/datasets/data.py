from typing import Optional, Sequence, Tuple, Hashable, Iterable, List
import numpy as np
import pandas as pd
import logging
from ..utils import ExampleId
from .store import Store
from ..samplers import Sampler
logger = logging.getLogger(__name__)


TOKENIZE_FIELD = 'flex.txt'


class Dataset:
    LOGGING_FREQ = 1000

    def __init__(
        self,
        labeled_store: Store,
        sampler: Sampler,
        total_samples: int = 999999999999,
        seed: int = 0,
        unlabeled_store: Optional[Store] = None,
        name: Optional[str] = None,
    ) -> None:
        """Init.

        Args:
            labeled_store: Store with labeled examples.
            total_samples: Total dataset size.
            seed: Seed.
            sampler_cfg: Config for sampler (see sample.py).
            unlabeled_store: Store with unlabeled examples, e.g.,
                to add to majority class.
            majority_class: The majority class, for adding random instances.

        """
        self.total_samples = total_samples
        self.seed = seed
        self.name = name or labeled_store.name
        self.majority_class = labeled_store.majority_class

        logger.info(f'Init {self.name} dataset')
        examples = pd.DataFrame.from_dict({
            'i': [ExampleId(i) for i in range(len(labeled_store.store))],
            'cls': [str(x) for x in labeled_store.store[labeled_store.label]],
            'node': labeled_store.store['node'] if 'node' in labeled_store.store.column_names else None,
        })
        if examples['node'].isnull().all():
            examples['node'] = 'default'

        self.examples_by_node_class = examples.set_index(['node', 'cls']).sort_index()

        self.labeled_store = labeled_store
        self.unlabeled_store = unlabeled_store
        self.sampler = sampler

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, i) -> Tuple:
        if i % self.LOGGING_FREQ == 0:
            logger.debug(
                f'Getting sample {i}/{len(self)} from dataset {self.name}'
            )
        return self.get_set(seed=self.seed + i)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.total_samples:
            raise StopIteration
        res = self[self.i]
        self.i += 1
        return res

    def get_example_info(self, id: str, unlabeled: bool) -> dict:
        """Try to get example info from multiple stores."""
        if unlabeled:
            return self.unlabeled_store[int(id)]
        else:
            return self.labeled_store[int(id)]

    def get_examples(self, ids: Iterable[ExampleId]) -> List[str]:
        """Return just text portion."""
        examples = []
        for id in ids:
            x = self.get_example_info(id=id.id, unlabeled=id.unlabeled)[TOKENIZE_FIELD]
            examples.append(x)

        return examples

    def get_set(self, seed: Hashable) -> Tuple:
        """Return a sample"""
        rng = np.random.RandomState(seed=seed)
        (
            support_ids,
            query_ids,
            support_y,
            query_y,
            labels,
            probs_subsampled,
            node,
            text_labels
        ) = self.sampler.get_set(
            rng=rng,
            examples_by_node_class=self.examples_by_node_class,
            labeled_store=self.labeled_store,
            unlabeled_store=self.unlabeled_store,
            majority_class=self.majority_class,
        )

        metadata = {
            'dataset': self,
            'support_ids': support_ids,
            'query_ids': query_ids,
            'labels': labels,
            'probs_subsampled': probs_subsampled,
            'node': node,
            'text_labels': text_labels  # set of possible labels in the dataset
        }
        support_x = self.get_examples(support_ids)
        query_x = self.get_examples(query_ids)
        support_y = np.array(support_y, dtype=np.float32)
        query_y = np.array(query_y, dtype=np.float32)
        return support_x, query_x, support_y, query_y, metadata


class MetaDatasetSampler:
    def __init__(
        self,
        datasets: Sequence[Dataset],
        seed: Hashable = 0,
        max_rejection_samples: int = 999999,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.datasets = datasets
        self.max_rejection_samples = max_rejection_samples

    def sample_dataset(self) -> int:
        return self.rng.choice(len(self.datasets))

    def __iter__(self):
        self.loaders = [iter(dataset) for dataset in self.datasets]
        return self

    def __len__(self) -> int:
        return 99999999999  # PTL needs a __len__ function in the dataiterator for val_percent_check to work

    def __next__(self) -> Tuple:
        # Rejection sampling until dataset with remaining samples is found
        for i in range(self.max_rejection_samples):
            dataset_i = self.sample_dataset()
            try:
                next_set = next(self.loaders[dataset_i])
                break
            except StopIteration:
                continue
        else:
            raise StopIteration
        return next_set
