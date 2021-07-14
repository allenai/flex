from typing import Optional, Mapping, Union, Dict, Hashable
from dataclasses import dataclass
from hydra.utils import instantiate
from fewshot.datasets import MetadataSamplerCfg, StoreCfg
from fewshot.datasets.data import MetaDatasetSampler, Dataset
from fewshot.datasets.store import Store
import fewshot.samplers
from fewshot.samplers.sample import Sampler
from fewshot.challenges.eval import Evaluator


def make(id: str, **evaluator_kwargs):
    return registry.make(id, **evaluator_kwargs)


@dataclass
class ChallengeSpec:
    id: str
    num_tasks: int
    metadatasampler: MetadataSamplerCfg
    hash: Optional[str] = None
    num_distractors: int = 0
    include_unlabeled: bool = False
    show_majority_class: bool = False
    train_stores: Optional[Mapping[str, StoreCfg]] = None
    val_stores: Optional[Mapping[str, StoreCfg]] = None

    def make(self, **evaluator_kwargs) -> Evaluator:
        return Evaluator(config_name=self.id, hash=self.hash, **evaluator_kwargs)

    def get_sampler(
        self,
        split: str,
        dataset_sampler: Optional[Union[
            Sampler,
            Dict[str, Sampler]
        ]] = None,
        seed: Optional[Hashable] = 0,
    ) -> MetaDatasetSampler:
        """Get a episode sampler for the training or validation datasets.

        The episode sampler is an iterable over episodes. First, it samples a dataset,
        then episode details (e.g., training and test sets) using a dataset sampler.

        Args:
            split: 'train' or 'val' for training or validation datasets.
            dataset_sampler: A sampler or a mapping of samplers for each dataset.
                Defaults to the fewshot.samplers.UnifewTrainCfg sampler config.
            seed: Random seed for sampling datasets and for sampling within datasets.

        """
        if split == 'train':
            store_cfgs = self.train_stores
        elif split == 'val':
            store_cfgs = self.val_stores
        else:
            raise ValueError('Available splits are "train" or "val"')
        if dataset_sampler is None:
            dataset_sampler = instantiate(fewshot.samplers.UnifewTrainCfg)
        if isinstance(dataset_sampler, Sampler):
            samplers = {dataset_name: dataset_sampler for dataset_name in store_cfgs}
        datasets = [
            Dataset(
                labeled_store=instantiate(store_cfgs[dataset_name]),
                sampler=samplers[dataset_name],
                seed=seed,
                name=dataset_name,
            )
            for dataset_name in store_cfgs
        ]
        return MetaDatasetSampler(
            seed=seed,
            datasets=datasets,
        )

    def get_stores(self, split: str) -> Dict[str, Store]:
        """Get the low-level stores for a split.

        The raw HuggingFace dataset is available at Store.store.
        The target label for the task is Store.label.

        Args:
            split: 'train' or 'val' for training or validation datasets.

        """
        if split == 'train':
            store_cfgs = self.train_stores
        elif split == 'val':
            store_cfgs = self.val_stores
        else:
            raise ValueError('Available splits are "train" or "val"')
        return {k: instantiate(store_cfgs[k]) for k in store_cfgs}


class ChallengeRegistry:
    def __init__(self) -> None:
        self.specs = dict()

    def register(self, spec: ChallengeSpec) -> None:
        self.specs[spec.id] = spec

    def get_spec(self, id: str) -> ChallengeSpec:
        if id not in self.specs:
            raise ValueError(f'{id} not in available challenges: {list(self.specs)}')
        return self.specs[id]

    def make(self, id: str, **evaluator_kwargs) -> Evaluator:
        return self.get_spec(id).make(**evaluator_kwargs)


registry = ChallengeRegistry()
