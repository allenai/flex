from typing import Optional, List
from dataclasses import dataclass
from fewshot.stores.base import StoreCfg
from ..samplers import DefaultSamplerCfg


@dataclass
class DatasetCfg:
    labeled_store: StoreCfg
    sampler: DefaultSamplerCfg
    total_samples: int = 999999999999
    seed: int = 0
    unlabeled_store: Optional[StoreCfg] = None
    name: Optional[str] = None
    _target_: str = 'fewshot.datasets.data.Dataset'


@dataclass
class MetadataSamplerCfg:
    datasets: List[DatasetCfg]
    seed: int = 0
    _target_: str = 'fewshot.datasets.data.MetaDatasetSampler'
