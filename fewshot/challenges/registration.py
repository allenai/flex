from typing import Optional, Mapping
from dataclasses import dataclass
from fewshot.datasets import MetadataSamplerCfg, StoreCfg
from .eval import Evaluator


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

    def make(self, **evaluator_kwargs):
        return Evaluator(config_name=self.id, hash=self.hash, **evaluator_kwargs)


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
