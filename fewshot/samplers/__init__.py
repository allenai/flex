from typing import Optional, Dict, List
from dataclasses import dataclass, field
from .sample import Sampler


@dataclass
class DefaultSamplerCfg:
    _target_: str = 'fewshot.samplers.Sampler'
    min_way: Optional[int] = None
    max_way: Optional[int] = None
    randomize_class_labels: bool = True
    max_num_support_samples: int = 10
    subsample_support_samples: bool = True
    max_num_target_samples: int = 10
    frac_available_for_support: float = 0.5
    pad_majority_class: bool = False
    min_num_support_samples: Optional[int] = 1
    min_num_support_samples_by_class: Optional[Dict[str, int]] = None
    subsample_support_samples_base: Optional[int] = 2


@dataclass
class FixedLabelsCfg(DefaultSamplerCfg):
    randomize_class_labels: bool = False


@dataclass
class SanityEvalCfg(FixedLabelsCfg):
    way: int = 2
    min_num_support_samples_by_class: Optional[Dict[str, int]] = field(
        default_factory=lambda: {'0': 0, '1': 1}
    )


@dataclass
class Sample2WayMax8ShotCfg(DefaultSamplerCfg):
    way: int = 2
    max_num_support_samples: int = 8


@dataclass
class Sample3WayMax8ShotCfg(DefaultSamplerCfg):
    way: int = 3
    max_num_support_samples: int = 8


@dataclass
class Sample5Way1ShotCfg(DefaultSamplerCfg):
    way: int = 5
    min_num_support_samples: int = 1
    max_num_support_samples: int = 1
    subsample_support_samples_base: Optional[int] = None


@dataclass
class Sample5Way5ShotCfg(DefaultSamplerCfg):
    way: int = 5
    min_num_support_samples: int = 5
    max_num_support_samples: int = 5
    subsample_support_samples_base: Optional[int] = None
    frac_available_for_support: float = 0.7


@dataclass
class Sample5Way1Shot1SampleCfg(DefaultSamplerCfg):
    way: int = 5
    min_num_support_samples: int = 1
    max_num_support_samples: int = 1
    subsample_support_samples_base: Optional[int] = None
    max_num_target_samples: int = 1


@dataclass
class Sample5_10Way1_5ShotCfg(DefaultSamplerCfg):
    max_way: Optional[int] = None
    way: Optional[List[int]] = field(
        default_factory=lambda: [5, 10]
    )
    min_num_support_samples: Optional[int] = None
    max_num_support_samples: Optional[int] = None
    num_support_samples: Optional[List[int]] = field(
        default_factory=lambda: [1, 5]
    )
    subsample_support_samples_base: Optional[int] = None
    max_num_target_samples: int = 10
    prob_balanced: float = 0.4
    prob_zero_shot: float = 0.1


@dataclass
class FlexTestCfg(DefaultSamplerCfg):
    max_num_support_samples: int = 5
    subsample_support_samples_base: Optional[int] = None
    max_num_target_samples: int = 110
    # Set high so that zero-shot get done towards beginning,
    # then hit max_num_target_samples_zero_shot
    prob_zero_shot: float = 0.8
    way: Optional[int] = None
    max_num_target_samples_zero_shot: Optional[int] = None
    max_zero_shot_episodes: Optional[int] = None


@dataclass
class GaoTestCfg(DefaultSamplerCfg):
    min_num_support_samples: Optional[int] = None
    max_num_support_samples: Optional[int] = None
    num_support_samples: Optional[List[int]] = None
    max_num_target_samples: int = 40
    way: Optional[int] = None
    prob_balanced: float = 1.0


@dataclass
class BaoTestCfg(DefaultSamplerCfg):
    min_num_support_samples: Optional[int] = None
    max_num_support_samples: Optional[int] = None
    num_support_samples: Optional[List[int]] = None
    max_num_target_samples: int = 40
    way: int = 5
    prob_balanced: float = 1.0
