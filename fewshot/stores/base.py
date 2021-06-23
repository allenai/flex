from typing import Optional, List
from dataclasses import dataclass, field

HF_SCRIPTS_VERSION = '1.5.0'


@dataclass
class Schema:
    document: List[str] = field(default_factory=list)
    sparse: str = 'tfidf_v0.0.1'
    dense: str = 'recommender@v0.1.1'
    example_id: Optional[str] = None
    node: Optional[str] = None
    label: Optional[str] = 'label'
    splits_no_labels: List[str] = field(default_factory=list)


@dataclass
class StoreCfg:
    formatter: str
    path: str
    local: bool
    script_version: str = HF_SCRIPTS_VERSION
    version: Optional[str] = None
    split: Optional[str] = None  # Needs to be filled out though...
    name: Optional[str] = None
    majority_class: Optional[str] = None
    schema: Optional[Schema] = None
    needs_local_datasets_dir: bool = False
    _target_: str = 'fewshot.datasets.store.Store'
