from typing import Optional
from dataclasses import dataclass, field
from .base import StoreCfg, Schema


@dataclass
class NewsgroupStoreCfg(StoreCfg):
    path: str = 'flex'
    name: str = 'newsgroupbao'
    local: bool = True
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=['text'],
    ))
    formatter: str = 'fewshot.formatters.newsgroupbao'


@dataclass
class ReutersStoreCfg(StoreCfg):
    path: str = 'flex'
    name: str = 'reutersbao'
    local: bool = True
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=['text'],
    ))
    formatter: str = 'fewshot.formatters.reutersbao'


@dataclass
class HuffpostStoreCfg(StoreCfg):
    path: str = 'flex'
    name: str = 'huffpostbao'
    needs_local_datasets_dir: bool = True
    local: bool = True
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=['headline'],
        label='category',
    ))
    formatter: str = 'fewshot.formatters.huffpostbao'


@dataclass
class FewrelStoreCfg(StoreCfg):
    path: str = 'flex'
    name: str = 'fewrelbao'
    local: bool = True
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=[],  # It's complicated
        label='wikidata_property_name',
    ))
    formatter: str = 'fewshot.formatters.fewrelbao'


@dataclass
class AmazonStoreCfg(StoreCfg):
    path: str = 'amazon'
    local: bool = True
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=['text'],
    ))
    formatter: str = 'fewshot.formatters.amazon'
