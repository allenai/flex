from typing import Optional
from dataclasses import dataclass, field
from .base import Schema, StoreCfg


@dataclass
class SNLICfg(StoreCfg):
    path: str = 'snli'
    local: bool = False
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=['premise', 'hypothesis'],
    ))
    formatter: str = 'fewshot.formatters.snli_batched'
    batched_formatter: bool = True


@dataclass
class TrecCfg(StoreCfg):
    path: str = 'trec'
    local: bool = False
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=['text'],
    ))
    formatter: str = 'fewshot.formatters.trec'


@dataclass
class ConllCfg(StoreCfg):
    path: str = 'conll2003'
    local: bool = False
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=['text'],
    ))
    formatter: str = 'fewshot.formatters.conll_batched'
    batched_formatter: bool = True


@dataclass
class SciTailCfg(StoreCfg):
    path: str = 'scitail'
    name: str = 'snli_format'
    local: bool = False
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=['sentence1', 'sentence2'],
    ))
    formatter: str = 'fewshot.formatters.scitail'


@dataclass
class MRCfg(StoreCfg):
    path: str = 'mr'
    local: bool = True
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=['text'],
    ))
    formatter: str = 'fewshot.formatters.mr'


@dataclass
class CRCfg(StoreCfg):
    path: str = 'cr'
    local: bool = True
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=['text'],
    ))
    formatter: str = 'fewshot.formatters.cr_batched'
    batched_formatter: bool = True


@dataclass
class SubjCfg(StoreCfg):
    path: str = 'subj'
    local: bool = True
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id=None,
        document=['text'],
    ))
    formatter: str = 'fewshot.formatters.subj'
