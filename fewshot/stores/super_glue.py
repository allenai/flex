from dataclasses import dataclass, field
from typing import Optional
from .base import Schema, StoreCfg

_SUPER_GLUE_VERSION = '1.0.2'


@dataclass
class SuperGlueBoolqStoreCfg(StoreCfg):
    path: str = 'super_glue'
    name: str = 'boolq'
    version: str = _SUPER_GLUE_VERSION
    local: bool = False
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id='idx',
        context=['passage'],
        question='question',
        splits_no_labels=['test'],
    ))
    formatter: str = 'fewshot.formatters.boolq'


@dataclass
class SuperGlueCbStoreCfg(StoreCfg):
    path: str = 'super_glue'
    name: str = 'cb'
    version: str = _SUPER_GLUE_VERSION
    local: bool = False
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id='idx',
        splits_no_labels=['test'],
    ))
    formatter: str = 'fewshot.formatters.cb'


@dataclass
class SuperGlueCopaStoreCfg(StoreCfg):
    path: str = 'super_glue'
    name: str = 'copa'
    version: str = _SUPER_GLUE_VERSION
    local: bool = False
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id='idx',
        splits_no_labels=['test'],
    ))
    formatter: str = 'fewshot.formatters.copa'


@dataclass
class SuperGlueMultiRCStoreCfg(StoreCfg):
    path: str = 'super_glue'
    name: str = 'multirc'
    version: str = _SUPER_GLUE_VERSION
    local: bool = False
    schema: Optional[Schema] = field(default_factory=lambda: Schema(
        example_id='idx',
        splits_no_labels=['test'],
    ))
    formatter: str = 'fewshot.formatters.multirc'
