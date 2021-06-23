from dataclasses import dataclass, field
from typing import Optional, List
from .base import Schema, StoreCfg

_GLUE_VERSION = '1.0.0'


@dataclass
class BaseGlueSchema(Schema):
    splits_no_labels: List[str] = field(default_factory=lambda: ['test'])
    example_id: str = 'idx'


@dataclass
class BaseGlueCfg(StoreCfg):
    path: str = 'glue'
    version: str = _GLUE_VERSION
    local: bool = False


@dataclass
class GlueColaCfg(BaseGlueCfg):
    name: str = 'cola'
    schema: Optional[Schema] = field(default_factory=lambda: BaseGlueSchema(
        document=['sentence'],
    ))
    formatter: str = 'fewshot.formatters.glue.sentence_formatter'


@dataclass
class GlueSst2Cfg(BaseGlueCfg):
    name: str = 'sst2'
    schema: Optional[Schema] = field(default_factory=lambda: BaseGlueSchema(
        document=['sentence'],
    ))
    formatter: str = 'fewshot.formatters.glue.sentence_formatter'


@dataclass
class GlueMrpcCfg(BaseGlueCfg):
    name: str = 'mrpc'
    schema: Optional[Schema] = field(default_factory=lambda: BaseGlueSchema(
        document=['sentence1', 'sentence2'],
    ))
    formatter: str = 'fewshot.formatters.glue.mrpc_formatter'


@dataclass
class GlueQqpCfg(BaseGlueCfg):
    name: str = 'qqp'
    schema: Optional[Schema] = field(default_factory=lambda: BaseGlueSchema(
        document=['question1', 'question2'],
    ))
    formatter: str = 'fewshot.formatters.glue.qqp_formatter'


@dataclass
class GlueMnliCfg(BaseGlueCfg):
    name: str = 'mnli'
    schema: Optional[Schema] = field(default_factory=lambda: BaseGlueSchema(
        document=['premise', 'hypothesis'],
    ))
    formatter: str = 'fewshot.formatters.glue.mnli_formatter'


@dataclass
class GlueQnliCfg(BaseGlueCfg):
    name: str = 'qnli'
    schema: Optional[Schema] = field(default_factory=lambda: BaseGlueSchema(
        document=['question', 'sentence'],
    ))
    formatter: str = 'fewshot.formatters.glue.qnli_formatter'


@dataclass
class GlueRteCfg(BaseGlueCfg):
    name: str = 'rte'
    schema: Optional[Schema] = field(default_factory=lambda: BaseGlueSchema(
        document=['sentence1', 'sentence2'],
    ))
    formatter: str = 'fewshot.formatters.glue.rte_formatter'


@dataclass
class GlueWnliCfg(BaseGlueCfg):
    name: str = 'wnli'
    schema: Optional[Schema] = field(default_factory=lambda: BaseGlueSchema(
        document=['sentence1', 'sentence2'],
    ))
    formatter: str = 'fewshot.formatters.glue.wnli_formatter'
