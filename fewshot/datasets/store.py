import importlib
from dataclasses import is_dataclass
import logging
from typing import Optional, Union
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from datasets import load_dataset, ClassLabel
from fewshot.stores.base import Schema
logger = logging.getLogger(__name__)


class Store:
    """Wrapper around Huggingface datasets."""

    def __init__(
        self,
        path: str,
        split: str,
        formatter: str,  # Path to formatter
        name: Optional[str] = None,
        script_version: Optional[str] = None,
        version: Optional[str] = None,
        local: bool = False,
        clean_cache: bool = False,
        tokenize: bool = False,
        needs_local_datasets_dir: bool = False,
        schema: Optional[Union[Schema, dict, DictConfig]] = None,
        majority_class: Optional[str] = None,
        batched_formatter: bool = False,
    ):
        """Init to look like the txt portion of a Store."""
        import dataclasses
        if is_dataclass(schema):
            schema = dataclasses.asdict(schema)
        elif OmegaConf.is_config(schema):
            schema = OmegaConf.to_container(schema)
        schema = schema or {}
        self.schema = schema
        # Store kwargs to let models reproduce unlabeled store.
        self.kwargs = {
            'path': path,
            'split': split,
            'name': name,
            'script_version': script_version,
            'version': version,
            'local': local,
            'clean_cache': clean_cache,
            'tokenize': tokenize,
            'needs_local_datasets_dir': needs_local_datasets_dir,
            'schema': schema,
            'formatter': formatter,
            'batched_formatter': batched_formatter,
        }
        self.majority_class = majority_class
        self.split = split
        self.name = path
        if name:
            self.name += f'.{name}'
        self.name += f'-{split}'
        here = Path(__file__).absolute().parent
        local_datasets_dir = here.parent / 'hf_datasets_scripts'
        local_dir = local_datasets_dir / path
        logger.info(f'Calling load_dataset with split={split}')
        hf_load_kwargs = dict(
            path=str(local_dir) if local else path,
            name=name,
            script_version=script_version if not local else None,
            split=split,
            data_dir=local_datasets_dir if needs_local_datasets_dir else None,
        )
        if version:
            hf_load_kwargs['version'] = version
        self.store = load_dataset(**hf_load_kwargs)
        if clean_cache:
            self.store.cleanup_cache_files()

        # Call dataset-specific formatters.
        def get_f(s: str):
            m = s.split('.')
            fname = m[-1]
            m = importlib.import_module('.'.join(m[:-1]))
            return getattr(m, fname)
        formatter_f = get_f(formatter)
        # TODO: Something weird here with loading from cache when making challenge
        self.store = self.store.map(formatter_f, batched=batched_formatter)
        # self.store = self.store.map(formatter, load_from_cache_file=False)

        # Convert ClassLabel into its string version for the label field.
        if isinstance(self.store.features[self.label], ClassLabel):
            self.store = self.store.map(lambda d: {
                self.label: self.store.features[self.label].int2str(d[self.label])
            })

    @property
    def label(self):
        return self.schema.get('label', 'label')

    def __getitem__(self, id: int):
        return self.store[id]

    def __iter__(self):
        return iter(range(len(self.store)))

    def __contains__(self, id: int):
        return id >= 0 and id < len(self.store)

    def __len__(self):
        return len(self.store)
