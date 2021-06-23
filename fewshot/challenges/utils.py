from pathlib import Path
from datasets import load_dataset, Dataset
from datasets.config import HF_DATASETS_CACHE
from . import registration
from ..utils import get_hash

_OOV_DELIMITER = '|'


def get_gold_dataset(challenge_name: str, ignore_verification: bool = False):
    d = load_dataset(
        str((Path(__file__).parent.parent / 'hf_datasets_scripts' / 'challenge').resolve()),
        name=f'{challenge_name}-answers',
        split='test',
    )
    challenge_spec = registration.registry.get_spec(challenge_name)
    if not ignore_verification and challenge_spec.hash is not None:
        h = get_challenge_hash(d)
        if challenge_spec.hash != h:
            raise ValueError(wrong_hash_message.format(
                hash=h,
                expected_hash=challenge_spec.hash,
                challenge=challenge_name,
            ))
    return d


def get_challenge_hash(
    dataset: Dataset,
    test_only: bool = True,
    hash_col: str = 'hashed_id',
    delimiter: str = _OOV_DELIMITER,
):
    if test_only and 'is_train' in dataset.features:
        dataset = dataset.filter(lambda x: not x['is_train'])
    return get_hash(delimiter.join(dataset[hash_col]))


wrong_hash_message = (
    'Hash for challenge {challenge} ({hash}) different from expected hash ({expected_hash}).'
    f' Please clear your Huggingface datasets cache ({HF_DATASETS_CACHE}) and try again.'
)
