from .challenges.registration import (  # noqa: F401
    make as make_challenge,
    registry as _registry,
)
from .challenges.eval import Model  # noqa: F401

get_challenge_spec = _registry.get_spec

__version__ = '0.1.0'
