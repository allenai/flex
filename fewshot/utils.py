from itertools import zip_longest
import hashlib
from dataclasses import dataclass


def get_hash(s: str):
    """Hash function."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


@dataclass(frozen=True, eq=True)
class ExampleId:
    id: int
    unlabeled: bool = False

    def __repr__(self):
        return f"ExampleId(id={self.id}, unlabeled={self.unlabeled})"


def grouper(n, iterable, fillvalue=None):
    """Group iterable into chunks of size n.

    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    from itertools cookbook.

    """
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def flatten(lsts):
    return [x for sublist in lsts for x in sublist]
