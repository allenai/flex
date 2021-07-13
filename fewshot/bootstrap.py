from typing import Collection, Callable, Optional
import numpy as np


def bootstrap(
    data: Collection[float],
    func: Callable = np.mean,
    n_boot: int = 10000,
    seed: Optional[int] = None,
):
    rng = np.random.default_rng(seed)

    samples = [
        func(rng.choice(data, size=len(data)))
        for _ in range(n_boot)
    ]
    return samples


def ci(a, percentile=95):
    p = 50 - percentile / 2, 50 + percentile / 2
    return np.nanpercentile(a, p)
