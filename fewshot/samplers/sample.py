# Sampling utils, as described in meta-dataset
import math
import itertools
from typing import Sequence, List, Tuple, Optional, Mapping, Union
import logging
import numpy as np
import pandas as pd
from ..datasets.store import Store
from ..utils import ExampleId
logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_SKIPPED = 1000


def _pows_in_interval(
    low: int,
    high: int,
    base: Optional[int] = 2,
) -> List[int]:
    """Get all powers of base in [low, high] interval.

    >>> _pows_in_interval(1, 9)
    [1, 2, 4, 8]
    >>> _pows_in_interval(3, 7)
    [4]
    >>> _pows_in_interval(1, 2)
    [1, 2]
    >>> _pows_in_interval(2, 2)
    [2]
    >>> _pows_in_interval(1, 1)
    [1]

    """
    if high < low:
        raise ValueError
    biggest_pow = math.floor(math.log(high, base))
    smallest_pow = math.ceil(math.log(low, base))
    return [base ** p for p in range(smallest_pow, biggest_pow + 1)]


def _get_query_size(
    class_n: Sequence[int],  # Number available in each class
    max_num_target_samples: int,  # Per class
    frac_available_for_support: float,  # Per class
) -> int:
    """Return the per-class query size

    >>> _get_query_size([1, 1], 10, 0.5)
    1
    >>> _get_query_size([0, 1], 10, 0.5)
    1
    >>> _get_query_size([70, 1], 10, 0.5)
    1
    >>> _get_query_size([70, 20], 10, 0.5)
    10
    >>> _get_query_size([70, 19], 10, 0.5)
    9

    """
    return int(max(
        1,
        min(
            max_num_target_samples,
            min(np.floor((1-frac_available_for_support) * n) for n in class_n)
        )
    ))


def _sample(
    rng: np.random.RandomState,
    class_n: Sequence[int],
    max_num_support_samples: int,
    subsample_support_samples: bool,
    max_num_target_samples: int,
    frac_available_for_support: float,
    min_num_support_samples: Sequence[int],
    subsample_support_samples_base: Optional[int] = None,
    num_support_samples: Optional[Union[Sequence[Sequence[int]], Sequence[int]]] = None,
    prob_balanced: Optional[float] = None,
    prob_zero_shot: Optional[float] = None,
    max_num_target_samples_zero_shot: Optional[int] = None,
) -> Tuple[
    List[int],  # Per-class number per support
    List[int],  # Per-class number per target
    List[float],  # Probability of selecting that support combination
]:
    do_zero_shot = prob_zero_shot and rng.random() <= prob_zero_shot
    if do_zero_shot:
        if max_num_target_samples_zero_shot is not None:
            max_num_target_samples = max_num_target_samples_zero_shot
        frac_available_for_support = 0
    # Determine per-class query size
    # Ensure at least frac_available_for_support of samples left for any class
    query_size = _get_query_size(
        class_n,
        max_num_target_samples,
        frac_available_for_support,
    )
    query_sizes = [query_size for _ in class_n]
    if do_zero_shot:
        support_sizes = [0] * len(class_n)
        probs_subsampled = [1] * len(class_n)
        return support_sizes, query_sizes, probs_subsampled

    remaining_for_support = [max(0, n - query_size) for n in class_n]
    support_sizes = []
    probs_subsampled = []
    for i, (n, d, min_support, num_support) in enumerate(zip(
        remaining_for_support,
        itertools.repeat(max_num_support_samples),
        min_num_support_samples,
        num_support_samples or itertools.repeat(None),
    )):
        if n > 0 and num_support_samples:
            if isinstance(num_support, int):
                num_support = [num_support]
            support_sizes.append(rng.choice(num_support))
            probs_subsampled.append(1 / len(num_support))
        elif n > 0 and subsample_support_samples:
            low = min_support
            high = min(d, n)
            if subsample_support_samples_base is not None:
                if low > 0:
                    choices = _pows_in_interval(
                        low,
                        high,
                        subsample_support_samples_base,
                    )
                else:
                    choices = [0] + _pows_in_interval(
                        low + 1,
                        high,
                        subsample_support_samples_base,
                    )
            else:
                choices = list(range(low, high + 1))
            support_sizes.append(rng.choice(choices))
            probs_subsampled.append(1 / len(choices))
        else:
            support_sizes.append(max(0, min(n, d)))
            probs_subsampled.append(1)

        # With P(prob_balanced), set rest of support sizes to match
        if (
            len(support_sizes) == 1
            and prob_balanced
            and rng.random() <= prob_balanced
            and all(n >= support_sizes[0] for n in remaining_for_support)
        ):
            support_sizes = support_sizes * len(class_n)
            probs_subsampled = probs_subsampled * len(class_n)
            break
    return support_sizes, query_sizes, probs_subsampled


class Sampler:
    def __init__(
        self,
        randomize_class_labels: bool,
        max_num_support_samples: int,
        subsample_support_samples: bool,
        max_num_target_samples: int,
        frac_available_for_support: float,
        pad_majority_class: bool,
        min_way: Optional[int] = None,
        max_way: Optional[int] = None,
        min_num_support_samples: Optional[int] = 1,
        min_num_support_samples_by_class: Optional[Mapping[str, int]] = None,
        subsample_support_samples_base: Optional[int] = None,
        num_support_samples: Optional[Union[int, List[int]]] = None,
        way: Optional[Union[int, List[int]]] = None,
        prob_balanced: Optional[float] = None,
        prob_zero_shot: Optional[float] = None,
        max_num_target_samples_zero_shot: Optional[int] = None,
        max_zero_shot_episodes: Optional[int] = None,
    ) -> None:
        """Init.

        Classes returned always have labels [0, n-1].

        Args:
            min_way: Minimum number of classes to sample. The "way" in n-way k-shot.
            randomize_class_labels: Shuffle class labels to avoid memorization.
            max_num_support_samples: Maximum number of samples per class to
                include in support.
            subsample_support_samples: For each class, uniformly sample
                a support set of size between 1 and the original support size.
            max_num_target_samples: Maximum number of samples per class to
                include in target.
            frac_available_for_support: Fraction of class examples to reserve
                for support.
            pad_majority_class: If node does not have n classes, add 1 example
                from majority class to test.
            max_way: Maximum number of classes to sample. Defaults to all classes.
            min_num_support_samples: Minimum number of samples to
                include in target.
            min_num_support_samples_by_class: Minimum number of samples *per class* to
                include in target.

        """
        self.min_way = min_way
        self.max_way = max_way
        self.way = way
        if way and self.min_way is None:
            self.min_way = way if isinstance(way, int) else min(way)
        if way and self.max_way is None:
            self.max_way = way if isinstance(way, int) else max(way)
        self.randomize_class_labels = randomize_class_labels
        self.max_num_support_samples = max_num_support_samples
        self.subsample_support_samples = subsample_support_samples
        self.max_num_target_samples = max_num_target_samples
        self.frac_available_for_support = frac_available_for_support
        self.pad_majority_class = pad_majority_class
        self.min_num_support_samples = min_num_support_samples
        self.min_num_support_samples_by_class = min_num_support_samples_by_class
        self.subsample_support_samples_base = subsample_support_samples_base
        self.num_support_samples = num_support_samples
        if num_support_samples:
            if isinstance(num_support_samples, int):
                self.min_num_support_samples = num_support_samples
                self.max_num_support_samples = num_support_samples
            else:
                self.min_num_support_samples = min(num_support_samples)
                self.max_num_support_samples = max(num_support_samples)
        self.prob_balanced = prob_balanced
        self.prob_zero_shot = prob_zero_shot
        self.max_num_target_samples_zero_shot = max_num_target_samples_zero_shot
        self.max_zero_shot_episodes = max_zero_shot_episodes
        self.zero_shot_episode_count = 0

    def _get_min_num_support_samples(self, cls: str, default=1) -> int:
        if self.min_num_support_samples_by_class:
            return self.min_num_support_samples_by_class.get(cls, default)
        else:
            return self.min_num_support_samples

    def get_set(
        self,
        rng: np.random.RandomState,
        examples_by_node_class: pd.DataFrame,
        labeled_store: Store,
        unlabeled_store: Optional[Store] = None,
        majority_class: Optional[str] = None,
    ) -> Tuple[
        List[ExampleId],
        List[ExampleId],
        List[str],
        List[str],
        List[str],
        List[float],
        str,
    ]:
        """Sample a set.

        Args:
            rng: Random number generator.
            examples_by_node_class: Examples indexed by (node, class).
            labeled_store: Labeled examples.
            unlabeled_store: Unlabeled examples that can be added to majority
                class.
            majority_class: Majority class, eg for adding unlabeled examples.

        Returns:
            support_ids: Ids for support set.
            query_ids: Ids for query set.
            support_labels: Labels for support set.
            query_labels: Labels for query set.
            label_set: All labels.
            probs_subsampled: Probability that the support set - query set
                size combination was chosen.
            node: Node sampled from.

        """
        nodes = examples_by_node_class.index.unique(level=0)
        classes = None
        # Get a node with at least self.min_way classes
        n_consecutive_skipped = 0
        while classes is None:
            node = rng.choice(nodes)
            class_candidates = examples_by_node_class.loc[node].index.unique(level=0).tolist()

            add_majority = (
                self.pad_majority_class
                and unlabeled_store is not None
                and majority_class is not None
                and majority_class not in class_candidates
                and len(class_candidates) == self.min_way - 1
            )
            if add_majority:
                # We will add majority class with random_instances
                class_candidates.append(majority_class)

            if len(class_candidates) < self.min_way:
                logger.debug(
                    f'Skipping node {node}.'
                    f' Not enough classes: {class_candidates}'
                )
                n_consecutive_skipped += 1
                if len(nodes) == 1:
                    raise Exception(
                        f'Requested at least {self.min_way} classes but only {len(class_candidates)} available')
                elif n_consecutive_skipped > MAX_CONSECUTIVE_SKIPPED:
                    raise Exception(f'Unable to find sample after {n_consecutive_skipped} samples')
                continue
            n_consecutive_skipped = 0

            if self.way is None:
                way = rng.randint(
                    self.min_way,
                    min(self.max_way or float('inf'), len(class_candidates)) + 1
                )
            elif isinstance(self.way, int):
                way = min(self.way, len(class_candidates))
                if way < self.way:
                    logger.warning(
                        f'Using way={way} less than specified way={self.way}'
                        f' due to {len(class_candidates)} available classes'
                    )
            else:
                way = rng.choice([
                    min(way, len(class_candidates)) for way in self.way
                ])
                if way not in self.way:
                    logger.warning(
                        f'Using way={way} not in specified way={self.way}'
                        f' due to taking the minimum of some specified way'
                        f' and {len(class_candidates)} available classes'
                    )

            classes = rng.choice(
                class_candidates,
                size=way,
                replace=False,
            )
            for c in classes:
                if not (add_majority and c == majority_class):
                    # Need at least support number and 1 in test
                    min_num = self._get_min_num_support_samples(c) + 1
                    if len(examples_by_node_class.loc[node, c]) < min_num:
                        logger.debug(
                            f'Skipping node {node}'
                            f' (class {c} with < {min_num}'
                            ' examples)'
                        )
                        classes = None
                        break

        if not self.randomize_class_labels:
            try:
                classes = np.sort(classes.astype(int)).astype(str)
            except ValueError:
                classes = np.sort(classes)
        logger.debug(f'Sampled node {node}, classes {classes}')

        ids_by_class = []
        for c in classes:
            if add_majority and c == majority_class:
                # TODO: Should this sample the same size as the class with
                #   the fewest examples?
                ids = rng.choice(
                    [ExampleId(id=i, unlabeled=True) for i in range(len(unlabeled_store))],
                    # 1 extra for test set
                    size=max(1, self._get_min_num_support_samples(c) + 1),
                    replace=False,
                )
                ids_by_class.append(pd.Series(ids))
            else:
                ids_by_class.append(examples_by_node_class.loc[node, c]['i'])

        max_zero_shot_episodes_reached = (
            self.max_zero_shot_episodes is not None
            and self.zero_shot_episode_count >= self.max_zero_shot_episodes
        )
        support_n, query_n, probs_subsampled = _sample(
            rng=rng,
            class_n=[len(ids) for ids in ids_by_class],
            max_num_support_samples=self.max_num_support_samples,
            subsample_support_samples=self.subsample_support_samples,
            max_num_target_samples=self.max_num_target_samples,
            frac_available_for_support=self.frac_available_for_support,
            min_num_support_samples=[
                self._get_min_num_support_samples(c) for c in classes
            ],
            subsample_support_samples_base=self.subsample_support_samples_base,
            num_support_samples=[self.num_support_samples for _ in classes] if self.num_support_samples else None,
            prob_balanced=self.prob_balanced,
            prob_zero_shot=self.prob_zero_shot if not max_zero_shot_episodes_reached else 0,
            max_num_target_samples_zero_shot=self.max_num_target_samples_zero_shot,
        )
        if all(x == 0 for x in support_n):
            self.zero_shot_episode_count += 1

        support_y = []
        query_y = []
        support_ids = []
        query_ids = []

        # Shuffle within a class
        perms = [rng.permutation(len(ids)) for ids in ids_by_class]

        for class_i, (c, n1, n2, ids, perm) in enumerate(zip(
            classes,
            support_n,
            query_n,
            ids_by_class,
            perms,
        )):
            # support
            s_ids = ids.iloc[perm[:n1]]
            support_y += [str(class_i) for _ in s_ids]
            support_ids += s_ids.tolist()

            # query
            q_ids = ids.iloc[perm[n1:n1+n2]]
            query_y += [str(class_i) for _ in q_ids]
            query_ids += q_ids.tolist()

        # Shuffle support and query sets
        support_perm = rng.permutation(len(support_y))
        query_perm = rng.permutation(len(query_y))
        support_ids = [support_ids[i] for i in support_perm]
        support_y = [support_y[i] for i in support_perm]
        query_ids = [query_ids[i] for i in query_perm]
        query_y = [query_y[i] for i in query_perm]

        logger.debug(f'Train labels: {support_y}')
        logger.debug(f'Test labels: {query_y}')

        text_labels = {label: i for i, label in enumerate(classes)}
        labels = list(str(c) for c in range(len(classes)))
        return (
            support_ids,
            query_ids,
            support_y,
            query_y,
            labels,
            probs_subsampled,
            node,
            text_labels
        )


if __name__ == '__main__':
    import doctest
    doctest.testmod()
