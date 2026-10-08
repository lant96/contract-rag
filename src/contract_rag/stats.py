"""Small statistics helpers for comparing two evaluation runs."""

import random

N_RESAMPLES = 2000


def bootstrap_difference(groups: list[list[float]], seed: int = 0) -> tuple[float, float, float]:
    """Average of the numbers in `groups`, with a 95% interval.

    `groups` holds one list per contract: the differences (run B minus run A) of all the
    clause pairs of that contract. Resampling whole contracts, because the clauses of one
    contract are not independent of each other (a messy contract is hard for all of them).

    Returns (mean, low, high). If the interval contains 0, the difference could be noise.
    """
    all_values = [value for group in groups for value in group]
    mean = sum(all_values) / len(all_values)

    rng = random.Random(seed)
    means = []
    for _ in range(N_RESAMPLES):
        resampled = []
        for _ in range(len(groups)):
            resampled.extend(groups[rng.randrange(len(groups))])
        means.append(sum(resampled) / len(resampled))

    means.sort()
    low = means[int(0.025 * N_RESAMPLES)]
    high = means[int(0.975 * N_RESAMPLES) - 1]
    return mean, low, high


def bootstrap_mean(groups: list[list[float]], seed: int = 0) -> tuple[float, float, float]:
    """Average of the numbers in `groups`, with a 95% interval, resampling whole groups.

    The same method as bootstrap_difference. Use this one for a single metric, with one
    list of values per contract.
    """
    return bootstrap_difference(groups, seed)
