"""
Percentile bootstrap of a distance matrix.
"""

from typing import (
    Callable
)

import numpy as np

def bootstrap_dmat(
        dmat: np.ndarray,
        size_sample: int,
        n_repeat: int,
        func: Callable
    ) -> np.ndarray:
    """
    Function to bootstrap a distance matrix.

    Parameters:
        dmat: np.ndarray : distance matrix
        size_sample: int : number of individuals in the sample
        n_repeat: int : number of bootstrap iterations
        func: Callable : statistics function

    Returns:
        medians: np.ndarray
    """

    n_sample = len(dmat)

    stats = np.full(n_repeat, np.nan)

    for i in range(n_repeat):
        idcs = np.random.choice(n_sample, size_sample, replace=True)
        stat = func(dmat[idcs][:, idcs])

        stats[i] = stat

    return stats
