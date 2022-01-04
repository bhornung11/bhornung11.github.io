"""

"""


from typing import (
    List,
    Callable
)

import numpy as np


def calc_rmsd(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculates the root mean squared deviation between
    two sequences of points.

    Parameters:
        x1: np.ndarray : first set of points
        x2: np.ndarray : second set of points

    Returns:
        rmsd: float : root mean squared deviation
    """

    diff = x1 - x2
    rmsd = np.sqrt(np.mean(diff * diff))

    return rmsd


def screen_distances(
        xs: List[np.ndarray],
        fun_dist: Callable,
) -> np.ndarray:
    """
    Calculates the distance matrix of a sequence of vector sequences.

    Parameters:
        xs: List[np.ndarray] : vector sequences
        fun_dist: distance function

    Returns:
        dmat: np.ndarray: distance matrix.
    """

    n = len(xs)
    dmat = np.zeros((n, n))

    for i in range(n):
        xi = xs[i]
        for j in range(i + 1, n):
            dist = fun_dist(xi, xs[j])
            dmat[i, j] = dmat[j, i] = dist

    return dmat
