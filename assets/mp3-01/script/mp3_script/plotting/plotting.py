"""
Plotting helper functions.
"""

from typing import (
    Tuple
)

import numpy as np


def make_connecting_lines(
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        idcs: np.ndarray
    ) -> Tuple[np.ndarray]:
    """
    Creates the coordinates of a sequence of lines to be plottend together
    connecting points of two curves.

    Parameters:
        x1: np.ndarray : abscissa of the first curve
        x2: np.ndarray : abscissa of the second curve
        y1: np.ndarray : ordinate of the first curve
        y2: np.ndarray : ordinate of the second curve
        idcs: np.ndarray : indices of points connected of the two curves
    
    Returns:
        x: np.ndarray : abscissa of the connecting lines
        y: np.ndarray : ordinate of the connecting lines
    """

    x = np.zeros(len(idcs) * 3)
    y = np.zeros(len(idcs) * 3)

    k = 0
    for i1, i2 in idcs:
        x[k] = x1[i1]
        y[k] = y1[i1]
        k += 1
        x[k] = x2[i2]
        y[k] = y2[i2]
        k += 1
        x[k] = np.nan
        y[k] = np.nan
        k += 1

    return x, y
