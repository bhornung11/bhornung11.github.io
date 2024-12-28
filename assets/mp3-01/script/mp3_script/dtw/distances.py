"""
Distance functions.
"""

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def calc_dist_l2_sq(
        x1:float,
        x2:float,
        y1:float,
        y2:float
    ) -> float:
    """
    2D L2 distance squared.
    """

    d = np.sqrt(
        (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    )

    return d