"""
Procustes analysis functions.
"""

import numpy as np

from util.data_structures import RotationResult
from util.geometry import (
    calc_rotation_angle,
    rotate_2D
)

from .generic import calc_rmsd


def calc_procrustes_distance(
        u: np.ndarray,
        v: np.ndarray,
        n_shift: int
) -> RotationResult:
    """
    Calculates the Procrustes distances between two closed curves.

    Parameters:
        u: np.ndarray
        v: np.ndarray
        n_shift: int : circular index shift

    Returns:
        result_best: RotationResult : best match
    """

    # preps
    result_best = RotationResult(None, None, 0, None, np.inf)

    shift = len(u) // n_shift

    # mirror x coordinates
    for i in [1, -1]:

        v_ = np.copy(v)
        v_[:, 0] *= i

        # mirror indices
        for j in [1, -1]:
            v_ = v_[::j]

            # cyclic index shift
            for k in range(n_shift):
                v_sh = np.roll(v_, shift * k, axis=0)

                # rotate
                angle = calc_rotation_angle(u, v_sh)
                v_rot = rotate_2D(v_sh, angle)

                # error
                rmsd = calc_rmsd(u, v_rot)

                if rmsd < result_best.score:
                    result_best = RotationResult(
                        u, k * shift, v_rot, angle, rmsd
                    )

    return result_best
