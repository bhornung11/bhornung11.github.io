"""
Statistical descriptor functions.
"""

from typing import (
    Any,
    Dict
)

import numpy as np

def describe_dmat(dmat: np.ndarray) -> Dict[str, Any]:
    """
    Calculate and collate desciptive stats of a distance matrix.
    """

    store = {
        "dmat": dmat,
        "mean": np.nanmean(dmat),
        "median": np.nanmedian(dmat),
        "std": np.nanstd(dmat),
        "p10_p90": np.nanquantile(dmat, q=[0.1, 0.9]),
        "means": np.nanmean(dmat, axis=0),
        "medians": np.nanmedian(dmat, axis=0),
        "stds": np.nanstd(dmat, axis=0),
        "p10_p90s": np.nanquantile(dmat, q=[0.1, 0.9], axis=0),
        "ij_min": np.unravel_index(np.nanargmin(dmat, axis=None), dmat.shape),
        "ij_max": np.unravel_index(np.nanargmax(dmat, axis=None), dmat.shape),
        "i_min": np.argmin(np.nansum(dmat, axis=1)),
        "i_max": np.argmax(np.nansum(dmat, axis=1))
    }

    return store
