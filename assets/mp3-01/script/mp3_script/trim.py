"""
Rudimetary subsampling utilities.
"""

import numba as nb
import numpy as np

from mp3_script.mp3_types import Song


def trim_series_w(
        song: Song,
        left: float,
        right: float
    ) -> Song:
    """
    Wrapped trimming function.

    Parameters:
        song: Song : song!
        left: float : number of seconds to remove from the left
        left: float : number of seconds to remove from the right

    Returns:
        trimmed: Song : trimmed song
    """

    trimmed = trim_series(
        song.time_series, song.sampling_rate, left, right
    )

    trimmed = Song(
        trimmed, song.sampling_rate, len(trimmed)
    )
    return trimmed


def trim_series(
        x: np.ndarray,
        sampling_rate: int,
        left: float,
        right: float
    ) -> np.ndarray:
    """
    Removes left and right ends of a series.

    Parameters:
        x: np.ndarray : time series
        sampling_rate: int : spacing between subsequent samples (1 / s)
        left: float : number of seconds to remove from the left
        left: float : number of seconds to remove from the right

    Returns:
        subsampled: np.ndarray : subsampled time series
    """

    i_left = int(sampling_rate * left)
    i_right = int(sampling_rate * right)

    trimmed = x[i_left:-i_right] + 0  # deepcopy

    return trimmed
