"""
Rudimetary subsampling utilities.
"""

import numba as nb
import numpy as np

from mp3_script.mp3_types import Song


def smooth_subsample_w(
        song: Song,
        spacing: float,
        width: int
    ) -> Song:
    """
    Wrapped subsampling function.

    Parameters:
        song: Song : song!
        spacing: float : spacing between subsampling points (s)
        width: int : interval width to sample from around sampling points (s)

    Returns:
        transformed: Song : subsamled song
    """

    subsampled, sampling_rate_trf = smooth_subsample(
        song.time_series, song.sampling_rate, spacing, width
    )

    transformed = Song(
        subsampled, sampling_rate_trf, len(subsampled)
    )
    return transformed


def smooth_subsample(
        x: np.ndarray,
        sampling_rate: int,
        spacing: float,
        width: int
    ) -> np.ndarray:
    """
    Subsampling function which convolves the time series.

    Parameters:
        x: np.ndarray : time series
        sampling_rate: int : spacing between subsequent samples (1 / s)
        spacing: float : spacing between subsampling points (s)
        width: int : interval width to sample from around sampling points (s)

    Returns:
        subsampled: np.ndarray : subsampled time series
    """

    weights = get_random_weights(width, sampling_rate)

    idcs = get_sample_indices(x, sampling_rate, spacing)

    subsampled = convolve_at_indices(x, idcs, weights)

    sampling_rate_trf = 1 / spacing

    return subsampled, sampling_rate_trf


@nb.jit(nopython=True)
def convolve_at_indices(
        x: np.ndarray,
        idcs: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
    """
    Convolves a series with weigths at the specified indices.
    The window around a point is two-sided.

    Parameters:
        x: np.ndarray : series
        idcs: np.ndarray : indices to convolve at
        weights: np.ndarray : weights

    Returns:
        x_out: np.ndarray : convolved signal
    """

    x_out = np.full(len(idcs), np.nan)
    n_s = len(x)
    n_w = len(weights)

    for i_write, i in enumerate(idcs):

        tally = 0.0
        n_left = n_w
        if n_w > i:
            n_left = i

        n_right = n_w
        if n_w + i >= n_s:
            n_right = n_s - i

        for j in range(n_left):
            tally += x[i - j] * weights[j]

        for j in range(1, n_right):
            tally += x[i + j] * weights[j]

        x_out[i_write] = tally

    return x_out


def get_random_weights(
        width: float,
        sampling_rate: float
    ) -> np.ndarray:
    """
    Creates an array of random numbers normalised to unit

    Parameters:
        width: float : interval width
        sample_rate: float : sampling rate

    Returns:
        w: np.ndarray : weights
    """

    w = np.random.rand(int(width * sampling_rate))
    w = w / w.sum() / 2

    return w


def get_sample_indices(
        x: np.ndarray,
        sampling_rate: int,
        spacing: float
    ) -> np.ndarray:
    """
    Determines the subsample indices.

    Parameters:
        x: np.ndarray : time series
        sampling_rate: int : time series sampling rate (1 / s)
        spacing: float : distance between subsequent subsamples (s)
    Returns:
        indices: np.ndarray : indices at which subsamples are taken
    """

    n = len(x)
    spacing_index = int(sampling_rate * spacing)
    indices = np.arange(0, n, spacing_index)

    return indices
