"""
Legendre polynomial expansion utilities.
"""

from typing import (
    Tuple,
    Union
)

import numpy as np
from scipy.special import eval_legendre

from mp3_script.mp3_types import Song


def expand_legendre_w(
        song: Song,
        order: int
    ) -> Song:
    """
    Wrapped song Legendre expansion utility.

    Parameters:
        song: Song : song!
        order: int : expansion order

    Returns:
        transformed: Song : tranformed song
            * `time_series`: np.ndarray : 
                the Legendre expansion coefficient
            * `sampling_rate` : float : samplig rate
                of the original song
            * `n_sample` : number of samples
                of the original song
    """

    coeffs = expand_legendre_ch(song.time_series, order)
    
    transformed = Song(
        coeffs, song.sampling_rate, song.n_sample
    )

    return transformed


def expand_legendre(
        y: np.ndarray,
        order: int
    ) -> Tuple[np.ndarray, int]:
    """
    Determines the Legendre polynomial expansion coeffcients if a series.

    Parameters:
        y: np.ndarray : series to expand with Legendre polynomials
        n: int : expansion order

    Returns:
        coefs: np.ndarray : expansion coefficients
         : float : dummy sampling rate
        n_points: int : number sumber of samples in y 
    """

    n_points = len(y)
    x_ = np.linspace(-1, 1, n_points)

    coeffs = np.zeros(order + 1, dtype=np.float64)

    for i in range(0, order + 1):
        poly = eval_legendre(i, x_)
        coeffs[i] = np.dot(y, poly) * (x_[1] - x_[0]) * (i + 0.5)

    return coeffs


def expand_legendre_ch(
        y: np.ndarray,
        order: int
    ) -> np.ndarray:
    """
    Determines the Legendre polynomial expansion coeffcients if a series.

    Parameters:
        y: np.ndarray : series to expand with Legendre polynomials
        n: int : expansion order

    Returns:
        coefs: np.ndarray : expansion coefficients
         : float : dummy sampling rate
        n_points: int : number sumber of samples in y 
    """

    n_points = len(y)
    x = np.linspace(-1, 1, n_points)
    x_nodes = _calc_chebysev_nodes(n_points)

    y_nodes = np.interp(x_nodes, x, y)

    coeffs = np.zeros(order + 1, dtype=np.float64)

    for i in range(0, order + 1):
        poly = eval_legendre(i, x_nodes)
        coeffs[i] = np.trapz(y_nodes * poly, x=x_nodes) * (2 * i + 1) / 2

    return coeffs


def _calc_chebysev_nodes(n: int) -> np.ndarray:
    """
    Calculate the Chebysev nodes.

    Parameters:
        n: int : number of nodes

    Returns;
        nodes: np.ndarray : node coordinates
    """
    nodes = np.cos(np.arange(n - 1, 0, - 1) / (n - 1) * np.pi)

    return nodes


def reconstruct_legendre_w(
        song: Song,
        n_sample: Union[int, None]
    ):
    """
    Reconstructs a time series from its Legendre coefficients.
    
    Parameters:
        song: Song : song!
        n_sample: Union[None, int] : number of samples
            if None. The number of samples will be that
            of the original song

    Returns:
        transformed: song : time series reconstructed
            with Legendre polynomials
    """

    duration = song.n_sample / song.sampling_rate

    sampling_rate = n_sample / duration

    y_hat = reconstruct_legendre(
        song.time_series, n_sample
    )

    transformed = Song(
        y_hat, sampling_rate, n_sample
    )

    return transformed


def reconstruct_legendre(
        coeffs: np.ndarray,
        n_sample: int
    ) -> Tuple[np.ndarray, float, int]:
    """
    Reconstructs a series with Legendre polynomials

    Parameters:
        coeffs: np.ndarray : Legendre expansion coefficients
        n_sample: int : length of the reconstructed series

    Returns:
        y_hat: np.ndarray : reconstructed series
    """

    y_hat = np.zeros(n_sample, dtype=np.float64)

    x = np.linspace(-1, 1, n_sample)

    for i, coeff in enumerate(coeffs):
        y_hat = y_hat + eval_legendre(i, x) * coeff

    y_hat = np.where(y_hat < 0, 0.0, y_hat)

    return y_hat
