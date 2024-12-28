"""
Discrete dynamic time warping distance functions.
"""

from typing import (
    Callable,
    List,
    Tuple
)

import numba as nb
import numpy as np

from mp3_script.mp3_types import AlbumEntry


def calc_dmat_dtw(
        store_series: List[AlbumEntry],
        func: Callable
    ) -> np.ndarray:
    """
    Calculates the DTW distance matrix.

    Parameters:
        store_series: List[np.ndarray],
        func: Callable : ground distance function

    Returns:
        dmat: np.ndarray : DTW distance matrix
    """

    # calculate distance matrix
    n_point = len(store_series[0].song.time_series)
    n_series = len(store_series)

    x = np.linspace(-1, 1, n_point)

    dmat = np.full((n_series, n_series), np.nan)

    for i, entry1 in enumerate(store_series):
        series1 = entry1.song.time_series

        for j, entry2 in enumerate(store_series[i + 1:]):
            series2 = entry2.song.time_series

            d = dtw_discrete_dp(x, x, series1, series2, func)
            dmat[j + i + 1, i] = dmat[i, j + i + 1] = d

    return dmat


@nb.jit(nopython=True)
def dtw_discrete_dp(
        x1s:np.ndarray,
        x2s:np.ndarray,
        y1s:np.ndarray,
        y2s:np.ndarray,
        func_dist:Callable
    ) -> float:
    """
    Poitwise dynamic time warping distance
    """

    n1 = len(x1s)
    n2 = len(x2s)

    # previous row of the DTW matrix
    ds_north = np.zeros(n2)

    # current row of the DTW matrix
    ds_curr = np.zeros(n2)

    # two starting points
    ds_north[0] = func_dist(x1s[0], x2s[0], y1s[0], y2s[0])

    # first point of x1s aligned with all points of x2s
    for i in range(1, n2):
        d = func_dist(x1s[0], x2s[i], y1s[0], y2s[i])
        ds_north[i] = d + ds_north[i - 1]

    # second and later points of the first series
    for i in range(1, n1):

        x1 = x1s[i]
        y1 = y1s[i]

        # starting point of the second series
        d = func_dist(x1, x2s[0], y1, y2s[0])

        # only the north value is available
        ds_curr[0] = d + ds_north[0]

        # second and later points of the second series
        for j in range(1, n2):

            # distance between i--j
            d = func_dist(x1, x2s[j], y1, y2s[j])

            # find the smallest of the north/northwest/west DTW matrix elements
            d_min, _, _ = _find_min_dtw_precedessor(ds_north, ds_curr, i, j)

            ds_curr[j] = d + d_min

        # move one row down
        ds_north = ds_curr + 0.0

    n_max = n1
    if n2 > n1:
        n_max = n2

    dtw = ds_curr[-1] / n_max

    return dtw


@nb.jit(nopython=True)
def dtw_discrete_dp_with_coupling(
        x1s:np.ndarray,
        x2s:np.ndarray,
        y1s:np.ndarray,
        y2s:np.ndarray,
        func_dist:Callable
    ) -> float:
    """
    Poitwise dynamic time warping distance
    """

    n1 = len(x1s)
    n2 = len(x2s)

    # warping path
    coupling = np.zeros((n1 + n2 + 1, 2), dtype=np.int64)

    # full dtw matrix with padding
    dtw_mat = np.full((n1 + 1, n2 + 1), np.infty)
    dtw_mat[0, 0] = 0.0

    for i in range(1, n1 + 1):

        x1 = x1s[i - 1]
        y1 = y1s[i - 1]

        # second and later points of the second series
        for j in range(1, n2 + 1):

            # distance between i--j
            d = func_dist(x1, x2s[j - 1], y1, y2s[j - 1])

            # find the smallest of the north/northwest/west DTW matrix elements
            d_min, _, _ = _find_min_dtw_precedessor(dtw_mat[i- 1], dtw_mat[i], i, j)
            dtw_mat[i, j] = d + d_min

    n_max = n1
    if n2 > n1:
        n_max = n2

    dtw = dtw_mat[- 1, - 1] / n_max

    # 2) traceback to read out the coupling
    i = n1
    j = n2
    k = 1
    coupling[0, 0] = 1
    coupling[k, 0] = i
    coupling[k, 1] = j

    while i != 1 and j != 1:
        _, i, j = _find_min_dtw_precedessor(dtw_mat[i- 1], dtw_mat[i], i, j)
        k += 1
        coupling[0, 0] = k
        coupling[k, 0] = i
        coupling[k, 1] = j

    return dtw, coupling[1:k + 1] -1


@nb.jit(nopython=True)
def _find_min_dtw_precedessor(
        dtw_mat_row_prev: np.ndarray,
        dtw_mat_row_curr: np.ndarray,
        i: int,
        j: int
    ) -> Tuple[float, int, int]:
    """
    Determines the precedessor point in the dtw matrix 

    Parameters:
        dtw_mat_row_prev: np.ndarray : DTW matrix current row
        dtw_mat_row_curr: np.ndarray : DTW matrix previous row
        i: int : row index of the current position
        j: int : column index of the current position

    Returns:
        d_min: float : minimum previous distance
        i_min: int : rowin dex of the minimum previous distance
        i_max: int : column index of the minimum previous distance
    """

    # find the smallest of the north/northwest/west DTW matrix elements
    d_nw = dtw_mat_row_prev[j - 1]
    d_n = dtw_mat_row_prev[j]
    d_w = dtw_mat_row_curr[j - 1]

    if d_nw <= d_n:
        if d_nw <= d_w:
            d_min = d_nw
            i_min, j_min = i - 1, j - 1
        else:
            d_min = d_w
            i_min, j_min = i, j - 1
    else:
        if d_n <= d_w:
            d_min = d_n
            i_min, j_min = i - 1, j
        else:
            d_min = d_w
            i_min, j_min = i, j - 1

    return d_min, i_min, j_min
