"""
Distance matrix query functions.
"""

from typing import Tuple

import numpy as np


def find_dmat_minrow(dmat: np.ndarray) -> int:
    """
    Finds the row of minimum sum.

    Parameters:
        dmat: np.ndarray : matrix

    Returns:
        i: int : index of the minimum sum row
    """
    i = np.argmin(dmat.sum(axis=1))

    return i


def find_dmat_maxrow(dmat: np.ndarray) -> int:
    """
    Finds the row of maximum sum.

    Parameters:
        dmat: np.ndarray : matrix

    Returns:
        i: int : index of the maximum sum row
    """

    i = np.argmax(dmat.sum(axis=1))

    return i


def find_dmat_max(dmat: np.ndarray) -> Tuple[float, int, int]:
    """
    Finds the maximum of a square matrix.
    
    Parameters:
        dmat: np.ndarray : matrix
        
    Returns:
        dmax: float : maximum
        i: int : row index
        j: int : column index
    """

    n = len(dmat)
    ij = np.argmax(dmat)
    i = ij // n
    j = ij % n
    
    dmax = dmat[i, j]
    
    return dmax, i, j


def find_dmat_min(dmat: np.ndarray) -> Tuple[float, int, int]:
    """
    Finds the minimum of a square matrix.
    
    Parameters:
        dmat: np.ndarray : matrix
        
    Returns:
        dmin: float : minimum
        i: int : row index
        j: int : column index
    """

    n = len(dmat)
    dmin = np.inf
    
    for i in range(1, n):
        for j in range(i + 1, n):
            if dmat[i, j] < dmin:
                dmin = dmat[i, j]
                im, jm = i, j
    
    return dmin, im, jm


def find_minmax_in_row(dmat, i, find_max=False):
    """
    
    """
    if find_max:
        j = np.argmax(dmat[i])
        dval = dmat[i, j]
        return dval, i, j
    
    dmin = np.inf
    for k, dval in enumerate(dmat[i]):
        if k == i:
            continue
            
        if dval < dmin:
            dmin = dval
            j = k
    dval = dmin

    return dval, i, j
