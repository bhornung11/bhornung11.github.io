"""
Shortest path distances.
"""

from typing import (
    Callable,
    List
)

import numpy as np
import numba as nb


@nb.jit(nopython=True)
def calc_path_distance(
        u: np.ndarray,
        v: np.ndarray,
        fun_dist: Callable,
        fun_acc: Callable
) -> float:
    """
    Calculates the path length between vertices (1, 1) and (-1, -1)
    of the edge -- edge graph.

    Parameters:
        u: np.ndarray : sequence of vectors 1.
        v: np.ndarray : sequence of vectors 2.
        fun_dist: Callable : distance function between two vectors
        fun_acc: Callable : function that increments the path length

    Returns:
        path_length: float : path length
    """

    # set up an empty distance matrix
    m = len(u)
    n = len(v)
    d = np.zeros((m, n), dtype=np.float64)

    # starting vertex
    u0 = u[0]
    v0 = v[0]
    d[0, 0] = fun_dist(u0, v0)

    # top row
    for j in range(1, n):
        dist = fun_dist(u0, v[j])
        d[0, j] = fun_acc(d[0, j - 1], dist)

    # left column
    for i in range(1, m):
        dist = fun_dist(v0, u[i])
        d[i, 0] = fun_acc(d[i - 1, 0], dist)

    # fill in the distance matrix with the sortest path to each vertex
    for i in range(m):
        ui = u[i]

        for j in range(n):

            d_left = d[i, j - 1]
            d_up = d[i, j - 1]
            d_left_up = d[i - 1, j - 1]

            # choose shortest incoming path
            if (d_left <= d_up) and (d_left <= d_left_up):
                d_prev = d_left

            elif  (d_up <= d_left) and (d_up <= d_left_up):
                d_prev= d_up
            else:
                d_prev = d_left_up

            # increment the path length with the distance at the current point
            dist = fun_dist(ui, v[j])
            d[i, j] = fun_acc(dist, d_prev)

    path_length = d[-1, -1]
    
    return path_length


@nb.jit(nopython=True)
def calc_l2_dist(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the L2 distance.

    Parameters:
        x: np.ndarray : vector 1.
        y: np.ndarray : vector 2.

    Returns:
        dist: float : L2 distance
    """

    dist = 0
    for x_, y_ in zip(x, y):
        diff = x_ - y_
        diff *= diff
        dist += diff

    dist = np.sqrt(dist)

    return dist


@nb.jit(nopython=True)
def fun_sum(x: float, y: float) -> float:
    """
    Returns the sum of two objects.

    Parameters:
        x: float : quantity 1
        y: float : quantity 2

    Returns:
        : float : the sum of them
    """
    return x + y


@nb.jit(nopython=True)
def fun_max(x: float, y: float) -> float:
    """
    Returns the larger of two objects.

    Parameters:
        x: float : quantity 1
        y: float : quantity 2

    Returns:
        : float : the maximum of them
    """

    if x < y:
        return y
    return x



def wrap_calc_path_distance(
        fun_dist: Callable,
        fun_acc: Callable
) -> Callable:
    """
    Creates an instances of the shortes path length calculator
    with specified distance and accumulator functions.
    
    Parameters:
        fun_dist: Callable : distance function between two vectors
        fun_acc: Callable : function that increments the path length

    Returns:
        inner: Callable: calculator 
    """
    def inner(x, y):
        return calc_path_distance(x, y, fun_dist, fun_acc)
    return inner


calc_dtw_distance = wrap_calc_path_distance(calc_l2_dist, fun_sum)


calc_frechet_distance = wrap_calc_path_distance(calc_l2_dist, fun_max)
