
"""
Piecewise linear interpolation.
"""

from typing import Union

import numpy as np


def piecewise_interpolate(
        xy: np.ndarray,
        n_point: Union[int, None] = None,
        spacing: Union[float,None] = None
) -> np.ndarray:
    """
    Parameters:
        xy: np.ndarray : raw 2D coordinate sequence
        n_point: Union[int, None] : number of interpolants
        spacing: Union[float,None] : distance between interpolants

    Returns:
       xy_intp: np.ndarray : interpolated coordinate sequence
    """

    if n_point is not None and spacing is not None:
        raise ValueError(
            "`n_point` and `spacing` cannot be specified simultaneously."
        )

    xy_looped = np.vstack([xy[0], xy, xy[0]])
    distances = np.diff(xy_looped, axis=0)
    distances = np.cumsum(
        np.sqrt(np.sum(distances * distances, axis=1))
    )
    d_max = max(distances)

    if spacing is not None:
        if spacing <=0:
            raise ValueError("`spacing` must be positive")

        n_point = int(d_max / spacing)
        d_calc = np.arange(n_point) * spacing
    
    else:
        if n_point is None:
            raise ValueError("boo")
        if n_point <= 0:
            raise ValueError("`n_point` must be positive")

        d_calc = np.linspace(0, d_max, num=n_point)

    # find bracket
    xy_intp = []
    i = 0
    for d in d_calc:
        for j, d_ in enumerate(distances[i:]):
            if d <= d_:
                break
        i += j

        # linearly interpolate x and y separately
        x_low, y_low = xy_looped[i]
        x_hgh, y_hgh = xy_looped[i+1]
        d_low = distances[i-1]
        d_hgh = distances[i]

        factor = (d - d_low) / (d_hgh - d_low)
        x_intp = x_low + factor * (x_hgh - x_low)
        y_intp = y_low + factor * (y_hgh - y_low)
        xy_intp.append([x_intp, y_intp])

    xy_intp = np.array(xy_intp)

    return xy_intp
