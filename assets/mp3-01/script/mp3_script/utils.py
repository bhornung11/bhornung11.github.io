"""
Utility functions.
"""

import numpy as np


def make_time_points(
        time_series: np.ndarray,
        sampling_rate: int
    ) -> np.ndarray:
    """
    Calculates the times at which samples were taken. (s)
    Parameters:
        time_series: np.ndarray : time series
        sampling_rate: int : sampling rate (1 / s)

    Returns:
        times: np.ndarray : sample times
    """

    times = np.arange(len(time_series)) / sampling_rate
    return times
