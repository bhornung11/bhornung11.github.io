"""
Miscs utilities.
"""

import numpy as np


def create_ellipse(
        a: float,
        b: float,
        steps: int = 200,
        phase: float = 0.0
) -> np.ndarray:
    """
    Creates an ellipse.

    Parameters:
        a: float : major semiaxis
        b: float : minor semiaxis
        steps: int = 200 : number of points
        phase: float = 0.0 : starting angle

    Returns:
        x: np.ndarray : xy coordinates of an ellipse
    """

    phi = np.linspace(0, 2 * np.pi, num=steps, endpoint=False)
    phi += phase
    x = np.zeros((steps, 2), dtype=np.float64)
    x[:, 0] = a * np.cos(phi)
    x[:, 1] = b * np.sin(phi)

    return x
