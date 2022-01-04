"""
Geometry utilities
"""

import numpy as np


def calc_rotation_angle(
        x1: np.ndarray,
        x2: np.ndarray
) -> float:
    """
    Calculates the rotation angle that bets overlays
    two sequences of vectors

    Parameters:
        x1: np.ndarray : first vector
        x2: np.ndarray : second vector

    Returns:
        angle: float : angle of rotation
    """

    angle = np.arctan(
        (x1[:, 1] * x2[:, 0] - x1[:, 0] * x2[:, 1]).sum() / (x1 * x2).sum()
    )

    return angle


def rotate_2D(
        x: np.ndarray,
        angle: float
) -> np.ndarray:
    """
    Rotates a sequence of 2D points in 2D.

    Parameters:
        x: np.ndarray : xy coordinates
        angle: float : angle of rotation

    Returns:
        x_rotated: np.ndarray : the sequence of rotated points
    """

    x_rotated  = np.zeros_like(x)
    x_rotated[:, 0] = x[:, 0] * np.cos(angle) - x[:, 1] * np.sin(angle)
    x_rotated[:, 1] = x[:, 0] * np.sin(angle) + x[:, 1] * np.cos(angle)

    return x_rotated


def shift_ref(x: np.ndarray) -> np.ndarray:
    """
    Shifts a vector so that its first element is the origin.

    Parameters:
        x: np.ndarray : VECTOR!

    Returns:
        shifted: np.ndarray : shited vector
    """
    shifted = x - x[0][None, :]
    return shifted


def rotate_ref(x: np.ndarray) -> np.ndarray:
    """
    Rotates a vector so that it is aligned along the
    vector between its first and last elements

    Parameters:
        x: np.ndarray : VECTOR!

    Returns:
        rotated: np.ndarray : rotated vector
    """
    
    dx, dy = x[-1] - x[0]
    alpha = np.arctan2(dy, dx)
    rotated = rotate_2D(x, -alpha)
    
    return rotated


def reflect_ref(x: np.ndarray) -> np.ndarray:
    """
    Reflects a vector so that its mean in the 2nd dimension
    is in the positive half plane

    Parameters:
        x: np.ndarray : VECTOR!

    Returns:
        reflected: np.ndarray : reflected vector
    """
    
    mean_y = np.mean(x[:, 1])
    reflected = np.copy(x)
    if mean_y < 0:
        reflected[:, 1] *= -1
        
    return reflected


def centre(x: np.ndarray) -> np.ndarray:
    """
    Centres a vector so that its mean is the origin.

    Parameters:
        x: np.ndarray : VECTOR!

    Returns:
        centred: np.ndarray : centred vector
    """
    centred = x - np.mean(x, axis=0)
    return centred
