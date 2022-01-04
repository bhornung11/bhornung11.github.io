"""
Plotting utilities.
"""

from typing import(
    Any,
    Dict,
    Iterable,
    List,
    Tuple
)

import numpy as np
import matplotlib.pyplot as plt


def plot_pair_from_seqs(
        ax: plt.axes,
        i: int,
        j: int,
        seq: Iterable[np.ndarray],
        names=None
    ) -> None:
    """
    Plots two 2D lines from a sequence of them.

    Parameters:
        ax: plt.axes : axis to draw on
        i: int : index of the first line
        j: int : index of the second line
        seq: Iterable[np.ndarray] : sequence of lines

    Returns:
        None
    """
    
    ax.plot(*seq[i].T, c="blue")
    ax.plot(*seq[j].T, c="purple")


def plot_track(
        ax: plt.axes,
        line: np.ndarray
    ) -> None:
    """
    Plots a closed track.

    Parameters:
        ax: plt.axes : axis to draw on
        line: np.ndarray : 2D line

    Returns:
        None
    """
    
    ax.plot(*line.T, lw=0.5, c="#999999")
    ax.scatter(*line[0], c="k")
    ax.scatter(*line[-1], c="r")
    ax.scatter(*np.mean(line, axis=0), c="#1133cc")
    

def plot_pair_from_result_dict(
        ax: plt.axes,
        i: int,
        j: int,
        results: Dict[Tuple[int, int], Any],
        names: List[str]
    ) -> None:
    """
    Draws a pair of tracks from a dictionary of results.

    Parameters:
        ax: plt.axes : axis to draw on
        i: int : index of the first track
        j: int : index of the second track
        results: Dict[Tuple[int, int], Any] : result container
        names: List[str] : list of entry names

    Returns:
        None
    """

    try:
        res = results[(i, j)]
        ref = res.start.T
        rot = res.rotated.T
    except:
        try:
            res = results[(j, i)]
            ref = res.rotated.T
            rot = res.start.T
        except:
            raise
            
    ax.plot(*ref, c="blue")
    ax.plot(*rot, c="purple")
    title = f"{names[i]}\n{names[j]} \n score: {res.score:2.2f}"
    ax.set_title(title)
