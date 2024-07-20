"""
Rectangle drawing.
"""

from collections import Counter

from typing import (
    Dict,
    List,
    Tuple
)

from encoding import (
    translate_to_colours
)


import numpy as np

def _arrange_rects(
        rectangles: List[np.ndarray],
        n_row: int,
        n_col: int,
        size_space_ver: int,
        size_space_hor: int
    ) -> np.ndarray:
    """
    Draws rectangles on a grid.
    
    Parameters:
        rectangles: List[np.ndarray],
        n_row: int,
        n_col: int,
        space_vert: int,
        space_hor: int
        
    Returns:
        arr: np.ndarray : image of arranged rectangles
    """
    
    n_rect = len(rectangles)
    n_slot = n_row * n_col
    
    if n_rect > n_slot:
        raise ValueError(
            f"More rectangles {n_rect} than slots {n_slot}"
        )
        
    size_rect_ver, size_rect_hor, _ = next(iter(rectangles)).shape
    
    dist_hor = size_space_hor + size_rect_hor
    dist_ver = size_space_ver + size_rect_ver
    
    size_hor = dist_hor * n_col + size_space_hor
    size_ver = dist_ver * n_row + size_space_ver

    arr = np.ones((size_ver, size_hor, 3), dtype=np.int64) * 255
    
    gen_rect = iter(rectangles)
    
    for i in range(n_row):
        i_start = i * dist_ver + size_space_ver
        i_end = i_start + size_rect_ver

        for j in range(n_col):
            rect = next(gen_rect, None)
            if rect is None:
                break
                
            j_start = j * dist_hor + size_space_hor
            j_end = j_start + size_rect_hor
            arr[i_start:i_end, j_start:j_end] = rect
        else:
            continue
        break

    return arr


def _make_coloured_rect(
        w: int,
        l: int,
        colours: np.ndarray,
        colour_frame: np.ndarray = np.array([0, 0, 0], dtype=np.int64),
        width_frame: int = 1
    ) -> np.ndarray:
    """
    Creates rectangular image composed of coloured bands.
    
    Parameters:
        w: int : width
        l: int : length
        colours: np.ndarray : RGB colours as integers
        colour_frame: np.ndarray = BLACK : frame colour
        width_frame: int = 1 : frame width
    
    Returns:
        rectangle: np.ndarray : a rectangle image composed of colourerd bands
    """
 
    n = len(colours)
    idcs = np.array_split(np.arange(l), n)
    
    rectangle = np.zeros((l, w, 3), dtype=np.int64)
    
    for idcs_, colour in zip(idcs, colours):
        rectangle[idcs_] = colour
        
    for idcs_ in idcs:
        i = idcs_[0]
        rectangle[i:i+width_frame] = colour_frame
        
    # add horizontal borders
    rectangle[:width_frame] = colour_frame
    rectangle[-width_frame:] = colour_frame
    
    # add vertical borders
    rectangle[:, 0, :] = colour_frame
    rectangle[:, -1, :] = colour_frame

    return rectangle


def make_coloured_rect_hist_from_encoded(
        encoded: List[Tuple[int]],
        mapping_colour: Dict[int, np.ndarray],
        n_show: int,
        width: int = 10,
        height_min: int = 10
    ) -> List[np.ndarray]:
    """
    Creates a list if banded rectangles based on codes.
   
    Parameters:
        encoded: List[Tuple[int]] : list of codes tuples
        mapping_colour: Dict[int, np.ndarray] : code to colour mapping
        n_show: int : how many rectangles to show
        
    Returns:
        rects_hist: List[np.ndarray]
    """    
    rects_histo = [
        _make_coloured_rect(
            width, count + height_min,
            translate_to_colours(encoded_, mapping_colour)
        )
        for encoded_, count in Counter(encoded).most_common(n_show)
    ]
    
    return rects_histo


def draw_coloured_rect_histo(
        rects: List[np.ndarray],
        size_space_hor: int
    ) -> np.ndarray:
    """
    Draws a histogram from banded rectangles.
    
    Parameters:
        rects: List[np.ndarray],
        size_space_hor: int
        
    Returns:
        arr: np.ndarray : image
    """
    n = len(rects)
    h_max = max(rect.shape[0] for rect in rects)
    w_max = max(rect.shape[1] for rect in rects)
    
    shift = w_max + size_space_hor
    size_arr_hor = n * shift + size_space_hor
    size_arr_ver = h_max + shift
    size_arr_ver = int(np.ceil(size_arr_ver / 100)) * 100

    arr = np.full((size_arr_ver, size_arr_hor, 3), 255, dtype=np.int64)

    starts = []
    j_start = size_space_hor
    for i, rect in enumerate(rects):
        h, v, _ = rect.shape
        j_end = j_start + v
        starts.append(j_start)

        arr[-h:, j_start:j_end] = rect
        j_start = j_start + shift
        
    return arr, starts

