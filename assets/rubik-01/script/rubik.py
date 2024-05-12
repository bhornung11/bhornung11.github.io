"""
Rubik colour transformation utilities.
"""

from typing import (
    Callable
)

import numba as nb
import numpy as np
from imageio import imread


def add_grid(
        image: np.ndarray,
        width_square: int,
        width_sticker: int
    ) -> np.ndarray:
    """
    Draws gridlines between pixels.
    
    Parameters:
        image: np.ndarray : image
        width_square: int : square width
        width_sticker: int : width of the painted area inside of the square
        
    Returns:
        gridded: np.ndarray : image with Rubik type grids
    """
    
    width_border = width_square - width_sticker
    if width_border % 2 != 0:
        raise ValueError("Square and sticker widths must differ by an even number")
        
    width_border = width_border // 2
    
    # make a blank image
    n, m, q = image.shape
    gridded = np.zeros((n * width_square, m * width_square, q))
    
    # copy enlarged pixels over
    # wb, wst + wb, 2 * wst + wb, ..., (n - 1) wst + wb
    for i in range(n):
        offset_x = i * width_square + width_border
        for j in range(m):
            offset_y = j * width_square + width_border
            gridded[
                offset_x: offset_x + width_sticker,
                offset_y: offset_y + width_sticker
            ] = image[i, j][None, None, :]
            
    return gridded

@nb.jit(nopython=True)
def dither(
        image: np.ndarray,
        palette: np.ndarray,
        func_dist: Callable,
        extent_x: int,
        extent_y: int,
        weights: np.ndarray
    ) -> np.ndarray:
    """
    Replaces the colour of an image from a palette.

    Parameters:
        image: np.ndarray : image
        palette: np.ndarray : palette
        func_dist: Callable,
        extent_x: int,
        extent_y: int,
        
    Returns:
        transformed: np.ndarray : image with replaced colour
    """
    
    transformed = np.zeros_like(image)
    
    n, m, _ = image.shape
    
    n_x = n // extent_x
    n_y = m // extent_y
    
    for i in range(n_x):
        offset_x = i * extent_x
        for j in range(n_y):
            offset_y = j * extent_y
            
            detail = image[offset_x:offset_x + extent_x, offset_y:offset_y + extent_y]
            
            r = choose_closest_colour(
                detail, palette, func_dist
            )
            
            transformed[offset_x:offset_x + extent_x, offset_y:offset_y + extent_y] = r
            
            # calculate error
            err = calc_rubikify_error(detail, r)
            
            # apply error
            apply_error(
                image, err, i, j, extent_x, extent_y, n_x, n_y, weights
            )

    
    return transformed


@nb.jit(nopython=True)
def calc_rubikify_error(
        image: np.ndarray,
        r: np.ndarray
    ) -> np.ndarray:
    """
    Calculates the quantisation error over an image.
    
    Parameters:
        image: np.ndarray : image
        r: np.ndarray : substitution colour
        
    Returns:
        err: np.ndarray : error colour
    """
    n, m, q = image.shape
    err = np.zeros(q)
    
    for i in range(n):
        for j in range(m):
            err = err + image[i, j] - r
    
    # get mean
    err /= (n * m)
    
    return err

@nb.jit(nopython=True)
def apply_error(
        image: np.ndarray,
        err: np.ndarray,
        i: int,
        j: int,
        extent_x: int,
        extent_y: int,
        n_x: int,
        n_y: int,
        weights: np.ndarray
    ) -> None:
    """
    Stucki dithering.
    
    Parameters:
        image: np.ndarray : image
        err: np.ndarray : quantisation error
        i: int : detail index
        j: int : detail index
        extent_x: int : detail size
        extent_y: int : detail size
        n_x: int : number of details
        n_y: int : number of details
        weights: np.ndarray : dither weigths
        
    Returns:
        None: applies dither in-place
    """
    
    for k in range(3):

        k1 = i  + k
        if  k1 >= n_x :
            continue

        offset_x = extent_x * k1

        for l in range(5):

            l1 = j + l - 2
            if l1 < 0 or l1 >= n_y:
                continue

            weight = weights[k, l]

            if weight == 0:
                continue
            
            offset_y = extent_y * l1
            
            for k2 in range(offset_x, offset_x + extent_x):
                for l2 in range(offset_y, offset_y + extent_y):
                    
                    a = image[k2, l2, 0] + err[0] * weight
                    if a < 0:
                        image[k2, l2, 0] = 0
                    elif a > 100:
                        image[k2, l2, 0] = 100
                    else:
                        image[k2, l2, 0] = a

                    a = image[k2, l2, 1] + err[1] * weight
                    if a < - 150:
                        image[k2, l2, 1] = - 150
                    elif a > 150:
                        image[k2, l2, 1] = 150
                    else:
                        image[k2, l2, 1] = a
  
                    a = image[k2, l2, 2] + err[2] * weight
                    if a < -150:
                        image[k2, l2, 2] = - 150
                    elif a > 150:
                        image[k2, l2, 2] = 150
                    else:
                        image[k2, l2, 2] = a

@nb.jit(nopython=True)
def rubikify(
        image: np.ndarray,
        palette: np.ndarray,
        func_dist: Callable,
        extent_x: int,
        extent_y: int,
    ) -> np.ndarray:
    """
    Replaces the colour of an image from a palette.

    Parameters:
        image: np.ndarray : image
        palette: np.ndarray : palette
        func_dist: Callable,
        extent_x: int,
        extent_y: int,
        
    Returns:
        transformed: np.ndarray : image with replaced colour
    """
    
    transformed = np.zeros_like(image)
    
    n, m, _ = image.shape
    
    n_x = n // extent_x
    n_y = m // extent_y
    
    for i in range(n_x):
        offset_x = i * extent_x
        for j in range(n_y):
            offset_y = j * extent_y
            r = choose_closest_colour(
                image[offset_x:offset_x + extent_x, offset_y:offset_y + extent_y],
                palette, func_dist
            )
            
            transformed[offset_x:offset_x + extent_x, offset_y:offset_y + extent_y] = r
    
    return transformed
    

@nb.jit(nopython=True)
def choose_closest_colour(
        image: np.ndarray,
        palette: np.ndarray,
        func_dist: Callable
    ) -> np.ndarray:
    """
    Chooses a colour from a palette
    which has the lowest distance to a set of pixels.

    Parameters:
        image: np.ndarray : (h*w*number of channels) image
        palette: np.ndarray : (number of colours * number of channels) palette
        func_dist: Callable : distance function

    Returns:
        c_star: np.ndarray : best matching colour over the image
    """
    n, m, _ = image.shape
    q = len(palette)
    
    dists = np.zeros(q)
    for k in range(q):
        r = palette[k]
        for i in range(n):
            for j in range(m):
                c = image[i, j]
            
                dists[k] = dists[k] + func_dist(c, r)
        
    d_min = dists[0]
    k_min = 0

    for k in range(1, q):
        d = dists[k]
        if d < d_min:
            d_min = d
            k_min = k

    return palette[k_min]


@nb.jit(nopython=True)
def calc_dist_rgb(
        colour1: np.ndarray,
        colour2: np.ndarray
    ) -> float:
    """
    Calculates the RGB colour distance.
    
    Parameters:
        colour1: np.ndarray : 1st colour
        colour2: np.ndarray : 2nd colour
        
    Returns:
        dist: float : distance
    """
    
    dist = 0.0
    for c1_, c2_ in zip(colour1, colour2):
        dist_ = c1_ - c2_
        dist = dist + dist_ * dist_
        
    return dist


@nb.jit(nopython=True)
def calc_dist_lab(
        colour1: np.ndarray,
        colour2: np.ndarray,
        k_l=1,
        k_c=1,
        k_h=1,
        k_1=0.045,
        k_2=0.015
    ) -> float:
    """
    Calculates the RGB colour distance.
    
    Parameters:
        c1: np.ndarray : 1st colour
        c2: np.ndarray : 2nd colour
        
    Returns:
        dist: float : distance  
    """
    
    l_delta = colour1[0] - colour2[0]

    c_1 = np.sqrt((colour1[1:] * colour1[1:]).sum(axis=-1))
    c_2 = np.sqrt((colour2[1:] * colour2[1:]).sum(axis=-1))
    c_delta = c_1 - c_2

    a_delta = colour1[1] - colour2[1]
    b_delta = colour1[2] - colour2[2]
    x = a_delta * a_delta + b_delta * b_delta - c_delta * c_delta
    if x < 0:
        x = 0
    h_delta = np.sqrt(x)

    s_l = 1
    s_c = 1 + k_1 * c_1
    s_h = 1 + k_2 * c_1
    
    term_1 = l_delta / (k_l * s_l)
    term_1 = term_1 * term_1
    
    term_2 = c_delta / (k_c * s_c)
    term_2 = term_2 * term_2
    
    term_3 = h_delta / (k_h * s_h)
    term_3 = term_3 * term_3
    
    dist = np.sqrt(term_1 + term_2 + term_3)

    return dist
