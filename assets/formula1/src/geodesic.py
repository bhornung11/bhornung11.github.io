"""
Geodesic conversion utilities.
"""

from typing import Iterable, Union, Tuple

import numpy as np


def _calc_avg_lon_lat(
        lons: Iterable[float],
        lats: Iterable[float]
) -> Tuple[float]:
    """
    Calculates the mean longotude and latitude.
    
    Parameters:
        lons: array-like : longitudes
        lats: array-like : latitudes
        
    Returns:
        lon_avg: float : mean longitude
        lat_avg: float : mean latitude
    """
    lon_avg = np.mean(lons)
    lat_avg = np.mean(lats)
    
    return lon_avg, lat_avg


def _calc_radius_at_latitude(
        lat: Union[Iterable[float], float],
        radius: float
) -> float:
    """
    Calculates the radius along a latittude.
    
    Parameters:
        lat: float : lattitude
        radius: float : radius of the sphere
    """
    
    factor = np.cos(np.radians(lat))
    radius_local = factor * radius

    return radius_local


def _convert_lonlat_to_xy(
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
        radius: float,
        radius_local: float
) -> Tuple[float]:
    """
    Calculates the planar x, y distances between two points given
    on a sphere in terms of longitude and latitude.
    
    Parameters:
        lon1: float : longitude of the first point
        lat1: float : longitude of the second point
        lon2: float : latitude of the first point
        lat2: float : latitude of the second point
        radius: float : radius of the sphere
        radius_local: float : radius at the give latitude
        
    Returns:
        d_x: float distance perpendicular to the meridian
        d_y: float : distance along meridian
        
    Note, there is a single radius for all latitudes.
    """

    lon_r1 = np.radians(lon1)
    lat_r1 = np.radians(lat1)
    lon_r2 = np.radians(lon2)
    lat_r2 = np.radians(lat2)   
    
    d_x = (lon_r2 - lon_r1) * radius_local
    d_y = (lat_r2 - lat_r1) * radius
    
    return d_x, d_y


def convert_lonlat_to_xy(
        lons: Iterable[float],
        lats: Iterable[float],
        radius: float
) -> np.ndarray:
    """
    Convert a set of points on sphere given in terms
    of longitudes and latitudes to planar coordinates.
    
    Parameters:
        lons: Iterable[float] : longitudes
        lats: Iterable[float] : latitudes
        radius: float : radius of the sphere
        
    Returns:
        coords: np,ndarray : xy coordinates
    """
    
    # get centre of the point set
    lon_avg, lat_avg = _calc_avg_lon_lat(lons, lats)
    
    # get longitudinal radius
    r_local = _calc_radius_at_latitude(lat_avg, radius)
    
    # calculate all differences wrt the centre
    d_xs, d_ys = _convert_lonlat_to_xy(
        lons, lats, lon_avg, lat_avg, radius, r_local
    )
    
    coords = np.stack([d_xs[:-1], d_ys[:-1]], axis=1)
    
    return coords

