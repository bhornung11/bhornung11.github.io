# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: sx-env
#     language: python
#     name: sx-env
# ---

# +
from typing import (
    Any,
    Generator,
    Iterable,
    List,
    Tuple,
    Union
)

from fit2gpx import Converter
import os.path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, DotProduct


# -

# ## Functions
#
# ### Misc

# +
def make_yielder_df(df: pd.DataFrame, cols: Union[List, None]) -> Generator:
    """
    Creates a generator of a dataframe.
    
    Parameters:
        df: pd.DataFrame
        cols: Union[List, None] : columns to in the yielded dataframe
        
    Returns:
        inner: Generator : generator of the same dataframe
    """
    
    def inner() -> pd.DataFrame:
        """
        Infinite sequence generator of a dataframe.
        
        Yields:
            unnamed: pd.DataFrame : a copy of the dataframe
        """
        while True:
            if cols:
                yield df.copy()
            else:
                yield df[cols].copy()

    return inner()

def make_yielder_palette(
        colours: Iterable[str]
    ) -> Generator:
    """
    Creates an infinite palette supply
    
    Parameters:
        colouts: Iterable[str] : sequence of colours
        
    Yield:
        sequence: Generator : generator of colours (palette)
    """
    def sequence() -> str:
        """
        Palette. 
        
        Yields:
            c: str : colours
        """
        for c in colours:
            yield c

    while True:
        yield sequence()


# -

# ### Geodesic and physical

# +
def calc_distances_from_coords(
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        elevations: np.ndarray
    ) -> np.ndarray:
    """
    Calculates the distances between the successive points
    of a sequence of points.
    
    Parameters:
        latitudes: np.ndarray : latitudes in ddegrees
        longitudes: np.ndarray : longitudes in degrees
        elevations: np.ndarray : elevations in metres
        
    Returns:
        distance: np.ndarray : distances in metres
    """

    deg_to_m = 111319
    
    n_points = len(latitudes)
    
    distances = np.full(n_points, np.nan)
    

    proj = np.cos(np.radians(latitudes[1:] + latitudes[:-1]) / 2)

    dx = (latitudes[1:] - latitudes[:-1]) * deg_to_m
    dy = (longitudes[1:] - longitudes[:-1]) * proj * deg_to_m
    dz = (elevations[1:] - elevations[:-1])

    distances[1:] = np.sqrt(dx * dx + dy * dy + dz * dz)
    
    return distances


def calc_cartesian_coord_diffs(
        latitudes1: np.ndarray,
        latitudes2: np.ndarray,
        longitudes1: np.ndarray,
        longitudes2: np.ndarray,
        elevations1: np.ndarray,
        elevations2: np.ndarray
    ) -> Tuple[np.ndarray]:
    """
    Calculates the distances between the successive points
    of a sequence of points.
    
    Parameters:
        latitudes1: np.ndarray : latitudes in ddegrees
        latitudes2: np.ndarray : latitudes in ddegrees
        longitudes1: np.ndarray : longitudes in degrees
        longitudes2: np.ndarray : longitudes in degrees
        elevations1: np.ndarray : elevations in metres
        elevations2: np.ndarray : elevations in metres
        
    Returns:
        dx: np.ndarray : x deviations in metres
        dy: np.ndarray : y deviations in metres
        dz: np.ndarray : z deviations in metres
    """

    deg_to_m = 111319

    proj = np.cos(np.radians(latitudes1 + latitudes2) / 2)

    dx = (latitudes1 - latitudes2) * deg_to_m
    dy = (longitudes1 - longitudes2) * proj * deg_to_m
    dz = (elevations1 - elevations2)

    return dx, dy, dz


# -

# ### Smoothing

def calc_velocity_thresh(
        distances: np.ndarray,
        delta_ds: np.ndarray,
        times: np.ndarray,
        error: float,
        factor: float,
        staggered: bool = False
    ) -> Tuple[np.ndarray]:
    """
    Calculates the velocity at distance interval which are at least
    long as a multiple of the distance error.
    
    Parameters:
        distances: np.ndarray : distances (lame to have this included)
        delta_ds: np.ndarray : distance differences
        times: np.ndarray : elapsed time
        error: float : estimated distance error
        factor: float : error multiplicator
        staggered: bool = False : whether to block or roll
        
    Returns:
        ds_sel: np.ndarray : distances
        times_sel: np.ndarray : times at the velocities are calculated
        velocities: np.ndarray : velocities
    """
    
    ds_sel = np.zeros_like(delta_ds)
    velocities = np.zeros_like(delta_ds)
    times_sel = np.zeros_like(times)
    
    thresh = error * factor
    n = len(delta_ds)
    
    # iterate
    i_curr = 1
    i_entry = 0
    t_prev = times[0]
    t_curr = 0
    path_length = 0.0
    
    i = i_curr
    
    while i_curr < n and i < n:
        # starting point of the investigation
        i = i_curr

        # step forward
        while True:
                       
            path_length += delta_ds[i]
        
            if path_length > thresh:
            
                # calulate
                t_curr = times[i]

                velocity = path_length / (t_curr - t_prev)
            
                # save
                ds_sel[i_entry] = distances[i]
                times_sel[i_entry] = t_curr
                velocities[i_entry] = velocity * (60 / 1000)

                i_entry += 1

                # reset counters
                path_length = 0.0
                
                # where to start the new calculation
                if staggered:
                    i_curr = i + 1
                else:
                    i_curr += 1

                t_prev = times[i_curr - 1]
                break
                
            i += 1

            # last path
            if i_curr == n or i == n:
                break
                
    return ds_sel[:i_entry], times_sel[:i_entry], velocities[:i_entry]


# # Keep your pace!
#
#
# This blog post discusses the properties of pace and its calculation from GPS records.
#
# ## Notes
#
# The raw notebook is available here. The scripts utilised are stored in this repository. The term `GPS` will be used synonymously any global navigational satellite system, such as GPS itself, Gallileo etc. The data presented below obtained through the Global Positioning System.
#
# ## Introduction
#
# The pace is the ratio of a time interval and the distance traveled under it. As such, it is almost the inverse of the speed. We will see soon why it is not its exact inverse.
#
# Any speed, $v$, that is the magnitude of a velocity of a body is bounded from below and above:
#
# $$
#     v \in [0, c_{0}] \, .
# $$
#
# It attains its lowest value, zero, when the body rests. The upper bound is the speed on light in vacuum. If that body happens to belong to a runner, no matter how lightweight they are, $c_{0}$ cannot be reached because of their mass.
#
# On the contrary, the pace, $p$ is unbounded from above:
#
# $$
#     p \in [ \frac{1}{c_{0}}, \infty) \, .
# $$
#
# Its lowest possible value is the recriprocal of the speed of light in vacuum. It is not defined at zero velocities, that is when a body is resting. As a consequence, it is almost the inverse of the speed apart from when that is zero.
#
# There is one more important issue with pace. The definition and the common notion of the term "pace" have different meanings and implied behaviour. Pace and speed are synonims in the everyday parlance. Larger pace means larger speed means faster movement. When referring to the calculated pace, larger pace indicates a smaller speed thus slower movement. The smaller the pace the faster the body.
#
# A popular platform seemingly resolves this contradiction by plotting the pace on a coordinate system where the values increase downwards (Figure 1.).
#
# ## Why to use pace?
#
# Pace and speed carry the same information when both defined. In some cases, it is seemingly easier to perform simple arithmetichal operations with pace. For instance, the time required, $\Delta t$ to complete a course of given length, $s$ at a constant pace is returned by a multiplication:
#
# $$
#     \delta t = p \cdot s \, .
# $$
#
# A division is needed when working with speeds:
#
# $$
#     \delta t = \frac{s}{v} \, .
# $$
#
# This ease disappears altogether when one wishes to calculate the pace required to complete a course during a desired time:
#
# $$
#     v = \frac{s}{\delta t} \\
#     p = \frac{\delta t}{ s}
# $$
#
# In reality, the mind attached to the moving body in question performs a succession of multiplications to estimate the pace needed, which amounts to a division.
#
# ## Estimation of pace from GPS tracks
#
# ### Problem definition
#
# The pace time series looks rather jaggedy. There are smaller downward spikes are narrow and long troughs which makes it a somewhat complicated to gather the pace characteristic to a given time interval. We set out to examine whether it is possible to achive this. If so, 
#
# A number question arises:
# * what are the spikes and trough due to?
# * should they be handled?
# * if so, how?
# * most importantly how reliable the GPS tracks are?
#
# ## Data analysis
#
# ### Data source
#
# The device converts the timestamps received from the satellites to lattitude, longitude and altitude in a map datum. A map datum is an ellipsoid model of the Earth. There are about 2,800 of them depending on the accurary required by the application it is used by. It is assumed that the datum in which the coordinates are referenced is accurate enough (fraction of a meter) over the entire track. Since the longest distance between two points is in the order of ten kilometres, it is a valid assumption.
#
#
# ### Data conversion
#
# The coordinates along with their time of recording are saved in `.fit` format. They are converted to tabulated data with the [fit2gpx](https://github.com/dodo-saba/fit2gpx).

# +
# constants

COLS = ["latitude", "longitude", "timestamp", "altitude"]

UNITS = {
    "altitude": "m",
    "latitude": "degrees",
    "longitude": "degrees",
    "dx": "m",
    "dy": "m",
    "dz": "m",
    "delta-t" : "min",
    "delta-d": "m",
    "distance": "km",
    "time": "h",
    "pace": "(min / km)",
    "velocity": "(km / h)"
}

# +
folder = "/home/bhornung/Documents/gps"
file = "2024-01-21-09-28-00.fit"
path_fit = os.path.join(folder, file)
path_gpx = os.path.splitext(path_fit)[0] + "+.gpx"

conv = Converter()
conv.fit_to_gpx(path_fit, path_gpx)

df_raw = conv.fit_to_dataframes(path_fit)[1]
                
yielder = make_yielder_df(df_raw, COLS)

# +
# calculations
df_trace = next(yielder)

# in minutes
df_trace["timestamp"] = (
    df_trace["timestamp"] - df_trace.loc[0, "timestamp"]
).dt.total_seconds() / 60

# in minutes
df_trace["delta-t"] = df_trace["timestamp"].diff()

# in metres
df_trace["delta-d"] = calc_distances_from_coords(
    df_trace["latitude"].values, df_trace["longitude"].values, df_trace["altitude"].values
)

# in kms
df_trace["distance"] = df_trace["delta-d"].cumsum() / 1000

# in km / h
df_trace["velocity"] = (df_trace["delta-d"] / 1000) / (df_trace["delta-t"] / 60)

# in min / km
df_trace["pace"] = df_trace["delta-t"] / (df_trace["delta-d"] / 1000)
# -

# ### Data overview
#
# The coordinates are plotted in Figure 1. The walk was a circular because the starting and ending longitude and latitude coordinates are identical. There is about fifteen metres difference in altitude, however. 

# +
labels = ["latitude", "longitude", "altitude"]
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for ax, label in zip(axes, labels):
    ax.scatter(df_trace["timestamp"], df_trace[label], s=0.2, color="navy")
    ax.grid(True); ax.set_ylabel(f"{label} / {UNITS[label]}"); ax.set_xlim(left=0)

axes[-1].set_xlabel(f"time / {UNITS['time']}")
# -

# Figure 1. Time series overview of the walk. The latitude (top panel), the longitude (middle panel) and altitude (bottom panel).

# ### Calculation of distance, velocity, pace
#
# It is assumed that the error of the timestamp is negligible when compared to that of the coordinates. The distances were computed using the Pythagoean theorem, that is the coordinate space is locally not curved which is a good approximation when the points are only a few metres away. The formulae assume a spherical earth. They are listed below.
#
#
# $$
# \begin{eqnarray}
#     i & \in & \mathcal{I}= [1, N] \\
#     \forall i, j & \in & \mathcal{I}, i < j : t_{i} < t_{j} \\
# \end{eqnarray}
# $$
#
# $$
# \begin{eqnarray}
#         i & \in & [2, n] \\
#     \Delta \lambda _{i} & = & \lambda_{i} - \lambda_{i - 1} \\
#     \Delta \phi_{i} & = & \phi_{i} - \phi_{i - 1} \\
#     \Delta z _{i} & = & z_{i} - z_{i - 1} \\
#     \Delta x_{i} & = &  \cos \left[ 
#                 \frac{\phi_{i} + \phi_{i - 1}}{2}
#         \right]
#         \frac{\Delta \lambda}{2 \pi} R
#         \\
#     \Delta y_{i} & = & \frac{\Delta \phi_{i}}{2 \pi} R \\
#     \Delta s_{i} & = & \left[ \Delta x_{i}^{2} + \Delta y_{i}^{2} + \Delta z_{i}^{2} \right]^{\frac{1}{2}}
# \end{eqnarray}
# $$
#
# The velocity and the pace are calculated as finite differences:
#
# $$
#     \begin{eqnarray}
#         i & \in & [2, n] \\
#         \Delta t_{i} & = & t_{i} - t_{i - 1} \\
#         v_{i} & = & \frac{\Delta s_{i}}{\Delta t_{i}} \\
#         p_{i} & = & \frac{\Delta t_{i}}{\Delta s_{i}}
#     \end{eqnarray}
# $$
#
# ### Overview
#
# Figure 2. displays the progression of distance, velocity and pace along with the sampling times. Histograms are also provided. The position is determined at most at every twelve second apart from a few instances. It is not shown, but the pair plots reveal that the sampling frequency is correlated with the speed. The faster one moves the more frequently they are located.
#
# The series as a fuction of the distance and time are similar. This is due to the fact that the speed varies little. When the speed goes down there is a dilation along the time axis. For instance, compare the plots of the pace around thirteen kilometres or hundred thirty minutes. The points are further apart in the latter plot.
#
# The downside of the pace is immediately obvious. When the speed is low the pace approaches large values creating sharp upward spikes (downward in the orange platform). The speed is, on the contrary, confined to a much smaller range.

# +
fix, axes = plt.subplots(4, 3, figsize=(12, 8))#, sharex=True)

labels = ["delta-t", "delta-d", "velocity", "pace"]

for ax, label in zip(axes[:, 0].flat, labels):
    ax.scatter(
        df_trace["distance"], df_trace[label],
        c="navy", s=2
    )
    ax.grid(True); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.grid(True); ax.set_ylabel(f"{label} / {UNITS[label]}")


# time series
for ax, label in zip(axes[:, 1].flat, labels):
    ax.scatter(
        df_trace["timestamp"], df_trace[label],
        c="navy", s=2
    )
    ax.grid(True); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    
# histograms
for ax, label in zip(axes[:, 2].flat, labels):
    
    if label == "pace":
        bins = np.logspace(
            np.log10(np.nanmin(df_trace[label])),
            np.log10(np.nanmax(df_trace[label]))
        )
    else:
        bins=50

    ax.hist(
        df_trace[label],
        color="navy", bins=bins, orientation='horizontal'
    )
    ax.grid(True); ax.set_xlim(left=0); ax.set_ylim(bottom=0)


    
ax = axes[-1, 0]
ax.set_xlabel(f"distance / {UNITS['distance']}")
ax.set_yscale("symlog", linthresh=20)


ax = axes[-1, 1]
ax.set_xlabel(f"time / {UNITS['time']}")
ax.set_yscale("symlog", linthresh=20)

ax = axes[-1, 2]
ax.set_xlabel("count")
ax.set_yscale("symlog", linthresh=20)
# -

# Figure 2. The sampling time intervals (top row), distances (secnd row), velocities (third row) and pace (bottom row) as a function of total distance (left column), time (middle column), and their histograms (right column).

# ### Analysis -- cause of the spikes
#
# When the walker is opening a gate, consulting the map -- especially if that is an OS Landranger one --, stuck in mud, or just rereshing themselves they are almost stationary. Hence the pace will go up. These temporary stoppages are easy to identify and exclude from the plots. Three questions arises, though:
#
# * How to identify short pauses?
# * What to do with them?
# * in general, are we interested in the overall pace, no matter what the actions are, or we seek to characterise the periods where there was intended movement?
#
# The last question leads to the problem of granularity. What is the length of the time during which the average pace should be calculated? There are three factors that can guide us in determining the length.
# * the time which is needed to change the pace substantially
# * the sampling resolution of the device used
# * precision of the coordinate measurements
#
# The speed of the movement can increase or decrease in a couple of strides, that in a span of a few seconds. The median sampling time is about five seconds for this walk. We naturally wish to use a percentile close to hundred so that it is ensured that there are more than one measurements in a window.
#
# The third point warrants its own paragraph.
#
# ### Error of the calculated distances
#
# Each measurement point has an associated error. A bias and a scatter, quantified by accuracy and precision It is assumed that the bias of any coordinate is constant throughout the course; and it is small enough no to introduce systematic but changing error. 
#
# Garmin states "Garmin GPS receivers are accurate to within 15 meters (49 feet) 95% of the time with a clear view of the sky.". That means the position measured by the device containing the receiver 95 out of 100 times will be at most 15 away from the true position. It does not state how the measured positions distributed in this circle. Whether they are spread all over it uniformly -- no bias, low precision, or concentrated around a point not being the true position -- bias, high precision.
#
#
# Scatter is treated as homogenous. This assumption may easily be invalidated as it depends on the availability of the satellites. If the visibility of the sky changes substantially the precision may not be uniform. E.g. it will be lower in a bottom of a canyon or in a built up area, under a motorway bridge then out in the fields among the cows (or sheep when in Wales).
#
# #### Determination of the measurement error
#
# In order for determining the precision, a large number of repeated measurements carried out in the same spot is required. Sadly enough, the device used stops collecting data when stationary. It was therefore affixed to a pendulum whose maximum deviation was about 0.3 metres. As long as the scatter is much smaller or larger this length the movement along this short path should be straightforward to deal with.
#
# The following steps were performed in order to obtain the estimated uncertainties
# * the stationary part of the timeseries were kept
# * the outliers (if existed) removed
# * it was assumed that the sample mean is an unbiased estimator of the true location. This is an incredibly strong assumption, but we can feet confident sometimes. (It is sufficient to require that the bias is constant as mentioned before).
# * the means were calculated
# * the means were subtracted from the time series
# * the differences were converted to calculated the Cartesian coordinate deviations
# * correlations checks (none)
#
# The histograms of the Cartesian errors are plotted in Figure 3.

# +
folder = "/home/bhornung/Documents/gps"
file = "2024-01-31-08-22-15.fit"
path_fit = os.path.join(folder, file)
path_gpx = os.path.splitext(path_fit)[0] + "+.gpx"

conv = Converter()
conv.fit_to_gpx(path_fit, path_gpx)

df_raw_error = conv.fit_to_dataframes(path_fit)[1]

# +
labels = ["latitude", "longitude", "altitude"]
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for ax, label in zip(axes, labels):
    ax.scatter(df_raw_error["timestamp"], df_raw_error[label], s=5.2, color="navy")
    ax.grid(True); ax.set_ylabel(f"{label} / {UNITS[label]}");

axes[-1].set_xlabel(f"time / {UNITS['time']}")

# +
yielder_error = make_yielder_df(df_raw_error, ["timestamp", "longitude", "latitude", "altitude"])
df_error = next(yielder_error)

# misc
df_error["timestamp"] = (
    df_error["timestamp"] - df_error.loc[0, "timestamp"]
).dt.total_seconds() / 60

# remove non-stationary
df_error.drop(df_error.index[:150], inplace=True)
df_error.reset_index(inplace=True)

# means
df_means = df_error.mean()
for label in ["longitude", "latitude", "altitude"]:
    df_error[f"mean-{label}"] = df_means[label]
    
# diff and Cartesian

dx, dy, dz = calc_cartesian_coord_diffs(
    df_error["latitude"].values, df_error["mean-latitude"].values,
    df_error["longitude"].values, df_error["mean-longitude"].values,
    df_error["altitude"].values, df_error["mean-altitude"].values
)

df_error["dx"] = dx
df_error["dy"] = dy
df_error["dz"] = dz

R = np.power(
    np.prod(df_error[["dx", "dy", "dz"]].apply(np.abs).quantile(0.9).values), 1 / 3
)

# +
pairs = [["dx", "dy"], ["dx", "dz"], ["dy", "dz"]]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (l1, l2) in zip(axes, pairs):
    ax.scatter(df_error[l1], df_error[l2], s=2, color="navy")
    ax.grid(True); ax.set_xlabel(f"{l1} / {UNITS[l1]}"); ax.set_ylabel(f"{l2} / {UNITS[l2]}")
# -

# Figure 3. The scatter plots of the Cartesian distance errors.

# The histogram characterise the presicion of the measurement. Is this the error we are looking for? Not quite. The term "error" may refer to two quantities
# 1. the distance between the true position $\mu$ and the measured position
# 2. a radius of a circle when centred of the measurement contains the true position with some probability
#
# We will never know the answer to the first one from a single measurement. The true location can be anything. As to the second one, it is possible to create an ellipsoid around $m$ in which the true position is with some high chance. In fact, there are infinitely many such shapes, but there is only one whose semi-axes are proportional to the uncertainties. Say, this probabilty can be set at 90%. Once again, the meaning of this error-ellipsoid is that if that the same point measurement is repeated hundred times it is expected that the true position will be inside of it on about 90 occasions.
#
# We are still unable to say where the real position is inside of the ellipsoid. It has an equal chance of being anywhere inside of it (if it is) using a single measurement (aside: Bayesian update on more points). This means laborious calculations to determine the average distance between a point in the shape and its centre. A more conservative approach will be used to save time. The maximum distance on the surface of the ellipsoid and its centre is equal to the largest semi-axis. The distance between two measurement points will have an associated maximum error 90 percent (or more) of the times twice the largest semi-axis. Alternatively, one can calculate the mean radius of the ellipsoid an consider its double as the error, for we still don't know where the true position is. 
#
# The maximum error estimate is about 8 metres. The measurement was taken in built-up area so it is indeed a likely upper bound of the distance error out in the fields. It is not far from the mean error of 5 metres cited by Garmin.

# ### Smoothing
#
# We set out to find possible ways to smooth the velocity and pace time series. By smoothing it is meant
# * mitigating the effects coming from the measurement uncertainty
# * reducing the granularity of the resultant time series.
#
# #### Moving window
#
# If the distance between to points is small relative to the error than the likelihood of obtaining a noisy velocity or pace value is larger. 
#
#
# To decrease the effect of these random fluctuations, the pace will be calculated only between points where the path connecting them has a length a certain number times of the error.
#
# $$
# \begin{eqnarray}
#     j & \in & [1,n-1], i \in [2, n]: j < i \\
#     S_{j, i} & = & \sum\limits_{k=j+1}^{i}s_{k} \quad \text{(path length)} \\
#     T_{j, i} & = & t_{i} - t_{j} \quad \text{(time to complete path)} \\
#     v_{i} & = & \frac{S_{j, i}}{T_{j, i}}
# \end{eqnarray}
# $$
#
# The above calculation can be performed in the staggered expanding procedure
# 1. fix a starting point
# 2. move forward until the path length is large enough
# 3. calculate velocity and pace
# 4. save velocity, pace and time (index)
# 5. set the current point as the starting point
# 6. repeat 2--5. until the end of the data
#
#
# The position of the start of the first window determines those of all other windows. As such it is rather arbitrary how the smoothed graph looks like. If the windowing is performed at each time step, a more jaggedy, but in a consistent way, final graph is obtained. The first procedure will be referred to as "staggered", the second one as "rolling".

# +
store = {"rolling":{}, "staggered": {}}
error = 5

# @NOTE, I know creating a np.ndarray at each iteration...

for key, is_staggered in zip(["rolling", "staggered"], [False, True]):

    for factor in [2, 5, 10]:
        d, t, v = calc_velocity_thresh(
            df_trace["distance"].values,
            df_trace["delta-d"].values,
            df_trace["timestamp"].values,
            error, factor, is_staggered
        )
    
        store[key].update({factor : (d,t, v)})

# +
colours = ["navy", "blue", "cornflowerblue"][::-1]
yielder_palette = make_yielder_palette(colours)


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, gridspec_kw={"wspace": 0.05})

gen_ax = (ax for ax in axes.flat)
for key, records in store.items():
    
    # top left -- quantity vs distance
    ax = next(gen_ax)
    ax.set_title(f"{key} smoothing")
    ax.plot(df_trace["distance"], df_trace["velocity"], c="grey", lw=1, label="raw")

    palette = next(yielder_palette)
    for factor, (d, t, v) in records.items():
        ax.plot(d, v, c=next(palette), label=f"{factor} " r"$\cdot$" " error")
    
    ax = next(gen_ax)
    ax.set_title(f"{key} smoothing")
    ax.plot(df_trace["timestamp"], df_trace["velocity"], c="grey", lw=1, label="raw")

    palette = next(yielder_palette)
    for factor, (d, t, v) in records.items():
        ax.plot(t, v, c=next(palette), label=f"{factor} " r"$\cdot$" " error")
        
for ax in axes.flat:
    ax.set_xlim(left=0); ax.set_ylim(0, 12); ax.grid(True)
    ax.legend(loc="lower right")
                
                
axes[1, 0].set_xlabel(f"distance / {UNITS['distance']}")
axes[1, 1].set_xlabel(f"time / {UNITS['delta-t']}")
axes[0, 0].set_ylabel(f"velocity / {UNITS['velocity']}")
axes[1, 0].set_ylabel(f"velocity / {UNITS['velocity']}")
# -

# Figure 4. The rolling window (top row), staggered (bottom row) smoothed velocities as a function of the total distance (left column) and time (right column). The raw velocity is in grey. The smoothed curves are plotted in colours light blue, medium blue and navy for the thresholds set at 2, 5, 10 times the distance error.

# +
colours = ["navy", "blue", "cornflowerblue"][::-1]
yielder_palette = make_yielder_palette(colours)


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, gridspec_kw={"wspace": 0.05})

gen_ax = (ax for ax in axes.flat)
for key, records in store.items():
    
    # top left -- quantity vs distance
    ax = next(gen_ax)
    ax.set_title(f"{key} smoothing")
    ax.plot(df_trace["distance"], df_trace["pace"], c="grey", lw=1, label="raw")

    palette = next(yielder_palette)
    for factor, (d, t, v) in records.items():
        ax.plot(d, 60 / v, c=next(palette), label=f"{factor} " r"$\cdot$" " error")
    
    ax = next(gen_ax)
    ax.set_title(f"{key} smoothing")
    ax.plot(df_trace["timestamp"], df_trace["pace"], c="grey", lw=1, label="raw")

    palette = next(yielder_palette)
    for factor, (d, t, v) in records.items():
        ax.plot(t, 60 / v, c=next(palette), label=f"{factor} " r"$\cdot$" " error")
        
for ax in axes.flat:
    ax.set_xlim(left=0); ax.set_ylim(0, 20); ax.grid(True)
    ax.legend(loc="lower right")
                
                
axes[1, 0].set_xlabel(f"distance / {UNITS['distance']}")
axes[1, 1].set_xlabel(f"time / {UNITS['delta-t']}")
axes[0, 0].set_ylabel(f"pace / {UNITS['pace']}")
axes[1, 0].set_ylabel(f"pace / {UNITS['pace']}")
# -

# Figure 5. The rolling window (top row), staggered (bottom row) smoothed paces as a function of the total distance (left column) and time (right column). The raw pace is in grey. The smoothed curves are plotted in colours light blue, medium blue and navy for the thresholds set at 2, 5, 10 times the distance error.

# The procedures indeed render the series smoother as proven by Figure 5. The spikes shrink and the angles flatten as the window expands. This is in part due the reduction of the effect of the uncertainty and the loss of temporal resolution. The latter effect is the most evidently observed when comparing the "rolling" and the "staggered" pairs of curves. The latter ones are smoother because the connecting line segments are longer.
#
# It is worth pointing out that the time series (right panels) contains wider troughs than their distance counterparts. The distance changes very little, if at all, during stationary periods, hence the series progress to a minor extent on the x-axis. 
#
# It also warrant is paragraph whether to look at these data as spatial or temporal series. It really does depend on the user of them and their intentions. More often than not, data scientist deal with series whose temporal coordinate is the driver of the story. On the other hand, walking and running are more of a spatial experiences. It is easier to recall approximate distances than times. Also, features of the terrain are also primarily change with distance rather than time. Unless someone runs on a geological timescale.
#
# ### Moving average
#
# The window reduces the resolution of the series i.e. the pace will be during the last 2--5 seconds. This might be just fine for the user who is interested in features on the scale of ten seconds when doing split training, or on the scale of (half-)minute when on a longer course. Should they be burning with desire to glean the most gleaming resolution, there is a source to quench their thirst. 
#
# A look back strategy can be invoked. Take all path lengths to a suitably distant point in the past and calculate the velocities/paces over each of them. A weighted average of these velocities are then formed where the weights represent the importance of path length (c.f. error) and how far their started in the past (relevance). The formulae are left to the reader to attain as their object of desire, should they be inclined so.
#
#

# ### Gaussian process regression
#
# Indeed, the moving average idea suggest that the velocity at a given time is more similar to its neighbouring values than to those far in time. Speaking of desire, the keen reader already felt just by looking at the title and perheps reading the first few sentences that Gaussian process regression (GPR) will be the apex of this post. GPR naturally takes into account the measurement errors and the similarity of close measurements.
#
# The fullest of treatment would require a GPR on the coordinates. We proceed somewhat quicker by applying the method to the distances with time as the independent coordinates. The dependent variable is the raw speed. The pace has a wide range with wild oscillations, whereas the velocity in steadily found in a smaller range. 
#
# Even though it has ben established that the distance is the driving coordinate from the user's (runner) perspective, it is ridden with errors. Colluding or nearly identical distances at subsequent time steps actually helps the procedure because they can used to estimate the error more accurately.
#
# #### Procedure
#
# The measurement times, the velocities have already been calculated. The error (uncertainty) of the latter is computed from the ratio of the distance error (5 metres) and the time difference used to calculate the velocity.

# +
df_trace["error-v"] = 0.005 / (df_trace["delta-t"] / 60)
error_v = df_trace["error-v"].values[1:] 

kernel = RBF()
regr = GaussianProcessRegressor(kernel=kernel, alpha=error_v / 15)

X = df_trace["timestamp"].values[1:].reshape(-1, 1)
y = df_trace["velocity"].values[1:]
regr.fit(X, y)

y_hat, y_std = regr.predict(X, return_std=True)

# filter out non-physical speeds
mask = y_hat < 0
y_hat[mask] = np.nan
y_std[mask] = np.nan

# +
fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

ax = axes[0]
ax.plot(df_trace["timestamp"].values, df_trace["velocity"].values, c="grey", alpha=0.5)
ax.plot(np.ravel(X), y_hat, color="navy", alpha=0.91)
ax.fill_between(np.ravel(X), y_hat - y_std, y_hat + y_std, color="navy", alpha=0.3)

ax.grid(True); ax.set_xlim(0, 180); ax.set_ylim(0, 12)
ax.set_ylabel(f"velocity / {UNITS['velocity']}")

ax = axes[1]
ax.plot(df_trace["timestamp"].values, df_trace["pace"].values, c="grey", alpha=0.45)
ax.plot(np.ravel(X), 1 / y_hat * 60, color="navy", alpha=0.91)
ax.fill_between(np.ravel(X), 60 / (y_hat + y_std), 60 / (y_hat - y_std), color="navy", alpha=0.3)

ax.grid(True); ax.set_xlim(0, 180); ax.set_ylim(0, 20)
ax.set_ylabel(f"pace / {UNITS['pace']}"); ax.set_xlabel(f"time / {UNITS['delta-t']}")
# -

# Figure 6. The Gaussian process regression smoothed (blue) and raw (grey) velocities (top panel) and paces (bottom panel) as a function of total distance (left column) time (right column).
#
# Of all methods, the GPR yields the smoothest curves. It also retains every original measurement points and incorporates the uncertainty. There is a little caveat, though. There are two total distances. The one used so far derived directly from the measurements, and one, $\hat{d}$ implied by the smoothed velocities $\hat{v}$. They are expected to be close to each other. The panels of Figure 7. indicate that the two distances are identical within numerical tolerance.

# +
# integrate velocity
d_implied = np.cumsum((df_trace["delta-t"][1:] * y))
d_raw = df_trace["distance"].values[1:]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[1]

y_diff = y_hat - y
y_diff_mean = np.nanmean(y_diff)

ax.hist(y_diff, bins=200, color="navy")
ax.axvline(y_diff_mean, ls="--", lw=2, color="#333333")
ax.grid(True); ax.set_xlabel(r"$(\hat{v} - v$) / " f"{UNITS['velocity']}"); ax.set_ylabel("count")

ax = axes[0]
ax.plot(df_trace["timestamp"].values[1:], d_implied / 60 - d_raw, c="navy")
ax.grid(True); ax.set_xlim(0, 20); ax.set_xlabel(f"time / {UNITS['delta-t']}")
ax.set_ylabel(r"$(\hat{d} - d$) / " f"{UNITS['distance']}")
# -

# Figure 7. The difference of the estimated and measured total distances (left panel). The histogrammed differences of the estimated and measured velocities (right panel).
