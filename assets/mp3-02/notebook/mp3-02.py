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
import os.path
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple
)

import numba as nb
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.realpath("dir_script"))

# +
from mp3_script.transforms.transforms import (
    transform_chain,
    make_gen_transform
)

from mp3_script.transforms.legenre import(
    reconstruct_legendre_w
)

from mp3_script.dtw.distances import (
    calc_dist_l2_sq
)

from mp3_script.dtw.dtw import (
    dtw_discrete_dp,
    dtw_discrete_dp_with_coupling,
    calc_dmat_dtw
)

from mp3_script.stat.describe import (
    describe_dmat
)

from mp3_script.stat.bootstrap import (
    bootstrap_dmat
)

from mp3_script.plotting.plotting import (
    make_connecting_lines
)
# -

SIZE_FIG_H1_V2 = 12, 8
SIZE_FIG_H2_V1 = 12, 5
SIZE_FIG_H1_V3 = 12, 12

# ## Intoduction
#
# The similarity of popular songs is quantified based on their perceived loudness in this blog post. The first part of the twin entries transformed `mp3` waveforms to smooth time series of loudness. These were brought to a scale whose minimum is the normal auditory limit and its maximum is the pain threshold -- averaged over all musical tastes. The loudness progressions are represented by a Legendre series expansion.
#
# We first define what is meant by likeness of two songs. A suitable mathematical expression is then sought to quantify this notion. With the help of it, we will finally savour various flavours of group similarity.
#
# ## Note
#
# The raw notebook is located here. The scripts which were utlised therein a stored in this folder.

# ## Data
#
# We will concern ourselves with the oeuvre of Adele. The songs released by the artist on her studio albums are the material of this investigation. 
#
# ### Data  preparation
#
# The songs were recast as smoothed perceived loudness time series in the previous post. Most of the transformations are reused here with two differences.
#
# #### Smoothing
#
# It migth be of interest for those technically inclided that the expansion coefficients were determined by approximating the overlap integral between the Legendre polynomials the subsampled series with the trapezoidal rule over the Chebysev nodes. This way, a more accurate, higher order smoothing is obtained by doing away with the spurious oscillations due to Runge's phenomenon.
#
# #### Trimming
#
# The first and last three seconds of each song were removed from in order to avoid spurious increase of the overlap due to the bracketing silent or barely audible sections.
#
# #### Loudness again
#
# The introduced loudness scale is an absolute one. Whilst every values on it are directly comparable i.e. 25 is twice as quiet as 50, the relative differences now depend on the values from which they are derived. For example, $10 = 80 - 70$ and $10 = 40 - 30$ does not anymore reflect the same loudness ratio.
#
#

# +
# TO HIDE -- SETUP

dir_adele = "/home/bhornung/Downloads/adele"
dir_random = "/home/bhornung/"

# +
# TO HIDE -- DATA MANIPULATION

# get subsampled series -- Adele
store_loudness_dotty_a = list(
    transform_chain(
        dir_adele,
        kwargs_trim={"left": 3, "right": 3},
        return_stage="trim"
    )
)

# get expansion coefficients -- Adele
list_coeff = list(
    transform_chain(
        dir_adele,
        kwargs_coeff={"order": 100},
        return_stage="coeff"
    )
)

# reconstruct series -- Adele
N_POINT = 1001
gen_rec = make_gen_transform(
    list_coeff, reconstruct_legendre_w, {"n_sample": N_POINT}
)

store_loudness_a = list(gen_rec)
# -

# ## Series
#
# A select set of the properties of the loudness time series are summarised here to aid drive the along the main direction of the present discussion and explore some cul-de-sacs. Let all series be defined over the same domain and codomain:
#
# $$
# \begin{eqnarray}
#     # !A, B & \in & \mathbb{R}: - \infty < A < B < \infty
#     \quad & \text{(finite domain)} \\
# %
#     \mathcal{D} & = & [A, B]
#     \quad & \text{(domain)} \\
# %
#     # ! U, V & \in & \mathbb{R}: 0 \leq U < V < \infty
#     \quad & \text{(non-negative finite function values)} \\
# %
#     \mathcal{R} & = & [A, B]
#     \quad & \text{(codomain)} \\
# %
#     \mathcal{F} & = & \left\{ f : [A, B] \rightarrow [U, V] \right\}
#     \quad & \text{(functions of interest)}
# \end{eqnarray}
# $$
#
# The domain is the closed--closed interval betwen -1 and 1 in our case, $[A = -1, B = 1]$. The codomain spans between zero and hundred, $\mathcal{R} = [U = 0, V = 100]$.
#
# Both of them are expanded with our choice of Legendre basis.
# $$
#     \begin{eqnarray}
#         f(x) & = & \sum\limits_{k=0}^{K} a_{k}P_{k}(x) 
#         \\
#         g(x) & = & \sum\limits_{k=0}^{L} b_{k}P_{k}(x)
#     \end{eqnarray}
# $$
#
# The orthogonality of the legendre polynomials comes handy too:
#
# $$
#     \begin{eqnarray}
#     \int\limits_{x=-1}^{1}P_{k}(x)P_{l}(x) \mathrm{d}x  & = &  \frac{1}{2k + 1} \delta_{kl}
#     \end{eqnarray}
# $$
#
# The expansion coefficients define a vector in $\mathbb{R}^{K}$ and $\mathbb{R}^{L}$. It is advantageous to pad that of lower the dimension with zeros to bring the in the same space:
#
# $$
# \begin{eqnarray}
#     # ! K, L & \in & \mathbb{N} : 0 < K \leq L < \infty
#     \quad & \text{(number of the expansion coefficients)} \\
#     %
#     \mathbf{a}, \mathbf{b} & \in & \mathbb{R}^{L + 1}
#     \quad & \text{(vectors)} \\
#     %
#     (\mathbf{a})_{i} & = &  \begin{cases}
#            a_{i - 1} & \text{ if }  i \leq K \\
#            0  & \text{ if }  f K < i \leq L
#     \end{cases}
#     \quad & \text{(pad dimension absent)} \\
#     %
#     (\mathbf{b})_{i} & = & b_{i - 1}
#     \quad & \text{(higher order expansion)}
# \end{eqnarray}
# $$

# The subsampled and the Legendre smoothed series are shown in Figure 1. The expansion order was set to one hundred. This gives a resolution of 3.6 seconds between adjacent peaks at a mean song length of three minutes.

# +
# TO HIDE -- PLOTTING

fig, axes = plt.subplots(
    10, 5, figsize=(12, 12),
    sharex=True, sharey=True, gridspec_kw={"hspace": 0.1, "wspace": 0.1}
)

for i, (ax, entry) in enumerate(zip(axes.flat, store_loudness_a)):

    # trimmed
    y_dotty = store_loudness_dotty_a[i].song.time_series
    x_dotty = np.linspace(-1, 1, len(y_dotty))
    ax.scatter(x_dotty, y_dotty, s=1, alpha=0.1)

    # reconstructed
    y_recon = entry.song.time_series
    x_recon = np.linspace(-1, 1, len(y_recon))
    ax.plot(x_recon, y_recon, c="navy")
    ax.grid(True); ax.set_xlim(-1, 1); ax.set_ylim(0, 50)
    
axes[-1, 0].set_xlabel("time / a.u.")
for ax in axes[-2, 1:]:
    ax.set_xlabel("time / a.u.")
    
for ax in axes[-1, 1:]:
    ax.axis("off")
    
for ax in axes[:, 0]:
    ax.set_ylabel("loudness / a.u.")
# -

# Figure 1. The subsampled loudness time series (pale blue dots) and their Legendre smoothed counterparts (bark blue lines).

# ## Similarity
#
# Two series are similar if they have close values which follow a like pattern. In other words, the progression of the values during a period in one series can be observed in the other one around the same time and position.
#
# ### Quantifying similarity

# #### Mean squared error
#
# The normalised mean squared error is the average pointwise absolute difference. It is zero when the two series are identical. Its drawback that is sensitive to shift: If two loudness progressions are similar, but changes in one lag behind those of the other, this measure quickly goes up in value. This will indicate dissimilarity contrary our perception and definition. See the top panel of Figure 1. as an example.
#
# ##### Cul-de-sac with a view 1.
#
# The mean square error is related to the Euclidean norm of the difference vector of the coefficient vectors. One can prove it by either through the orthogonality of the basis functions, or, equivalently, recalling Parseval's identity.
#
# $$
#     \begin{eqnarray}
#    D(f, g) & = & \frac{1}{B - A}\int\limits_{x=A}^{B} (f(x) - g(x))^{2} \mathrm{d}x 
#    \quad & \text{(integral of the squared error)} \\
#    %
#    & = & 
#            \frac{1}{B - A} \left[ \sum\limits_{k=0}^{N}a_{k}^{2} + \sum\limits_{k=0}^{N}b_{k}^{2}
#            - 2  \sum\limits_{k=0}^{N}a_{k}b_{k}
#            \right]
#        \quad & \\
#    %
#    & = & 
#       \frac{1}{B - A} || \mathbf{a} - \mathbf{b}||
#        \quad &
#    \end{eqnarray}
# $$
#
# Because of the domain and the codomain of all functions, $[A, B], [U, V]$, are identical, the maximum possible mean sqaured error is $(V - U)^{2}$. Normalising by which the most dissimilar series will have a distance of unit.

# #### Correlation
#
# The various correlation coefficients only consider trends but not the location by design. 
#
# $$
# \begin{eqnarray}
#     \varphi(f, g) & = & \frac{
#         \int\limits_{x=A}^{B}(f(x) - \bar{f})(g(x) - \bar{g}) \mathrm{d}x
#     }{
#         \left[ \int\limits_{x=A}^{B}(f(x) - \bar{f})^{2} \mathrm{d}x \right]^{\frac{1}{2}}
#         \cdot
#         \left[ \int\limits_{x=A}^{B}(g(x) - \bar{g})^{2} \mathrm{d}x \right]^{\frac{1}{2}} 
#     }
#     \quad & \text{(Pearson correlation coefficient)} \\
#     %
#         \{f, g \} & \ni & h: \bar{h} = \int\limits_{x=A}^{B}f(x)\mathrm{d}x
#     \quad & \text{(means)}
# \end{eqnarray}
# $$
#
# The middle panel of Figure 2. juxtaposes two series which have Pearson coefficient of unit. They are however not similar, one belonging to a hypothetical song which is quiet throughout whilst the other one is loud.
#
# ##### Cul-de-sac with a view 2.
#
# A certain advantage of the Pearson correlation coefficient that it can be calculated in $\mathcal{O}(1)$ time from the expansion coefficients:
#
# $$
#     \begin{eqnarray}
#     \varphi(f, g) & = & \frac{
#         \sum\limits_{k=1}^{L}a_{k}b_{k}
#     }{
#         \left[
#             \sum\limits_{k=1}^{L}a_{k}^{2}   
#          \right]^{\frac{1}{2}}
#          \cdot
#          \left[
#             \sum\limits_{k=1}^{L}b_{k}^{2}
#         \right]^{\frac{1}{2}}
#     } \, .
#     \end{eqnarray}
# $$
#
# Let us define the demeaned coefficient vectors:
# $$
#     \begin{eqnarray}
#         ! \mathbf{a_{1}}, \mathbf{b_{1}} & \in & \mathbb{R}^{L}
#         \quad &  \\
#         %
#         (\mathbf{a_{1}})_{i}  & = & (\mathbf{a})_{i + 1}
#         \quad & \text{(drop mean)} \\
#         %
#         (\mathbf{b_{1}})_{i} & = & (\mathbf{b})_{i + 1}
#         \quad & \text{(drop mean)} \\
#     \end{eqnarray}
# $$
#
# Then the Pearson correlation is the angle between these two vectors:
#
# $$
#     \begin{eqnarray}
#     \varphi(f, g) & = & \frac{\mathbf{a_{1}} \cdot  \mathbf{b_{1}}}{||\mathbf{a_{1}}|| \cdot ||\mathbf{b_{1}}||}
#     = \phi(\mathbf{a_{1}} ,\mathbf{b_{1}})
#     \end{eqnarray} \, .
# $$

# #### Dynamic time warping distance
#
# The dynamic time warping distance is able to link multiple points from one curve to multiple points on the other one. Through this way, features spanning more than a single point can be matched and compared. 
#
# The continuous DTW allows, similarly, for pairing of subdomains of different lengths between the two series. This is achieved by stretching or contracting subsections of the two domains ($x$ values) with the functions $\alpha(x)$ and $\beta(x)$.
#
# $$
#     \begin{eqnarray}
#     (1) & : & ! \alpha,  \beta:  \mathcal{D} \rightarrow \mathcal{D} 
#     \quad & \text{(stretching, contracting)} \\
#     %
#     (2) & : & \forall x, y \in \mathcal{D}, x < y : \alpha(x) < \alpha(y) \land \beta(x) < \beta(y)
#     \quad & \text{(matched point pairs do not cross)} \\
#     %
#     (3) & : & \alpha(0) = \beta(0) = A \land \alpha(B) = \beta(B) = B
#     \quad & \text{(start and endpoints must be matched)} \\
#     %
#     \alpha, \beta \in \mathcal{F} & \iff & \alpha, \beta: (1) \land (2) \land (3)  
#     \quad & \text{(set of mapping functions)}
#     \end{eqnarray}
# $$
#
# Once the $x$-values are aligned, the images ($y$-values) are calculated from which the pairwise distances are determined and then integrated. The integration is driven by $x$, but the arc length is taken into account through the derivative of the mapping functions.
#
# $$
#     \begin{eqnarray}
#     %
#     D(f, g) & = & \int\limits_{x=A}^{B}
#     ||f(\alpha(x)) - g(\beta(x))||_{q}^{q} \cdot \left|\left| \left( 
#         \frac{\mathrm{d}\alpha(x)}{\mathrm{d} x},
#         \frac{\mathrm{d}\beta(x)}{\mathrm{d} x}
#     \right) \right|\right|^{p}_{p} \mathrm{d} x
#     \end{eqnarray}
# $$
#
# There are infinitely many $\alpha, \beta$ mappings. Some of them decrease others of them increase the distance defined by above integral. DTW is the minimum distance over all permitted $\alpha, \beta$. They are those which "best align" the temporal features of the two series.
#
# $$  
#     DTW(f, g) = \underset{(\alpha, \beta) \in \mathcal{F}^{2} }{\min} 
#      \int\limits_{x=A}^{B}
#     ||f(\alpha(x)) - g(\beta(x))||_{q}^{q} \cdot  \left|\left| \left( 
#         \frac{\mathrm{d}\alpha(x)}{\mathrm{d} x},
#         \frac{\mathrm{d}\beta(x)}{\mathrm{d} x}
#     \right) \right|\right|^{p}_{p} \mathrm{d} x
# $$
#
# Regretfully enough, there is no general closed form of the DTW for continuous functions. Calculating DTW is costly. If the two curves are composed of $n$ segments each
# 1. $\mathcal{O}(n^{2})$ : discrete DTW over segment endpoints
# 1. $\mathcal{O}(n^{2} \log(n^{2}))$ : discrete DTW over the segments
# 1. $\mathcal{O}(n^{5})$ : continuous DTW
#     
# The present time series are treated as sequences of unconnected points. The integration in the equation above is replaced by a summation. In order for the distances be comparable between sequences composed of varying number of points, the DTW is divided by the lager length:
#
# $$
#     \begin{eqnarray}
#     # ! f & = & \bigcup\limits_{i=1}^{n}\{(x_{1,i}, y_{1,i}) \} \\
# #     %
#     # ! g & = & \bigcup\limits_{i=1}^{m}\{(x_{2,i}, y_{2,i}) \} \\
# #     %
#     d(f,g) & = & \frac{1}{\max\{n, m\}} DTW(f, g) \, .
#     %
#     \end{eqnarray}
# $$

# +
# TO HIDE -- PLOT SETUP

x1 = np.linspace(-1, 1, 51)

# 1 overlap example
y11 = np.hstack([np.linspace(10, 12, 25), np.linspace(15, 10, 26)]) * 3
y12 = np.roll(y11, 5) - 2
y12[:5] = y11[:5]

idcs = np.stack([np.arange(len(x1)), np.arange(len(x1))], axis=-1)
xx1, yy1 = make_connecting_lines(x1, x1, y11, y12, idcs)

# 2 correlation example
y21 = y11
y22 = y21 / 3
xx2, yy2 = make_connecting_lines(x1, x1, y21, y22, idcs)

# 3 DTW example
y31 = y21
y32 = y12 * 0.7

_, idcs = dtw_discrete_dp_with_coupling(x1, x1, y31, y32, calc_dist_l2_sq)
xx3, yy3 = make_connecting_lines(x1, x1, y31, y32, idcs[:-1] - 1)

# +
# TO HIDE -- PLOTTING

fig, axes = plt.subplots(3, 1, figsize=SIZE_FIG_H2_V1, sharex=True, sharey=True)

ax = axes[0]
ax.scatter(x1, y11, color="navy", s=5)
ax.scatter(x1, y12, color="cornflowerblue", s=5)
ax.plot(xx1, yy1, c="#222222", lw=0.5)

ax = axes[1]
ax.scatter(x1, y21, color="navy", s=5)
ax.scatter(x1, y22, color="cornflowerblue", s=5)
ax.plot(xx2, yy2, c="#222222", lw=0.5)

ax = axes[2]
ax.scatter(x1, y31, color="navy", s=5)
ax.scatter(x1, y32, color="cornflowerblue", s=5)
ax.plot(xx3, yy3, c="#222222", lw=0.5)
for ax in axes:
    ax.set_xlim(-1, 1); ax.set_ylim(0, 50); ax.grid(True)
# -

# Figure 2. The ground distances between two series contributing to the overall similarity. Overlap (top panel), correlation (middle panel), DTW (bottom panel). The points between which the ground distances are calculated are represented by dark gray lines.

# ## Results
#
# ### Statistical descriptors
#
# We are in the most fortunate position of having a set whose all elements are known. The population is all Adele songs and the sample is all Adele songs. All statistical descriptors, such as mean, median etc are therefore exact.

# +
# TO HIDE -- CALC

dmat_a = calc_dmat_dtw(store_loudness_a, calc_dist_l2_sq)
store_a = describe_dmat(dmat_a)
# -

# #### Pairwise song distances
#
# Let us describe the pairwise distances, $d_{i,j}$. They range from about 0.5 unit to 16 units. The maximum possible distance is hundred. It would be observed between two tracks where one was on the auditory the other was of the pain threshold throughout. We know from the previous post that the smoothed loudness does not exceed 40 units with its median being around 22 units. However, the DTW must be unitless in this case as it is a combination of temporal and auditory separations. The distances are shown in left panel of Figure 3. by tracks.

# +
# TO HIDE -- PLOTTING

fig, axes = plt.subplots(1, 2, figsize=SIZE_FIG_H2_V1)

dmat = store_a["dmat"]
y = np.ones(len(dmat))

ax = axes[0]
for i, x in enumerate(dmat):
    y += 1
    
    ax.scatter(x, y, c="navy", alpha = 0.25, s=5)
    ax.scatter(store_a["means"][i], y[:1], c="r", marker="|", s=50)
    ax.scatter(store_a["medians"][i], y[:1], c="k", marker="|", s=50)
    
ax.set_xlabel(r"$d_{ij}$" " / a.u."); ax.set_ylabel("song #"); ax.set_xlim(0, 12); ax.grid(True)

ax = axes[1]

_ = ax.hist(store_a["dmat"].flat, histtype="step", color="navy", bins=200, lw=2)
ax1 = ax.twinx()
_ = ax1.hist(
    store_a["dmat"].flat, histtype="step", color="purple", bins=200,
    density=True, cumulative=True, lw=2
)

xm, xs = store_a["mean"], store_a["std"]
for x in (xm - xs, xm, xm + xs):
    ax1.axvline(x, color="r", ls=":", lw=1.5);

for x in (store_a["p10_p90"][0], store_a["median"], store_a["p10_p90"][1]):
    ax1.axvline(x, color="k", ls="--", lw=1.5);
    
ax.set_xlabel(r"$D_{ij}$" " / a.u."); ax.set_ylabel("count / a.u.")
ax1.set_ylabel(r"$P(d_{ij} < D_{ij})$" " / a.u."); ax1.grid(True)
_ = ax.set_xlim(0, 12)
# -

# Figure 3. The song pair distances with track-wise means (red bars) and medians (black bars), (left panel). The histogram of the song pair distances, (blue line) and their cumulative distribution function (purple line), (right panel).

# The mean pairwise distance, $\bar{d}_{ij}$ is 3.3. It means that any two songs is expected to be separated by about 3% loudness from the first to the last second. Strictly speaking, this is only true if the points are coupled diagonally. Given that the timepoints are separated by 1/ 1000 units, the bulk of the distance is due to auditory differences.
#
# The queries that can be satisfied with these numbers are
# * which two songs are the most similar
#     * `I miss you` -- `Sweetest devotion`
# * which two songs are the least similar
#     * `Chasing pavements` -- `Rumour has it`
#
# #### Average song distance
#
# The question: how similar is a song to all the other is answered by the mean (or median) song distance:
#
# $$
#     \bar{d}_{i} = \frac{1}{N - 1}\sum\limits_{j=1}^{N}d_{ij}(1 - \delta_{ij})
# $$
#
# This quantifier enables us to execute simple queries such as
# 1. finding the song which is the most similar to every other
#     * `Someone like you`
# 1. identifiying the song which is the least akin to the rest.
#     * `Chasing pavements`

# +
# TO HIDE -- PLOTTING

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=SIZE_FIG_H1_V3)

ijs = [
    store_a["ij_min"],
    store_a["ij_max"],
    (store_a["i_min"], store_a["i_max"])
]

titles = [
    "most similar pair",
    "least similar pair",
    "most and least similar songs"
]
for k, (ax, (i,j)) in enumerate(zip(axes.flat, ijs)):
    y1 = store_loudness_a[i].song.time_series
    y2 = store_loudness_a[j].song.time_series
    x1 = np.linspace(-1, 1, len(y1))
    x2 = np.linspace(-1, 1, len(y2))
    

    
    ax.set_title(titles[k])
    ax.plot(
        x1, y1, c="cornflowerblue",
        label=store_loudness_a[i].name_song
    )
    
    ax.plot(
        x2, y2, c="navy",
        label=store_loudness_a[j].name_song
    )
    
    if k == 0:
        _, idcs = dtw_discrete_dp_with_coupling(x1, x2, y1, y2, calc_dist_l2_sq)
        xx, yy = make_connecting_lines(x1, x2, y1, y2, idcs)
        ax.plot(xx, yy, c="#444444", lw=0.1)

    ax.grid(True)
    ax.set_xlim(-1, 1); ax.set_ylim(0, 50); ax.set_ylabel("loudness / a.u.")
    ax.legend(loc="upper left")
    
_ = axes[-1].set_xlabel("time / a.u.")
# -

# Figure 4. The most similar pair of songs (top panel), the least similar pair of songs (middle panel), the most and least songs on average (bottom panel). The warping path is represented by drk gray lines in the top panel.

# ## Comparison to reference data
#
# It is intriguiging to know whether the songs above are distinguishable from the body of pop songs. More accurately, with what degree of certainty their similarity can be ascertained?
#
# ### Data source
#
# Two hundred songs that charted in the UK were downloaded. Fifty from each year -- 2008, 2011, 2015 and 2021 -- when a studio album by Adele were released. Care was taken not to include any of her output in this sample. These lots were handled together and subjected to the exact same treatment as those by the singer scrutinised.
#
# ### Distances
#
# The DTW-s were calculated for all pairs. The raw distances and their histogram look different to those of the Adele songs. The reference pairwise distances are, on average, larger. A grand mean of 4.2 as opposed to 3.3 is observed. The mean song distances are also shifted to greater values with respect to those of the tracks of Adele.

# +
# TO HIDE -- DATA MANIPULATION

# process reference songs
dir_random = "/home/bhornung/repos/mp3/albums-rnd"
# get expansion coefficients -- Adele
list_coeff_r = list(
    transform_chain(
        dir_random,
        kwargs_coeff={"order": 100},
        return_stage="coeff"
    )
)

# reconstruct series -- reference
N_POINT = 1001
gen_rec = make_gen_transform(
    list_coeff_r, reconstruct_legendre_w, {"n_sample": N_POINT}
)

store_loudness_r = list(gen_rec)

# +
# TO HIDE -- CALC

# calculate and describe DTW matrix
dmat_r = calc_dmat_dtw(store_loudness_r, calc_dist_l2_sq)
store_r = describe_dmat(dmat_r)

# +
# TO HIDE -- PLOTTING

fig, axes = plt.subplots(1, 2, figsize=SIZE_FIG_H2_V1)

dmat = store_r["dmat"]
y = np.ones(len(dmat))

ax = axes[0]
for i, x in enumerate(dmat):
    y += 1
    
    ax.scatter(x, y, c="navy", alpha = 0.1, s=5, edgecolor="none")
    ax.scatter(store_r["means"][i], y[:1], c="r", marker="|", s=50)
    ax.scatter(store_r["medians"][i], y[:1], c="k", marker="|", s=50)
    
ax.set_xlabel("d / a.u."); ax.set_ylabel("song #"); ax.set_xlim(0, 20); ax.grid(True)
ax.set_ylim(0, 200)
ax = axes[1]

_ = ax.hist(store_r["dmat"].flat, histtype="step", color="navy", bins=200, lw=2)
ax1 = ax.twinx()
_ = ax1.hist(
    store_r["dmat"].flat, histtype="step", color="purple", bins=200,
    density=True, cumulative=True, lw=2
)

xm, xs = store_r["mean"], store_r["std"]
for x in (xm - xs, xm, xm + xs):
    ax1.axvline(x, color="r", ls=":", lw=1.5);

for x in (store_r["p10_p90"][0], store_r["median"], store_r["p10_p90"][1]):
    ax1.axvline(x, color="k", ls="--", lw=1.5);
    
ax.set_xlabel("d / a.u."); ax.set_ylabel("count / a.u.")
ax1.set_ylabel("P(d < d)"); ax1.grid(True)
ax.set_xlim(0, 20)
# -

# Figure 5. The song pair distances with track-wise means (red bars) and medians (black bars), (left panel). The histogram of the song pair distances, (blue line) and their cumulative distribution function (purple line), (right panel).

# ### Statistical testing
#
# It is almost always nearly impossible to prove whether two samples are from the same distribution as a binary answer. In order to do it, one either need to confirm that the two samples are generated by the exact same process. Alternatively, it is demanded to deduce that the two distributions are identical i.e. all moments -- if defined -- are equal, or they have the same characteristic function.
#
# What is possible, on the other hand, to derive the probability of observing a quantity given
# * the two distributions are identical
# * the two distributions are not identical
#
# #### Setup
#
# Let $\mathcal{S}$ be the set of all songs that charted in the years when the artist released an album.. $\mathcal{A}$ denotes the collection of peices by Adele. It is called the test set. Then $\mathcal{R} = \mathcal{S} \setminus \mathcal{A}$ is the reference set.
#
# $$
#     \begin{eqnarray}
#         d_{ij} & = & d(s_{i}, s_{j}): s_{i}, s_{j} \in \mathcal{S}
#         \quad & \text{(shorthand for pairwise distances)} \\
#         %
#         \tilde{D}_{A} & = & \text{Median}(\{ d(s_{i}, s_{j}) : i \neq j, s_{i}, s_{j} \in \mathcal{A} \})
#         \quad & \text{(median of test set DTW-s)} \\
#         %
#         \tilde{D}_{R} & = & \text{Median}(\{ d(s_{i}, s_{j}) : i \neq j, s_{i}, s_{j} \in \mathcal{R} \})
#         \quad & \text{(median of reference set DTW-s)} \\
#         %
#     \end{eqnarray}
# $$
#
# The question is what is the probability of observing the test median if it comes from the reference distribution of the medians:
#
# $$
# \begin{eqnarray}
#     \tilde{d}_{a} & \sim &  P_{A} 
#     \quad & \text{(distribution of test median)} \\
#     %
#    P_{A}(\tilde{d}_{a}) & = & \begin{cases} 
#        1 \iff \tilde{d}_{a} = \tilde{D}_{a} \\
#        0 \iff \tilde{d}_{a} \neq \tilde{D}_{a} \\
#    \end{cases}
#    \quad & \text{(test median is exactly known)} \\
#   %
#   \tilde{d}_{r} & \sim & P_{R}
#   \quad & \text{(distribution of reference median)} \\
#   %
#   P (  
#       \tilde{d}_{a} & \sim &  P_{R}
#       \land \tilde{d}_{a} = \tilde{D}_{a}
#       |  \tilde{d}_{r} \sim  P_{R}
#    )
#    \quad & \text{(test median is from the reference distribution)}
# \end{eqnarray}
# $$
#
# We are in the priviledged position of knowing the test median exactly. Sadly enough, we only have a rather flimsy estimate of the reference one. There were thousands of songs that charted in the years investigated. The entirety of them is the population. Only two hundred of them are included in our sample, $\mathcal{R}^{'}$. Thus only a point estimate of the median is known.
#
# Our worry is twofold:
# 1. how representative is the sample of the population? (i.e. is it biased?)
# 1. if it is representative, how closely can the population median be estimated?
#
# #### Sample selection
# Let us address the first of our concerns. The reference set is the population of all songs that reached the top hundred position in the UK in the year when a studio album was released by Adele. A sample from reference set was chosen randomly (recurrent and new Christmas singles were excluded). It is believed that the author had no bias at selecting them. The reference songs can still, nevertheless, share characteristics that set them aside from those which did not reach the top hundred. This is due to something called popular state (Zeitgeist in older literature) and is not disputed here no matter the temptation.

# #### Bootstrapping
#
# Even if the sample supposedly shares the required properties with the population, its small size renders any single point estimate doubtful in its validity. What if these two hundred songs constitute a subset whose median matches the test median by chance. Or, on the contrary, it is off by a long chalk?
#
# Let us assume that the sample is representative of the population. Let us then first just randomly choose 47 songs of the 200, and calculate the median. Then estimate the median from a new selection of 47. It is likely to be different to the first one. If we repeat this procedure thousands of more times we will obtain an array of sample medians. The estimate of the reference median is the mean of those of the samples. The percentiles can also be read off from the array of values.
#
# $$
#     \begin{eqnarray}
#      ! \mathcal{I} & = & [1, 2, ..., |\mathcal{S}|
#      \quad & \text{(original sample indices)} \\
#     %
#     # ! N & \in & \mathbb{N}, 0 < N < \infty
#     \quad & \text{(number of individuals in a bootstrap sample)} \\
#     %
#     ! \mathcal{N} & = & [1, ..., N]
#     \quad & \text{(bootstrap sample index set)} \\
#     %
#     # ! B & \in & \mathbb{N}, 0 < B < \infty
#     \quad & \text{(number of bootstrap samples)} \\
#     %
#     # !n & \in & [1, ..., B]
#     \quad & \text{($n$-th bootstrap sample)} \\
#     %
#     (i_{1}, i_{2}, ..., i_{N})_{n} & = & \mathcal{B}_{n}  \in  \mathcal{I}^{\mathcal{N}}
#     \quad & \text{(individuals by index in a single bootstrap sample)} \\
#     %
#     \hat{\tilde{d}}_{n} & = & \text{Median}\left\{ 
#         \forall i_{k}, i_{l} \in \mathcal{B}_{n}, i_{k} \neq i_{l} : d_{i_{k}i_{l}}
#     \right\}
#     \quad & \text{(median of a single bootstrap sample)} \\
#     %
#     \hat{\tilde{d}} & = & \frac{1}{B}\sum_{n=1}^{B}\hat{\tilde{d}}_{n}
#     \quad & \text{(final median estimate)}
#     \end{eqnarray}
# $$
#
#
# **Note:** It is not required to choose 47 songs. We can -- and would be better to -- select a larger number of them. The selection is by replacement, that a single song can appear in the sample multiple times.

# #### Verdict
#
# The bootstrapped reference set median, $\hat{\tilde{d}}_{R}=3.2$, and the test median, $\tilde{d}_{A} = 2.8$, are not equal. They are show in Figure 6. along with a succession of bootstrapped median distributions. Since $tilde{d}_{A} < \hat{\tilde{d}}_{R}$ we are interested in the probability of observing a test median less than or equal to the actual one _given_ that it is distribution is that of the reference medians. This value can readily be obtained from the final, sixth, Figure, and it is 7%.

# +
# TO HIDE -- CALC

medians_bootstrapped = bootstrap_dmat(dmat_r, 47, 5000, np.nanmedian)

# +
# TO HIDE  -- PLOTTING

fig, ax = plt.subplots(1,1)
colours = [
     "lightblue", "cornflowerblue", "blue", "darkblue", "navy", "purple"
]
for i, n in enumerate([10, 50, 100, 500, 1000, 5000]):
    _ = ax.hist(
        medians_bootstrapped[:n], bins=500,
        cumulative=True, density=True,
        histtype="step",
        color=colours[i], label=n
    )

ax.axvline(store_a["median"], c="k", ls="--")
ax.axvline(np.mean(medians_bootstrapped), c="purple", ls="-.")
ax.legend(loc="upper left")


ax.set_ylim(0, 1); ax.set_xlim(2.5, 4.5); ax.grid(True); 
# -

# Figure 6. A sequence of bootstrap reference median distributions (pale blue tp purple solid lines). The reference median (purple dot-dashed line), the test median (black dashed line).
#
# As a summary, it can be concluded that the Adele songs released on albums are more similar to each other that the songs charted by different artists.
