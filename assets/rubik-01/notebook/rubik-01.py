# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python (stp)
#     language: python
#     name: stp
# ---

# +
# TO HIDE -- SETUP

import os.path
import sys

import numpy as np

from imageio import (
    imread,
    imsave
)
from skimage.color import (
    rgb2lab,
    lab2rgb
)

import matplotlib.pyplot as plt

# +
# TO HIDE -- SETUP
plt.rcParams.update({"savefig.dpi": 150})

folder_image = "/home/bhornung/repos/rubik/images"
folder_src = "/home/bhornung/repos/bhornung11.github.io/assets/rubik-01/script"

sys.path.append(folder_src)

# +
# TO HIDE -- SETUP

from rubik import (
    add_grid,
    calc_dist_lab,
    calc_dist_rgb,
    dither,
    rubikify
)
# -

# ## Introduction
#
# This post seeks to replicate images as if they had been assembled from the faces of a Rubik's cube.
#
# ## Notes
#
# As per usual, a set of trimmed notes are presented here. The scripts are deposited in [this file](https://github.com/bhornung11/bhornung11.github.io/blob/master/assets/rubik-01/script/rubik-01.py). The raw notebook can be found [here](https://github.com/bhornung11/bhornung11.github.io/blob/master/assets/rubik-01/notebook/rubik-01.py)
#
#
# ## Problem statement
#
# Let $\mathcal{C}\neq \emptyset$ is a set of colours. Then an image, $P$, of size $(m, n)$ is simply a function that assigns a colour to a pair of indices:
#
# $$
#     \begin{eqnarray}
#         # ! n, m & \in & \mathbb{N}: n > 0, m > 0
#         \quad & \text{(height and width of the image)} \\
# %
#     P & \in & \mathcal{C}^{n \times m}
#     \quad & \text{(image)} \\
# %
#     \Leftrightarrow p_{i,j} & \in & \mathcal{C}
#     \quad & \text{(colour of the pixel)}
#     \end{eqnarray}
# $$
#
# Let $\emptyset \neq \mathcal{R} \subset \mathcal{C}$ a palette. In or case, it will be the colours of the Rubik's cube.
#
# Then we seek a function, $T$ that replaces the original colours of the image with those chosen from the palette:
#
# $$
# \begin{eqnarray}
#    T & \in & \mathcal{T} =  \mathcal{C}^{n \times m} \times \mathcal{P}^{n \times m} \, .
# \end{eqnarray}
# $$
#
# How this tranform function is realised depends on the aesthetic requirements. 
#
# ## Minimum distance replacements
#
# ### Single pixel
#
# There is a distance function,$d$ , defined over the colours:
#
# $$
#     d \in \mathcal{D} = \mathbb{R}_{0}^{\mathcal{C} \times \mathcal{C}} \, .
# $$
#
# The distance between two colours is zero if and only if they are identical. In general, smaller distances express a larger degree of similarity of colours. The transform then simply chooses that colour from the palette which minimises the distance:
#
# $$
#     \begin{eqnarray}
#        c^{*} = T(c) = \underset{r \in \mathcal{R}}{\operatorname{arg min}} \left\{ d(c, r) \right\} \, .
#     \end{eqnarray}
# $$
#
# We utilised the fact that a single colour constitutes a matrix of a single row and column.

# ### Multiple pixels
#
# Alternatively, multiple image pixels can be replaced by a single colour. This leads to the reduction of the resolution. The quantity that is being minimised is the sum of individual pixel distances to the palette colours.
#
# $$
#     \begin{eqnarray}
#         # ! n, m & \in & \mathbb{N}: n > 0, m > 0
#         \quad & \text{(height and width of the image)} \\
# %
#     P & \in & \mathcal{C}^{n \times m}
#     \quad & \text{(image)} \\
# %
#     # ! I & \subset & [1, n] \times [1, m]
#     \quad & \text{(selection of grid indices)} \\
# %
#     P_{I} & = & \left\{ c_{i}: i \in I \right\}
#     \quad & \text{(image detail)} \\
# %
#     c^{*} & = & T(P_{I}) = \underset{r \in \mathcal{R}}{\operatorname{arg min}} \left\{ 
#         \sum\limits_{i \in I} d(c_{i}, r)
#     \right\}
#     \quad & \text{(colour closest to all pixels)} 
# %
#     \end{eqnarray}
# $$
#
# ### 

# +
# TO HIDE -- SETUP

SIZE_FIG_H2_V2 = (9, 6)

COLOURS_HEX = {
   "red":    "#BA0C2F",
   "green":  "#009A44",
   "blue":   "#003DA5",
   "orange": "#FE5000",
   "yellow": "#FFD700",
   "white":  "#FFFFFF"
}

COLOURS_RGB = {
   "red":   np.array([186, 12, 47]),
   "green": np.array([0, 154, 68]),
   "blue":  np.array([0, 61, 165]),
   "orange": np.array([254, 80, 0]),
   "yellow": np.array([255, 215, 0]),
   "white":  np.array([255, 255, 255])  
}


COLOURS_VEC_LAB = np.array(
    [rgb2lab(v.reshape((1, 1, 3)) / 255).reshape(3)
    for v in COLOURS_RGB.values()]
)


COLOURS_VEC_RGB = np.array(list(COLOURS_RGB.values()))

WEIGHTS = np.array(
    [[0, 0, 0, 8, 4], 
     [2, 4, 8, 4, 2],
     [1, 2, 4, 2, 1]]
) / 42
# -

# ## Examples
#
# ### Transform in the RGB colour space
#
# The two images are shown below stacked with their rubikified version. The colour of each original pixel was replaced by the closest one from the Rubik palette; one that mininised the $L_{2}$ distance in the RGB space:
#
# $$
#     d^{RGB}(c, r) = \left[ (c_{R} - r_{R})^{2} + (c_{G} - r_{G})^{2} + (c_{B} - r_{B})^{2}\right]^{\frac{1}{2}} \, .
# $$

# +
# TO HIDE -- CODE

image_mtn_rgb = imread(os.path.join(folder_image, "rub-mtn-01.jpg"))
image_zeb_rgb = imread(os.path.join(folder_image, "rub-zeb-01.jpg"))

image_mtn_lab = rgb2lab(image_mtn_rgb / 255)
image_zeb_lab = rgb2lab(image_zeb_rgb / 255)

# +
# TO HIDE -- CODE

image_mtn_rub_rgb = rubikify(image_mtn_rgb, COLOURS_VEC_RGB, calc_dist_rgb, 1 , 1)
image_zeb_rub_rgb = rubikify(image_zeb_rgb, COLOURS_VEC_RGB, calc_dist_rgb, 1 , 1)

image_mtn_rub_lab = lab2rgb(rubikify(image_mtn_lab, COLOURS_VEC_LAB, calc_dist_lab, 1 , 1))
image_zeb_rub_lab = lab2rgb(rubikify(image_zeb_lab + 0, COLOURS_VEC_LAB, calc_dist_lab, 1 , 1))

# +
# TO HIDE -- PLOT

fig, axes = plt.subplots(2, 2, figsize=SIZE_FIG_H2_V2, gridspec_kw={"hspace": 0.1, "wspace": 0.01})
axes[0, 0].imshow(image_mtn_rgb)
axes[1, 0].imshow(image_mtn_rub_rgb)

axes[0, 1].imshow(image_zeb_rgb)
axes[1, 1].imshow(image_zeb_rub_rgb)

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.savefig(os.path.join(folder_image, "rubik-01-fig-01.png"),bbox_inches='tight')
# -

# The first thing to notice is the green stripes of the zebras. Since black is not a colour of the cube, the algorithm replaces it with green which is the most similar to it in the RGB space with the $L_{2}$ distance. 
#
# ### Transform in the CIELab colour space with CIE94 distance
#
# We somehow feel that dark blue is more akin to black than green is. Zebras sporting a Celtic Glasgow outfit is the most plausably not due to their particular dispostion in football, but to the fact that the RGB $L_{2}$ colour distance does not rely on human colour perception. This was recognised early on and colour spaces and distances were developed which were sought to approximate colour proximities as we recognise them. A particularly successfull pair of them is the CIELab space coupled with the CIE94 distance. (The reader is kindly referred to the corresponding [Wikipedia article](https://en.wikipedia.org/wiki/Color_difference#CIE94) for the details). We, from now on, thus recourse to this space and distance.
#
# The transformed images are juxtaposed with their originals in Figure 2.

# +
# TO HIDE -- PLOT

fig, axes = plt.subplots(2, 2, figsize=SIZE_FIG_H2_V2, gridspec_kw={"hspace": 0.1, "wspace": 0.01})
axes[0, 0].imshow(image_mtn_rgb)
axes[1, 0].imshow(image_mtn_rub_lab)

axes[0, 1].imshow(image_zeb_rgb)
axes[1, 1].imshow(image_zeb_rub_lab)

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.savefig(os.path.join(folder_image, "rubik-01-fig-02.png"),bbox_inches='tight')
# -

# Indeed the stripes are blue as we previously expected. The grey slopes of the mountains are replaced by areas dominated by white as opposed to green.
#
# ### Dithering
#
# The transitions between colours are somewhat too sharp. For instance, the sky is collated in a suddenly separated sequence of white, orange and yellow bands. The transition is smooth between the shades in the original image. The banding can be softened by adding random noise to the pixels of the starting image before mapping to the restricted palette. By doing so, adjacent pixels that had the same colours will now painted differently, but similarly. This process is called dithering and has a history of sixty years.
#
# For the sake of a brief explanation let us assume that the colours range between nought and one. The palette has two colours: 0 (black) and 1 (white). Consider four pixels arranged in a square. Say, they all have the same colour of 0.65. Without dithering we would end up with four white points after the transform without dithering.
#
# With dithering, 
# * the top left will be repainted with "1". 
# * half of the error (-0.175) is then added to the top right pixel which will be 0.475 thus repainted with black.
# * the quarter of the error (-0.0875) is added to the bottom two pixels turning them to 0.5625. Their final colour is therefore white.
# * In total we have one black and three white points. Their average colour is 0.75 which is much closer to the initial 0.65 than the all over white through the simple assignment.
#
# By choosing which pixels receive what amount of the error it is possible introduce a randomness which, on average, approximates the original colours.
#
# Figure 3. extends the previous figure with the dithered images.

# +
# TO HIDE -- BORING

image_mtn_rub_lab_dither = lab2rgb(
    dither(
        image_mtn_lab + 0.0, COLOURS_VEC_LAB, calc_dist_lab, 1 , 1, WEIGHTS
    )
)
image_zeb_rub_lab_dither = lab2rgb(
    dither(
        image_zeb_lab + 0.0, COLOURS_VEC_LAB, calc_dist_lab, 1 , 1, WEIGHTS
    )
)

# +
# TO HIDE -- PLOT

fig, axes = plt.subplots(2, 2, figsize=SIZE_FIG_H2_V2, gridspec_kw={"hspace": 0.1, "wspace": 0.01})
axes[0, 0].imshow(image_mtn_rgb)
axes[1, 0].imshow(image_mtn_rub_lab_dither)

axes[0, 1].imshow(image_zeb_rgb)
axes[1, 1].imshow(image_zeb_rub_lab_dither)


for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.savefig(os.path.join(folder_image, "rubik-01-fig-03.png"),bbox_inches='tight')
# -

# It is remarkable that a palette of just seven colours reproduces the shades to an unexpectedly satisfying degree. This is a result of a handful of facts. Firstly, five of the six cardinal colours of the CIELab space are on the cube: yellow, blue, red, green and white. Apart from black and so tinted colours, all colours are close to one of this selection. Secondly, the pixels are so tiny that the adjecent squares coalesce (cf. pointilism). Had the tranformed image been magnified tenfold, the apparent colours would not have blended by the eyes (cf. Mondrian). The left and the middle panels of Figure illustrate the breakdown of coalescence in a zoomed-in section of the zebra image.

# TO HIDE -- CODE
gridded = add_grid(image_zeb_rub_lab_dither[50:100, 50:100], 38, 34)

# +
# TO HIDE -- PLOTTING

fig, axes = plt.subplots(1, 3, figsize=(6, 6))
axes[0].imshow(image_zeb_rgb[50:100, 50:100])
axes[1].imshow(image_zeb_rub_lab_dither[50:100, 50:100])
axes[2].imshow(gridded)

for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    
plt.savefig(os.path.join(folder_image, "rubik-01-fig-04.png"),bbox_inches='tight')
# -

# ### Adding the borders
#
# Each square of a Rubik's cube has a black border. The width of a square is 19 millimetres, the sticker stretches across the central 17 by 17 millimetres area. This means that each transformed pixel must have an approximately 5% wide black perimeter to attain that unique Rubik's feel. The pixels thus cannot anymore be represented by undivided sqaures. They need to be magnified (19 by 19) times to make space for the black background. The left panel of Figure 4. shows a detail of a properly rubikified image. The entire images can be accessed [1](https://github.com/bhornung11/bhornung11.github.io/blob/master/assets/rubik-01/images/img-rub-zeb-lgt-dither.png), [2](https://github.com/bhornung11/bhornung11.github.io/blob/master/assets/rubik-01/images/img-rub-zeb-dither.png).. Be warned, its area is 360 times that of the original one. (Note, obviously, interlacing (9 by 9) squares with black lines of width of one would result in an imperceptibly similar image. Apart from the edges, where the background would be drawn with twice its correct thickness.
#
# ## Colour interpolation
#
# The reader is implored to imagine a properly rubikified picture being hung on the wall of their favourite hipster bar -- most likely adorned by equally aesthetic vinyl covers on its both sides. When stading close to the artwork, the black lines separate from the oversized pixels of the transformed picture. As one distances themselves, as they should, from the picture and still faces it, the unit of perception becomes a tied but non-blended mixture of black border a palette colour. Futher still, the border starts to blend in the square that it embraces. At greater distances, or having consumed ~~sufficient~~, substantial amount of drinks a single colour is seen over a square. One that is tinted with black or gray.

# It demands but little sobriety to recognise that our multipixel formula is able to find the matching palette colour in all three scenarios. The squares of discrete and partially blended colours are modelled with a grid of pixels. The square over which a fully blended colour spreads is equivalent to a single pixel. In that, it is a special case of the formula below:
#
# $$
#     c^{*} = T(P_{I}) = \underset{r \in \mathcal{R}}{\operatorname{arg min}} \left\{ 
#         \sum\limits_{i \in I} d(c_{i}, r) \right\}\, .
# $$
#
# We will only consider the likely event when the colour of the border bleeds in the squares making them darker, but they are still discernible
#  
# The difficulty hides in properly mixing the colours. Let us consider two diffferent one, $\mathbf{c}_{1}$ and the aptly named $\mathbf{c}_{2}$. The mingled colour $\mathbf{c}_{3}$ is four times closer to the first colour than to the second one. The mixing weight is then the ratio of the individual distances and their sum
#
# $$
#     \begin{eqnarray}
#         ! \mathbf{c}_{1}, \mathbf{c}_{2} & \in & \mathcal{C}
#             \quad & \\
# %
#         \mathbf{c}_{1} & \neq & \mathbf{c}_{2}
#             \quad & \\
# %
#     \mathbf{c}_{3} & = & \frac{
#     d(\mathbf{c}_{1}, \mathbf{c}_{3})}{
#         d(\mathbf{c}_{1}, \mathbf{c}_{3}) + d(\mathbf{c}_{2}, \mathbf{c}_{3})
#     } \mathbf{c}_{1}
#     + 
# %
# \frac{
#     d(\mathbf{c}_{2}, \mathbf{c}_{3})}{
#         d(\mathbf{c}_{1}, \mathbf{c}_{3}) + d(\mathbf{c}_{2}, \mathbf{c}_{2})
#     } \mathbf{c}_{3}
#         \quad & \\
# %
#     \mathbf{c}_{3} & = & \lambda \mathbf{c}_{1} + (1 - \lambda) \mathbf{c}_{2} 
#     \quad & \\   
# %
#     \end{eqnarray}
# $$
#
# Regretfully enough, the linear interpolation above is only correct in Euclidean spaces such as RGB. In those spaces the mixing weights can be equated to the ratio of distances as we just did. CIELab with CIE94 is not Euclidean. Moreover there is no analytic formula for the distance-based mixing weights. They are thus ought to be determined numerically. But we won't. No matter how exciting or enchanting to spend the evening chatting to unknown equations. If we think about it again, it does not take too long to realise that the insertion of black colour would primarily affect the lightness of the large pixels. What is really needed is to only increase the lightness of the original images by the amount which balances the expected darkening, before the transformation. Since the first coordinate of the CIELab space is exactly the lightness, the required procedure is of the simplest kind.
#
# Figure 5. arranges once more the original picture its lightened version and the Rubikified product.

# +
# TO HIDE -- BORING

image_mtn_lab_lgt = image_mtn_lab + 0
a = image_mtn_lab_lgt[:, :, 0] * 1.05
a = np.where(a < 100, a, 100)
image_mtn_lab_lgt[:, :, 0] = a

image_zeb_lab_lgt = image_zeb_lab + 0
a = image_zeb_lab_lgt[:, :, 0] * 1.05
a = np.where(a < 100, a, 100)
image_zeb_lab_lgt[:, :, 0] = a


image_mtn_rub_lab_lgt_dither = lab2rgb(
    dither(
        image_mtn_lab_lgt + 0.0, COLOURS_VEC_LAB, calc_dist_lab, 1 , 1, WEIGHTS
    )
)
image_zeb_rub_lab_lgt_dither = lab2rgb(
    dither(
        image_zeb_lab_lgt + 0.0, COLOURS_VEC_LAB, calc_dist_lab, 1 , 1, WEIGHTS
    )
)

# +
# TO HIDE -- PLOTTING

fig, axes = plt.subplots(2, 2, figsize=SIZE_FIG_H2_V2, gridspec_kw={"hspace": 0.1, "wspace": 0.01})
axes[0, 0].imshow(image_mtn_rgb)
axes[1, 0].imshow(image_mtn_rub_lab_lgt_dither)

axes[0, 1].imshow(image_zeb_rgb)
axes[1, 1].imshow(image_zeb_rub_lab_lgt_dither)


for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.savefig(os.path.join(folder_image, "rubik-01-fig-05.png"), bbox_inches='tight')
# -

# Five percent extra ligthness was applies to the images. This washed out most of the clouds by The Pinnacles, and the sunset over the savannah has become paler too. The full sized versions are saved too [1](https://github.com/bhornung11/bhornung11.github.io/blob/master/assets/rubik-01/images/img-rub-mtn-lgt-dither.png), [2](https://github.com/bhornung11/bhornung11.github.io/blob/master/assets/rubik-01/images/img-rub-zeb-lgt-dither.png).
#
# Finally, it must be remarked a number of colour theoretical and mathematical nuances have been glossed over the previous paragraph. The reader must not be timid, though, to use the utilities linked to this notebook to create their own artwork and strap it on the once plastered wall of a fitting establishment preferably in A0 format.

# +
# TO HIDE -- CHORE

imgs = {
    "img-rub-mtn-dither.png": image_mtn_rub_lab_dither,
    "img-rub-zeb-dither.png": image_zeb_rub_lab_dither,
    "img-rub-mtn-lgt-dither.png": image_mtn_rub_lab_lgt_dither,
    "img-rub-zeb-lgt-dither.png": image_zeb_rub_lab_lgt_dither
}

for fname, img in imgs.items():
    print(fname)
    
    bordered = add_grid(img, 19, 17)
    imsave(os.path.join(folder_image, fname), bordered)
