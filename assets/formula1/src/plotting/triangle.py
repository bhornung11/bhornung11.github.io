"""
Snippets to create the big upper--lower triangle plot.
"""

import numpy as np
import matplotlib.pyplot as plt

cm = plt.get_cmap("BuPu")

def create_piechart_colours(steps, cmap):

    cm = plt.get_cmap(cmap)
    colours = np.ones((steps + 1, 4), dtype=np.float64)
    colours[:-1] = cm(np.arange(steps * 1) / steps / 1)[:steps]

    return colours


def _preppie_slices(ratio, stepsize, max_ratio):
    
    # get slices
    # 1) coloured
    n_slices = int(ratio / stepsize)
    sizes = [stepsize] * n_slices
    max_slice = n_slices * stepsize
    if  max_slice < ratio:
        sizes.append(ratio - max_slice)

    # white
    sizes.append(max_ratio - ratio)
    return sizes

def plot_fancy_pie(ax, ratio, stepsize, max_ratio, palette):

    sizes = _preppie_slices(ratio, stepsize, max_ratio)

    n_slices = len(sizes)
    if n_slices > len(palette):
        raise ValueError("Not enough colours")

    colours = palette[:n_slices - 1].tolist()
    colours.append(palette[-1])

    wedges, _ = ax.pie(sizes, colors=colours, wedgeprops = {'linewidth': 0})
    for w in wedges:
        w.set_linewidth(20)

clss = create_piechart_colours(50, "BuPu_r")

fig, axes = plt.subplots(
    32, 32,
    figsize=(32, 32),
    facecolor="white",
    gridspec_kw={"hspace": 0, "wspace": 0}
)

for i in  range(32):
    for j in range(i+1, 32):
        ax = axes[i, j]
        res = results[(i, j)]
        ax.plot(*res.start.T, c="b", lw=1.5)
        ax.plot(*res.rotated.T, c="purple", lw=1.5)

        ax.set_xlim((-1.35, 1.35)); ax.set_ylim((-1.35, 1.35))
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[j, i]
        ax.set_xticklabels([]); ax.set_yticklabels([])
        max_ratio = 0.50
        ratio = max_ratio - res.score

        plot_fancy_pie(ax, ratio, 0.01, 0.5, clss)
        ax.set_visible(True)

for i in range(32):
    ax = axes[i, i]
    ax.plot(*intpds[i].T, c="b", lw=1.5)
    ax.set_xlim((-1.2, 1.2)); ax.set_ylim((-1.35, 1.35))
    ax.set_xticklabels([]); ax.set_yticklabels([])

ttls = ("\n".join(x["name"].split()) for x in data.values())
for ax in axes[0]:
    ax.set_title(next(ttls), rotation=90)

ttls = ("\n".join(x["name"].split()) for x in data.values())
for ax in axes[:, 0]:
    ax.set_ylabel(next(ttls), rotation=0)