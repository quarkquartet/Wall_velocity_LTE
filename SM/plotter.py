"""
Plotter.py
Written to define some functions so that the plot can be easier to customize.
Some fonts are also set.
Author: Tanner Trickle
Modified and packaged by: Isaac R. Wang
"""

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

rcParams["figure.facecolor"] = "w"
rc("text", usetex=True)
rc(
    "text.latex",
    preamble=r"\usepackage{amsmath}\usepackage{amssymb}\usepackage[version=4]{mhchem}",
)

# rcParams['text.latex.preamble'] = [
#     r'\usepackage{amsmath}',
#     r'\usepackage{amssymb}',
#     r'\usepackage[version=4]{mhchem}']

import numpy as np
import seaborn as sns

color_list_disc = sns.color_palette("Set1")
color_list_cont = sns.color_palette("viridis", as_cmap=True)


def set_ticks(
    ax,
    minor_bottom=True,
    minor_top=True,
    minor_left=True,
    minor_right=True,
    major_bottom=True,
    major_top=True,
    major_left=True,
    major_right=True,
):
    """
    Personal preference for the tick options in most plots.
    """

    ax.minorticks_on()

    ax.tick_params(
        which="major",
        direction="in",
        length=6,
        width=1,
        bottom=major_bottom,
        top=major_top,
        left=major_left,
        right=major_right,
        pad=5,
    )

    ax.tick_params(
        which="minor",
        direction="in",
        length=3,
        width=1,
        bottom=minor_bottom,
        top=minor_top,
        left=minor_left,
        right=minor_right,
    )


def set_axes(
    axes,
    xy_char,
    ax_min,
    ax_max,
    ax_type="lin",
    label="",
    font_size=30,
    show_first=True,
    step=1,
):
    """

    Wrapper function for nice axes.

    xy_char = 'x'/'y'
    ax_type = 'lin'/'log'

    ax_min - min axes value
    ax_max - max axes value

    """

    if xy_char == "x":

        axes.set_xlabel(label, fontsize=font_size)

        if ax_type == "log":

            set_log_xticks(
                axes,
                ax_min,
                ax_max,
                font_size=font_size,
                show_first=show_first,
                step=step,
            )

        elif ax_type == "lin":

            set_lin_xticks(
                axes,
                ax_min,
                ax_max,
                font_size=font_size,
                show_first=show_first,
                step=step,
            )

    if xy_char == "y":

        axes.set_ylabel(label, fontsize=font_size)

        if ax_type == "log":

            set_log_yticks(
                axes,
                ax_min,
                ax_max,
                font_size=font_size,
                show_first=show_first,
                step=step,
            )

        elif ax_type == "lin":

            set_lin_yticks(
                axes,
                ax_min,
                ax_max,
                font_size=font_size,
                show_first=show_first,
                step=step,
            )


def set_lin_xticks(ax, minval, maxval, font_size=30, show_first=True, step=1):

    ax.tick_params(axis="x", labelsize=font_size)

    if show_first:

        ax.xaxis.set_ticks(
            np.linspace(minval, maxval, round((maxval - minval) / step) + 1)
        )

    else:

        ax.xaxis.set_ticks(
            np.linspace(minval + step, maxval, round((maxval - minval) / step))
        )

    ax.set_xlim(minval, maxval)


def set_lin_yticks(ax, minval, maxval, font_size=30, show_first=True, step=1):

    ax.tick_params(axis="y", labelsize=font_size)

    if show_first:

        ax.yaxis.set_ticks(
            np.linspace(minval, maxval, round((maxval - minval) / step) + 1)
        )

    else:

        ax.yaxis.set_ticks(
            np.linspace(minval + step, maxval, round((maxval - minval) / step))
        )

    ax.set_ylim(minval, maxval)


def set_log_xticks(
    ax, minval, maxval, font_size=30, show_first=True, step=1, n_minor=8
):

    major_tick_labels = []

    if not show_first:
        label_start = minval + 1
    else:
        label_start = minval

    for i in range(minval, maxval + 1, 1):

        if i % step == 0 and i >= label_start:

            # special cases
            if i == 0:
                major_tick_labels.append(r"$1$")

            elif i == 1:
                major_tick_labels.append(r"$10$")

            else:
                major_tick_labels.append(r"$10^{" + str(i) + "}$")
        else:

            major_tick_labels.append("")

    major_ticks = np.arange(minval, maxval + 1, 1)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_tick_labels, fontsize=font_size)

    minor_ticks = []

    for i in range(minval, maxval, 1):
        for j in range(n_minor):

            minor_ticks.append(np.log10((j + 2) * 10**i))

    ax.set_xticks(minor_ticks, minor=True)

    ax.set_xlim(minval, maxval)


def set_log_yticks(
    ax, minval, maxval, font_size=30, show_first=True, step=1, n_minor=8
):

    major_tick_labels = []

    if not show_first:
        label_start = minval + 1
    else:
        label_start = minval

    for i in range(minval, maxval + 1, 1):

        if i % step == 0 and i >= label_start:

            # special cases
            if i == 0:
                major_tick_labels.append(r"$1$")

            elif i == 1:
                major_tick_labels.append(r"$10$")

            else:
                major_tick_labels.append(r"$10^{" + str(i) + "}$")
        else:

            major_tick_labels.append("")

    major_ticks = np.arange(minval, maxval + 1, 1)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels(major_tick_labels, fontsize=font_size)

    minor_ticks = []

    for i in range(minval, maxval, 1):
        for j in range(n_minor):

            minor_ticks.append(np.log10((j + 2) * 10**i))

    ax.set_yticks(minor_ticks, minor=True)

    ax.set_ylim(minval, maxval)
