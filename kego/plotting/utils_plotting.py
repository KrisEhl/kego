import typing as t
from typing import Optional

import matplotlib.colors
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt

import kego.checks
import kego.constants
import kego.lists


def set_font(font_size=10):
    SMALL_SIZE = font_size
    MEDIUM_SIZE = font_size
    BIGGER_SIZE = font_size

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def pair_in_list(pair, _list):
    return tuple(pair) in list(_list) or list(pair) in list(_list)


def annotate_values(
    H: np.ndarray,
    axes: plt.axes,
    size_x: int,
    size_y: int,
    color: str = "black",
    round_to_base: int | None = None,
    font_size: float | None = None,
):
    """
    Overplot values on plot based on 2d matrix whose values are
    plotted in equidistant intervals

    Parameters:
    ----------
    H: Matrix (2d) whose values will be overplot
    ax: Matplotlib axes
    size_x: Number of elements along x-axis
    size_y: Number of elements along y-axis

    Returns
    -------
    matplotlib figure and axes
    """
    x_start = 0
    x_end = 1
    y_start = 0
    y_end = 1
    jump_x = (x_end - x_start) / (2.0 * size_x)
    jump_y = (y_end - y_start) / (2.0 * size_y)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size_x, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size_y, endpoint=False)
    H_processed = H.copy()
    H_processed = np.flip(H_processed, axis=0)
    if round_to_base is not None:
        H_processed = np.round(H, round_to_base)
        if round_to_base < 0:
            H_processed = np.array(H_processed, int)
    for x_index, x in enumerate(x_positions):
        for y_index, y in enumerate(y_positions):
            label = H_processed[y_index, x_index]
            text_x = x + jump_x
            text_y = y + jump_y
            text_y = 1 - text_y
            axes.text(
                text_x,
                text_y,
                label,
                color=color,
                ha="center",
                va="center",
                transform=axes.transAxes,
                fontsize=font_size,
            )


def _get_values_from_bar_object(
    bar_object: matplotlib.container.BarContainer,
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Debug function to obtain plotted values from a bar plot object

    Returns X, Y and H values of original plot
    Parameters:
    ----------
    bar_object: Bar plot object

    Returns
    -------
    X, Y, H
    """
    X = []
    Y = []
    H = []
    for b in bar_object:
        x, y = b.get_xy()
        X.append(x)
        Y.append(y)
        H.append(b.get_height())
    return np.array(X), np.array(Y), np.array(H)


def _get_values_from_pcolormesh_object(
    pc_object: matplotlib.collections.QuadMesh,
) -> np.ndarray:
    """
    Debug function to obtain plotted values from pcolormesh object

    Returns flattened H values
    Parameters:
    ----------
    pc_object: Pcolormesh plot object

    Returns
    -------
    Flattened matrix values
    """
    return pc_object.get_array().data


def to_list(x, n=2):
    if not (isinstance(x, list) or isinstance(x, tuple)):
        return [x] * n
    if len(x) != n:
        raise ValueError(f"{x} doesn't have expected length {n=}")
    return x


def get_norm(
    norm: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm_symlog_linear_threshold: float | None = None,
) -> matplotlib.colors.LogNorm | matplotlib.colors.Normalize:
    """
    Returns matplotlib norm used for normalizing matplotlib's colorbars

    Parameters:
    ----------
    norm:
        Normalization type (None/"linear", "log", "symlog")
    vmin:
        Minimum value of normalized colors
    vmax:
        Maximum value of normalized colors
    norm_symlog_linear_threshold:
        Threshold value below which symlog of norm becomes linear.

    Returns
    -------
    Norm object from matplotlib.colors
    """
    if norm == "log":
        return matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm == "symlog":
        if norm_symlog_linear_threshold is None:
            raise ValueError(
                f"{norm_symlog_linear_threshold=} needs to specified to use {norm=}"
            )
        return matplotlib.colors.SymLogNorm(
            vmin=vmin, vmax=vmax, linthresh=norm_symlog_linear_threshold
        )
    elif norm == "linear" or norm is None:
        return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        raise ValueError(f"Norm: {norm} unknown!")
