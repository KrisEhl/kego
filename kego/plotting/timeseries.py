import typing as t
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kego.plotting.axes_utils
import kego.plotting.utils_plotting


def set_axes_timeseries(
    ax: plt.axes,
    xlim: t.Optional[list] = None,
    ylim: t.Optional[list] = None,
    fontsize: int = 8,
    title: t.Optional[str] = None,
    label_x: t.Optional[str] = None,
    label_y: t.Optional[str] = None,
):
    """
    Customizes matplotlib axes for timeseries plotting

    Parameters:
    ----------
    ax: matplotlib axes
    xlim: Axes limits along the x-axis
    ylim: Axes limits along the y-axis
    fontsize: Size of the font
    title: Title of the axes
    label_x: Label of the x-axis
    label_y: Label of the y-axis

    Returns
    -------
    """
    if label_x is None:
        label_x = "time"
    kego.plotting.axes_utils.set_axes(
        ax,
        xlim=xlim,
        ylim=ylim,
        fontsize=fontsize,
        title=title,
        label_x=label_x,
        label_y=label_y,
        labelrotation_x=45,
    )


def plot_timeseries(
    time: t.Union[np.ndarray, str],
    y: t.Union[np.ndarray, str],
    ds: Optional[pd.DataFrame] = None,
    ax: t.Optional[plt.axes] = None,
    fig: t.Optional[plt.figure] = None,
    xlim: t.Optional[list] = None,
    ylim: t.Optional[list] = None,
    label_x: t.Optional[str] = None,
    label_y: t.Optional[str] = None,
    sort_by_time: bool = True,
    label: Optional[str] = None,
    **kwargs,
):
    """
    Plot timeseries with matplotlib

    Parameters:
    ----------
    time: Time values of timeseries
    y: y-values of timeseries
    ds: When provided and time/y are keys their values are drawn from `ds`
    ax: matplotlib axes
    fig: matplotlib figure
    xlim: Axes limits along the x-axis
    ylim: Axes limits along the y-axis
    label_x: Label of the x-axis
    label_y: Label of the y-axis
    sort_by_time: Sort time/y based on time before plotting
    label: Label of the plot
    kwargs: Key word arguments for plt.plot
    Returns
    -------
    """
    fig, ax = kego.plotting.utils_plotting.create_figure_axes(
        figure=fig, axes=ax, font_size=None
    )
    if ds is not None and isinstance(y, str):
        if label_y is None:
            label_y = y
        y = ds[y].values
    if ds is not None and isinstance(time, str):
        if label_x is None:
            label_x = time
        time = ds[time].values
    if sort_by_time:
        mask_time_sort = np.argsort(time)
        time = time[mask_time_sort]
        y = y[mask_time_sort]
    ax.plot(time, y, label=label, **kwargs)
    if xlim is None:
        min_x = np.nanmin(time)
        max_x = np.nanmax(time)
        xlim = [min_x, max_x]
    if ylim is None:
        min_y = np.nanmin(y)
        max_y = np.nanmax(y) * 1.1
        ylim = [min_y, max_y]
    set_axes_timeseries(ax, label_x=label_x, label_y=label_y, xlim=xlim, ylim=ylim)
    return fig
