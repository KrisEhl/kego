import typing as t
from collections.abc import Sequence
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

import kego.constants


def set_axes(
    ax: plt.axes,
    xlim: t.Optional[list] = None,
    ylim: t.Optional[list] = None,
    fontsize: int = 8,
    title: t.Optional[str] = None,
    label_x: t.Optional[str] = None,
    label_y: t.Optional[str] = None,
    labelrotation_x: t.Optional[float] = None,
    labelrotation_y: t.Optional[float] = None,
):
    """
    Customizes matplotlib axes object

    Parameters:
    ----------
    ax: matplotlib axes
    xlim: Axes limits along the x-axis
    ylim: Axes limits along the y-axis
    fontsize: Size of the font
    title: Title of the axes
    label_x: Label of the x-axis
    label_y: Label of the y-axis
    labelrotation_x: Angle of rotation of the label of the x-axis
    labelrotation_y: Angle of rotation of the label of the y-axis

    Returns
    -------
    """

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if label_x is not None:
        ax.set_xlabel(label_x, fontsize=fontsize)
    if label_y is not None:
        ax.set_ylabel(label_y, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.tick_params(axis="both", which="minor", labelsize=fontsize)
    if labelrotation_x is not None:
        ax.tick_params(axis="x", labelrotation=labelrotation_x)
    if labelrotation_y is not None:
        ax.tick_params(axis="y", labelrotation=labelrotation_y)


def set_colorbar(ax_colorbar: plt.axes, label_y: str, fontsize: float):
    """
    Customizes matplotlib colorbar axes

    Parameters:
    ----------
    ax_colorbar: matplotlib colorbar axes
    label_y: Label of the y-axis
    fontsize: Size of the font

    Returns
    -------
    """
    ax_colorbar.set_ylabel(label_y, fontsize=fontsize)
    ax_colorbar.tick_params(axis="both", which="major", labelsize=fontsize)
    ax_colorbar.tick_params(axis="both", which="minor", labelsize=fontsize)


def set_x_lim(axes: kego.constants.TYPE_MATPLOTLIB_AXES, xlim: tuple | None):
    if xlim is not None:
        axes.set_xlim(*xlim)


def set_y_lim(axes: kego.constants.TYPE_MATPLOTLIB_AXES, ylim: tuple | None):
    if ylim is not None:
        axes.set_ylim(*ylim)


def set_x_log(
    axes: kego.constants.TYPE_MATPLOTLIB_AXES,
    log: str = "false",
    axis_symlog_linear_threshold: float | None = None,
) -> kego.constants.TYPE_MATPLOTLIB_AXES:
    """
    Sets scale of x-axis

    Parameters:
    ----------
    axes:
        Matplotlib axes
    log:
        Type of bins. Log types included "false", "symlog", "log"
    axis_symlog_linear_threshold:
        Threshold below which bins are linear to include zero values (when `log`="symlog")

    Returns
    -------
    axes
    """
    if log == "symlog":
        if axis_symlog_linear_threshold is None:
            raise ValueError(
                f"If log=='symlog', setting "
                f"{axis_symlog_linear_threshold=} required!"
            )
        axes.set_xscale("symlog", linthresh=axis_symlog_linear_threshold)
    elif log == "log":
        axes.set_xscale("log")
    return axes


def set_y_log(
    axes: kego.constants.TYPE_MATPLOTLIB_AXES,
    log: str = "false",
    axis_symlog_linear_threshold: float | None = None,
) -> kego.constants.TYPE_MATPLOTLIB_AXES:
    """
    Sets scale of y-axis

    Parameters:
    ----------
    axes:
        Matplotlib axes
    log:
        Type of bins. Log types included "false", "symlog", "log"
    axis_symlog_linear_threshold:
        Threshold below which bins are linear to include zero values (when `log`="symlog")

    Returns
    -------
    axes
    """
    if log == "symlog":
        if axis_symlog_linear_threshold is None:
            raise ValueError(
                "If log=='symlog', "
                f"setting: {axis_symlog_linear_threshold=} required!"
            )
        axes.set_yscale("symlog", linthresh=axis_symlog_linear_threshold)
    elif log == "log":
        axes.set_yscale("log")
    return axes


def set_axes_label(
    axes: kego.constants.TYPE_MATPLOTLIB_AXES,
    label: str | None,
    axis: str = "x",
    font_size: float = kego.constants.DEFAULT_FONTSIZE_SMALL,
):
    if label is None:
        return
    if axis == "x":
        axes.set_xlabel(label, fontdict={"fontsize": font_size})
    if axis == "y":
        axes.set_ylabel(label, fontdict={"fontsize": font_size})


def set_axis_tick_labels(
    axes: kego.constants.TYPE_MATPLOTLIB_AXES,
    values: Sequence[float] | np.ndarray | None = None,
    labels: Sequence | np.ndarray | None = None,
    date_formatter: str | None = None,
    axis: str = "x",
    rotation: int = 0,
    font_size: float = kego.constants.DEFAULT_FONTSIZE_SMALL,
    max_tick_labels: int | None = None,
) -> kego.constants.TYPE_MATPLOTLIB_AXES:
    """
    Set new tick labels for given values

    Parameters:
    ----------
    axes:
        Matplotlib axes
    values:
        Values for corresponding new labels (should also set labels)
    labels:
        New labels for corresponding values (should also set values)
    date_formatter:
        Format string to datetime values as tick labels, e.g. "%Y-%m-%d"
        See https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior for format codes.
    axis:
        Axis to set, i.e. "x" or "y"
    rotation:
        Angle of rotation of tick labels
    font_size:
        Fontsize of axis tick labels
    max_tick_labels:
        Maximum number of tick labels

    Returns
    -------
    axes
    """
    if axis == "x":
        if values is not None:
            axes.set_xticks(values)
        if labels is not None:
            axes.set_xticklabels(labels)
        if date_formatter is not None:
            axes.xaxis.set_major_formatter(
                matplotlib.dates.DateFormatter(date_formatter)
            )
        if max_tick_labels is not None:
            axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_tick_labels))
        axes.tick_params(axis="x", labelrotation=rotation, labelsize=font_size)
    elif axis == "y":
        if values is not None:
            axes.set_yticks(values)
        if labels is not None:
            axes.set_yticklabels(labels)
        if date_formatter is not None:
            axes.yaxis.set_major_formatter(
                matplotlib.dates.DateFormatter(date_formatter)
            )
        if max_tick_labels is not None:
            axes.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_tick_labels))
        axes.tick_params(axis="y", labelrotation=rotation, labelsize=font_size)
    return axes


def remove_tick_labels(ax: plt.axes, axis: str = "x"):
    """Remove ticks and tick labels for specified axis"""
    set_axis_tick_labels(axes=ax, values=[], labels=[], axis=axis)


def _plot_colorbar(
    plot, cax: kego.constants.TYPE_MATPLOTLIB_AXES | None = None
) -> kego.constants.TYPE_MATPLOTLIB_COLORBAR:
    colorbar = plt.colorbar(plot, cax=cax)
    return colorbar


def plot_colorbar(
    plot: matplotlib.cm.ScalarMappable,
    cax: Optional[kego.constants.TYPE_MATPLOTLIB_AXES] = None,
    label: str | None = None,
    font_size: float = kego.constants.DEFAULT_FONTSIZE_SMALL,
) -> kego.constants.TYPE_MATPLOTLIB_COLORBAR:
    colorbar = _plot_colorbar(plot, cax=cax)
    ax_colorbar = colorbar.ax
    if label is not None:
        ax_colorbar.set_ylabel(label, fontdict={"fontsize": font_size})
    set_axis_tick_labels(ax_colorbar, font_size=font_size, axis="y")
    return colorbar


def set_title(
    axes: kego.constants.TYPE_MATPLOTLIB_AXES,
    title: str | None,
    font_size: float = kego.constants.DEFAULT_FONTSIZE_SMALL,
):
    if title is not None:
        if isinstance(title, str) and len(title) > 0:
            axes.set_title(title, fontdict={"fontsize": font_size})
