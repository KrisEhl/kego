import collections
import typing as t
from collections.abc import Sequence
from typing import Optional

import matplotlib.colors
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt

import kego.checks
import kego.constants
import kego.lists


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


def _create_figure(figure_size):
    return plt.figure(figsize=figure_size)


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


def set_title(
    axes: kego.constants.TYPE_MATPLOTLIB_AXES,
    title: str | None,
    font_size: float = kego.constants.DEFAULT_FONTSIZE_SMALL,
):
    if title is not None:
        if isinstance(title, str) and len(title) > 0:
            axes.set_title(title, fontdict={"fontsize": font_size})


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


def create_figure_axes(
    figure: t.Optional[plt.figure] = None,
    axes: t.Optional[plt.axes] = None,
    figure_size: Optional[t.Sequence] = None,
    font_size: t.Optional[int] = 10,
    aspect: str = "auto",
) -> t.Tuple[plt.figure, plt.axes]:
    """
    Creates figure with axes and sets font size

    Parameters:
    ----------
    fig: Figure if available
    ax: Axes if available
    figure_size: Size of figure (height, width), default (10, 6)
    font_size: Font size

    Returns
    -------
    matplotlib figure and axes
    """
    if font_size is not None:
        set_font(font_size=font_size)
    if figure_size is None:
        figure_size = (10, 6)
    if figure is None and axes is None:
        figure = plt.figure(figsize=figure_size)
        axes = plt.gca()
        axes.set_aspect(aspect)
    if axes is None:
        axes = plt.gca()
        axes.set_aspect(aspect)
    if figure is None:
        figure = axes.get_figure()
    return figure, axes


def create_axes_grid(
    n_columns: int,
    n_rows: int,
    figure_size: tuple[float, float] | None = None,
    widths_along_x: list[float] | None = None,
    heights_along_y: list[float] | None = None,
    top: float = 0.05,
    bottom: float = 0.05,
    right: float = 0.05,
    left: float = 0.05,
    spacing_x: float = 0.05,
    spacing_y: float = 0.05,
    spacing_colorbar: float = 0.04,
    colorbar_width: float = 0.07,
    colorbar_skip_row_col: list[tuple[int, int]] | None = None,
    colorbar_include_row_columns: list[tuple[int, int]] | None = None,
    colorbar_off: bool = True,
    skip_columns: list[int] | None = None,
    skip_rows: list[int] | None = None,
    skip_row_column: list[tuple[int, int]] | None = None,
    unravel: bool = False,
) -> t.Tuple[plt.figure, np.ndarray, np.ndarray]:
    """
    Create figure with grid of axes. Units are scaled to normalized figure size (max value: 1)
    unless specified otherwise.
    See also https://docs.google.com/presentation/d/1Ec-000rszefjCsv_sgUO62eGyT0-YbYzbk1aLQkU2gM/edit?usp=sharing
    Parameters:
    ----------
    n_columns: Number of columns
    n_rows: Number of rows
    figure_size: Size of figure
    widths_along_x: Normalized width ratios along horizontal direction
    heights_along_y: Normalized height ratios along vertical direction
    top: Offset of axes grid from the top
    bottom: Offset of axes grid from the bottom
    right: Offset of axes grid from the right
    left: Offset of axes grid from the left
    spacing_x: Spacing between axes along horizontal direction
    spacing_y: Spacing between axes along vertical direction
    spacing_colorbar: Spacing between axes and colorbar axes
    colorbar_width: Width of the colorbars
    colorbar_skip_row_col: (row, col) pairs of colorbars to be skipped
    colorbar_include_row_col: (row, col) pairs of colorbars to be plotted
    colorbar_off: No colorbars plotted
    skip_cols: Leave these columns blank
    skip_rows: Leave these rows blank
    skip_row_col: (row, col) pairs of colorbars for which no axes plotted

    Returns
    -------
    Figure, List of axes, List of colorbar axes
    """
    if figure_size is None:
        figure_size = (10.0, 6.0)
    if (
        colorbar_off
        and colorbar_include_row_columns is None
        and colorbar_skip_row_col is None
    ):
        colorbar_skip_row_col = [
            (j, i) for i in range(n_columns) for j in range(n_rows)
        ]
    fig = _create_figure(figure_size)
    kego.checks.all_same_type([n_columns, n_rows], int)
    kego.checks.all_same_type(
        [
            top,
            bottom,
            right,
            left,
            spacing_x,
            spacing_y,
            spacing_colorbar,
            colorbar_width,
        ],
        float,
    )
    kego.checks.assert_same_type_as(skip_row_column, [()])
    kego.checks.assert_same_type_as(colorbar_skip_row_col, [()])
    kego.checks.assert_same_type_as(colorbar_include_row_columns, [()])
    kego.checks.assert_shape(widths_along_x, (n_columns,), "widths_along_x")
    kego.checks.assert_shape(heights_along_y, (n_rows,), "heights_along_y")

    if skip_rows is None:
        skip_rows = []
    if skip_columns is None:
        skip_columns = []
    if skip_row_column is None:
        skip_row_column = []

    height_total = 1 - (n_rows - 1) * spacing_y - top - bottom
    if heights_along_y is not None:
        heights_along_y = [
            x / sum(heights_along_y) * height_total for x in heights_along_y
        ]
        axes_heights = np.array(
            [[x for i in range(n_columns)] for x in heights_along_y]
        )
    else:
        height = height_total / n_rows
        axes_heights = np.array(
            [[height for i in range(n_columns)] for j in range(n_rows)]
        )

    colorbar_heights = np.zeros((n_rows, n_columns))
    colorbar_widths = np.zeros((n_rows, n_columns))
    spacings_colorbar = np.zeros((n_rows, n_columns))
    skip_col_colorbar = []
    if colorbar_include_row_columns is not None:
        include_col_colorbar = [x[1] for x in colorbar_include_row_columns]
        skip_col_colorbar = [
            x for x in range(n_columns) if x not in include_col_colorbar
        ]
    if colorbar_skip_row_col is not None:
        skip_col = [x[1] for x in colorbar_skip_row_col]
        counts = collections.Counter(skip_col)
        for k, v in counts.items():
            if v == n_rows:
                skip_col_colorbar.append(int(k))
    for i_col in range(n_columns):
        for i_row in range(n_rows):
            if i_col in skip_col_colorbar:
                continue
            colorbar_heights[i_row, i_col] = axes_heights[i_row, i_col]
            colorbar_widths[i_row, i_col] = colorbar_width
            spacings_colorbar[i_row, i_col] = spacing_colorbar
    width_total = (
        1
        - (n_columns - 1) * spacing_x
        - left
        - right
        - max(sum(colorbar_widths[i, :]) for i in range(n_rows))
        - max(sum(spacings_colorbar[i, :]) for i in range(n_rows))
    )
    if widths_along_x is not None:
        widths_along_x = [x / sum(widths_along_x) * width_total for x in widths_along_x]
        axes_widths = np.array([widths_along_x for x in range(n_rows)])
    else:
        width = width_total / n_columns
        axes_widths = np.array(
            [[width for i in range(n_columns)] for j in range(n_rows)]
        )

    axes = [[None for i in range(n_columns)] for j in range(n_rows)]
    axes_colorbar = [[None for i in range(n_columns)] for j in range(n_rows)]
    for i_col in range(n_columns):
        if i_col in skip_columns:
            continue
        for i_row in range(n_rows):
            if i_row in skip_rows:
                continue
            if pair_in_list([i_row, i_col], skip_row_column):
                continue
            (
                axes_left,
                axes_bottom,
                axes_width,
                axes_height,
            ) = _determine_axes_dimensions(
                n_rows,
                bottom,
                left,
                spacing_x,
                spacing_y,
                axes_heights,
                colorbar_widths,
                spacings_colorbar,
                i_col,
                i_row,
                axes_widths,
            )
            axes[i_row][i_col] = plt.axes(
                [axes_left, axes_bottom, axes_width, axes_height]
            )
            if colorbar_skip_row_col is not None and pair_in_list(
                [i_row, i_col], colorbar_skip_row_col
            ):
                continue
            if colorbar_include_row_columns is not None and not pair_in_list(
                [i_row, i_col], colorbar_include_row_columns
            ):
                continue
            (
                colorbar_width,
                colorbar_left,
                colorbar_bottom,
                colorbar_height,
            ) = _determine_colorbar_axes_dimensions(
                colorbar_heights,
                colorbar_widths,
                spacings_colorbar,
                i_col,
                i_row,
                axes_left,
                axes_bottom,
                axes_width,
            )
            axes_colorbar[i_row][i_col] = plt.axes(
                [colorbar_left, colorbar_bottom, colorbar_width, colorbar_height]
            )
    axes, axes_colorbar = np.array(axes, dtype=object), np.array(
        axes_colorbar, dtype=object
    )
    if unravel:
        axes = kego.lists.flatten_list(axes)
        axes_colorbar = kego.lists.flatten_list(axes_colorbar)
        if n_columns * n_rows == 1:
            axes = axes[0]  # type: ignore
            axes_colorbar = axes_colorbar[0]  # type: ignore
    return fig, axes, axes_colorbar


def pair_in_list(pair, _list):
    return tuple(pair) in list(_list) or list(pair) in list(_list)


def _determine_colorbar_axes_dimensions(
    colorbar_heights,
    colorbar_widths,
    spacings_colorbar,
    i_col,
    i_row,
    axes_left,
    axes_bottom,
    axes_width,
):
    colorbar_left = axes_left + axes_width + spacings_colorbar[i_row][i_col]
    colorbar_bottom = axes_bottom
    colorbar_width = colorbar_widths[i_row, i_col]
    colorbar_height = colorbar_heights[i_row, i_col]
    return colorbar_width, colorbar_left, colorbar_bottom, colorbar_height


def _determine_axes_dimensions(
    n_rows,
    bottom,
    left,
    spacing_x,
    spacing_y,
    axes_heights,
    colorbar_widths,
    spacings_colorbar,
    i_col,
    i_row,
    axes_widths,
):
    axes_heights_cumulative = axes_heights[slice(i_row + 1, None), i_col]
    axes_left = (
        left
        + sum(axes_widths[i_row, 0:i_col])
        + sum(colorbar_widths[i_row, 0:i_col])
        + sum(spacings_colorbar[i_row, 0:i_col])
        + i_col * spacing_x
    )
    axes_bottom = (
        bottom + sum(axes_heights_cumulative) + (n_rows - i_row - 1) * spacing_y
    )
    axes_width = axes_widths[i_row, i_col]
    axes_height = axes_heights[i_row, i_col]
    return axes_left, axes_bottom, axes_width, axes_height


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
