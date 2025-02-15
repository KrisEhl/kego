import collections
import logging
import os.path
import pathlib
from typing import Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt

import kego.constants
from kego.plotting.utils_plotting import pair_in_list, set_font


def save_figure(
    fig: kego.constants.TYPE_MATPLOTLIB_FIGURES,
    filename: str | pathlib.Path | None = None,
    dpi: int = 450,
) -> None:
    """Save figure to filename"""
    if filename is not None:
        logging.info(f"... saving {filename}")
        folder = os.path.split(filename.__str__())[0]
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)


def plot_legend(axes):
    axes.legend()
    return axes


def create_figure_axes(
    figure: Optional[plt.figure] = None,
    axes: Optional[plt.axes] = None,
    figure_size: Optional[Sequence] = None,
    font_size: Optional[int] = 10,
    aspect: str = "auto",
) -> tuple[plt.figure, plt.axes]:
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


def _create_figure(figure_size):
    return plt.figure(figsize=figure_size)


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
) -> tuple[plt.figure, np.ndarray, np.ndarray]:
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
    unravel: Whether to flatten axes lists.

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
