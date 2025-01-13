import itertools
import logging
from typing import Sequence

import numpy as np

import kego.plotting.utils_plotting
from kego.lists import is_listlike, to_nlength_tuple
from kego.plotting.figures import plot_legend, save_figure
from kego.plotting.utils_plotting import create_axes_grid


def plot_lines(
    xs: np.ndarray,
    ys: np.ndarray,
    labels: None | tuple[np.ndarray] = None,
    nx_max=4,
    filename: str | None = None,
    log: str | tuple[str, str] = "false",
):
    assert len(np.shape(xs)) != 2, f"xs needs to be of shape 2 but is {xs.shape=}!"
    assert len(np.shape(ys)) != 2, f"xs needs to be of shape 2 but is {ys.shape=}!"

    n_plots = np.shape(xs)[0]
    labels = to_nlength_tuple(labels, n_plots)
    if n_plots >= nx_max:
        n_rows = n_plots
    n_columns = np.ceil(n_plots / nx_max)
    figure, axes_grid, _ = create_axes_grid(n_columns=n_columns, n_rows=n_rows)
    for i_plot, (i_row, i_column) in enumerate(
        itertools.product(range(n_rows), range(n_columns))
    ):
        axes = axes_grid[i_row, i_column]
        plot_line(x=xs[i_plot], y=ys[i_plot], label=labels[i_plot], axes=axes)
        plot_legend(axes=axes)
    save_figure(fig=figure, filename=filename)


def plot_line(
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    log: str | tuple[str, str] = "false",
    label: str | None = None,
    replace_x_labels_at: Sequence | None = None,
    replace_x_labels_with: Sequence | None = None,
    replace_y_labels_at: Sequence | None = None,
    replace_y_labels_with: Sequence | None = None,
    rotation_x_labels: int = 0,
    rotation_y_labels: int = 0,
    filename: str | None = None,
    axes=None,
    font_size: int = 12,
    symlog_linear_threshold: float | None = None,
    label_x: str | None = None,
    label_y: str | None = None,
):
    if axes is None:
        figure, axes, _ = create_axes_grid(n_columns=1, n_rows=1, unravel=True)
    _log = to_nlength_tuple(log)
    _xlim = to_nlength_tuple(xlim)
    _ylim = to_nlength_tuple(ylim)
    axes.plot(x, y, label=label)
    kego.plotting.utils_plotting.set_x_log(
        axes, _log[0], axis_symlog_linear_threshold=symlog_linear_threshold
    )
    kego.plotting.utils_plotting.set_y_log(
        axes, _log[1], axis_symlog_linear_threshold=symlog_linear_threshold
    )
    kego.plotting.utils_plotting.set_axis_tick_labels(
        axes,
        replace_x_labels_at,
        replace_x_labels_with,
        axis="x",
        rotation=rotation_x_labels,
        font_size=font_size,
    )
    kego.plotting.utils_plotting.set_axis_tick_labels(
        axes,
        replace_y_labels_at,
        replace_y_labels_with,
        axis="y",
        rotation=rotation_y_labels,
        font_size=font_size,
    )
    kego.plotting.utils_plotting.set_axes_label(axes, label_x, "x", font_size=font_size)
    kego.plotting.utils_plotting.set_axes_label(axes, label_y, "y", font_size=font_size)
    axes.set_ylim(_ylim)
    axes.set_xlim(_xlim)
    save_figure(fig=figure, filename=filename)
    return axes
