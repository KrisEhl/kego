import itertools
import logging

import numpy as np

from mlframework.lists import is_listlike, to_nlength_tuple
from mlframework.plotting.figures import plot_legend, save_figure
from mlframework.plotting.utils_plotting import create_axes_grid


def plot_lines(
    xs: np.ndarray,
    ys: np.ndarray,
    labels: None | list[np.ndarray] = None,
    nx_max=4,
    filename: str | None = None,
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
        plot_line(x=xs[i_plot], y=ys[i_plot], label=labels[i_plot])
        plot_legend(axes=axes)
    save_figure(fig=figure, filename=filename)


def plot_line(
    x: np.ndarray,
    y: np.ndarray,
    label: str | None = None,
    filename: str | None = None,
):
    figure, axes, _ = create_axes_grid(n_columns=1, n_rows=1, unravel=True)
    axes.plot(x, y, label=label)
    save_figure(fig=figure, filename=filename)
