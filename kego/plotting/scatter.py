import numpy as np

import kego.constants
from kego.plotting.figures import create_figure_axes


def plot_scatter(x=None, y=None, colors=None, axes=None, figure=None):
    if x is None:
        x = np.arange(len(y))
    if y is None:
        y = np.arange(len(x))
    figure, axes = create_figure_axes(figure=figure, axes=axes)
    axes.scatter(x=x, y=y, c=colors)
    return axes
