import kego.constants
from kego.plotting.figures import create_figure_axes


def plot_scatter(x, y, colors, axes=None, figure=None):
    figure, axes = create_figure_axes(figure=figure, axes=axes)
    axes.scatter(x=x, y=y, c=colors)
    return axes
