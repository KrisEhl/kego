import logging
from typing import Literal

import matplotlib.colors
import matplotlib.dates
import numpy as np

import kego.constants
from kego.plotting.axes import plot_colorbar, set_axes
from kego.plotting.figures import create_figure_axes
from kego.plotting.utils_plotting import get_norm

logger = logging.getLogger(__name__)


def _plot_colormesh(
    axes: kego.constants.TYPE_MATPLOTLIB_AXES,
    matrix: np.ndarray,
    xx: np.ndarray | None,
    yy: np.ndarray | None,
    norm_object: matplotlib.colors.LogNorm | matplotlib.colors.Normalize | None,
    colormap: str,
    linewidth: float = 0,
    rasterized: bool = True,
    edgecolor: str = "face",
    shading: str = "auto",
    convert_zeros_to_nan: bool = False,
):
    if convert_zeros_to_nan:
        matrix[matrix == 0] = np.nan
        logger.debug("Convert zeros to np.nan in matrix.")
    args: tuple = (xx, yy, matrix)
    if xx is None and yy is None:
        args = (matrix,)
    plot = axes.pcolormesh(
        *args,
        norm=norm_object,
        cmap=colormap,
        linewidth=linewidth,
        rasterized=rasterized,
        shading=shading,
    )
    plot.set_edgecolor(edgecolor)
    return plot


def plot_colormesh(
    matrix: np.ndarray,
    xx: np.ndarray | None = None,
    yy: np.ndarray | None = None,
    norm: Literal["log", "symlog", "linear"] | None = None,
    vmin=None,
    vmax=None,
    colormap="bwr",
    figure=None,
    axes=None,
    shading: str = "auto",
    title: str | None = None,
    label_matrix: str | None = None,
    font_size=kego.constants.DEFAULT_FONTSIZE_SMALL,
):
    figure, axes = create_figure_axes(figure=figure, axes=axes, font_size=font_size)
    norm = get_norm(norm=norm, vmin=vmin, vmax=vmax)
    plot = _plot_colormesh(
        axes=axes,
        xx=xx,
        yy=yy,
        matrix=matrix,
        norm_object=norm,
        colormap=colormap,
        shading=shading,
    )
    plot_colorbar(plot=plot, label=label_matrix, font_size=font_size)
    set_axes(ax=axes, title=title, fontsize=font_size)
    return axes
