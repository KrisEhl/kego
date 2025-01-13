import logging

import matplotlib.colors
import matplotlib.dates
import numpy as np

import kego.constants

logger = logging.getLogger(__name__)


def _plot_colormesh(
    axes: kego.constants.TYPE_MATPLOTLIB_AXES,
    xx: np.ndarray,
    yy: np.ndarray,
    matrix: np.ndarray,
    norm_object: matplotlib.colors.LogNorm | matplotlib.colors.Normalize | None,
    colormap: str,
    linewidth: float = 0,
    rasterized: bool = True,
    edgecolor: str = "face",
    convert_zeros_to_nan: bool = False,
):
    if convert_zeros_to_nan:
        matrix[matrix == 0] = np.nan
        logger.debug("Convert zeros to np.nan in matrix.")
    plot = axes.pcolormesh(
        xx,
        yy,
        matrix,
        norm=norm_object,
        cmap=colormap,
        linewidth=linewidth,
        rasterized=rasterized,
    )
    plot.set_edgecolor(edgecolor)
    return plot
