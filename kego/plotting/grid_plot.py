from typing import Literal

import numpy as np

from kego.constants import DEFAULT_COLORMAP
from kego.lists import to_nlength_tuple
from kego.plotting.axes_utils import set_axes
from kego.plotting.colormesh import _plot_colormesh
from kego.plotting.utils_plotting import create_axes_grid, get_norm


def grid_plot(
    data_as_dict: dict,
    xs: list | None = None,
    ys: list | None = None,
    style: Literal["colormesh"] = "colormesh",
    labels: list | None = None,
    nx_max: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
    colormap: str = DEFAULT_COLORMAP,
):
    if data_as_dict is None:
        if xs is None or ys is None:
            raise ValueError(
                f"{data_as_dict=} is not set. Please specify both {xs=} and {ys=} instead"
            )
    if data_as_dict is not None and (xs is not None or ys is not None):
        raise ValueError(f"Can only specify {data_as_dict=} or [{xs=} and {ys=}].")
    if xs is not None or ys is not None:
        raise NotImplementedError(f"Specifying {xs=} or {ys=} is not implemented!")

    n_plots = len(data_as_dict.items())
    n_columns = nx_max if n_plots >= nx_max else n_plots
    n_rows = int(np.ceil(n_plots / nx_max))
    figure, axes_grid, axes_colorbar = create_axes_grid(
        n_columns=n_columns, n_rows=n_rows
    )

    for i_plot, (label, data) in enumerate(data_as_dict.items()):
        if style == "colormesh":
            axes = np.ndarray.flatten(axes_grid)[i_plot]
            xx, yy = np.meshgrid(
                np.arange(np.shape(data)[1]), np.arange(np.shape(data)[0])
            )
            print(f"{xx.shape=}, {yy.shape=}, {data.shape=}")
            norm = get_norm("linear", vmin=vmin, vmax=vmax)
            _plot_colormesh(
                axes=axes,
                xx=xx,
                yy=yy,
                matrix=data,
                norm_object=norm,
                colormap=colormap,
            )
            set_axes(ax=axes, title=label)
    return figure
