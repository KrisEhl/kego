import typing as t

import matplotlib.pyplot as plt

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
