import logging
import pathlib
from collections.abc import Sequence
from typing import Literal

import matplotlib.container
import numpy as np
import pandas as pd

import kego.checks
import kego.constants
import kego.lists
import kego.plotting.axes_utils
import kego.plotting.colormesh
import kego.plotting.figures
import kego.plotting.histogram_2d_utils
import kego.plotting.utils_plotting

logger = logging.getLogger(__name__)


def plot_histogram_2d(
    x_key_or_values: np.ndarray | str,
    y_key_or_values: np.ndarray | str,
    df: pd.DataFrame | None = None,
    bin_edges: tuple[np.ndarray, np.ndarray] | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    log: str | tuple[str, str] = "false",
    filename: str | pathlib.Path | None = None,
    n_bins: int | tuple[int, int] = 60,
    n_bins_linear: int = 10,
    norm: str | None = None,
    norm_symlog_linear_threshold: float | None = None,
    axis_symlog_linear_threshold: float | tuple[float, float] = 1e-9,
    label_x: str | None = None,
    label_y: str | None = None,
    label_colorbar: str | None = "counts",
    font_size: float = kego.constants.DEFAULT_FONTSIZE_SMALL,
    annotate: bool = False,
    annotate_round_to_base: int | None = None,
    annotate_color: str = "gray",
    annotate_line: Literal["unity"] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    facet_column: str | None = None,
    facet_row: str | None = None,
    facet_show_label_x: list[int] | str = "minimal",
    facet_show_label_y: list[int] | str = "minimal",
    marginal_x: str | None = "default",
    marginal_x_label_x: str | None = None,
    marginal_x_label_y: str | None = "counts",
    marginal_x_show_xticks: bool = False,
    marginal_y: str | None = "default",
    marginal_y_label_x: str | None = "counts",
    marginal_y_label_y: str | None = None,
    marginal_y_show_yticks: bool = False,
    marginal_color: str | None = "#1f77b4",
    figure_size: tuple[float, float] | None = None,
    colormap: str = kego.constants.DEFAULT_COLORMAP,
    colorbar_width: float = 0.02,
    spacing_x: float | None = None,
    spacing_y: float | None = None,
    spacing_colorbar: float = 0.03,
    title: str = "",
    convert_zeros_to_nan: bool = True,
    axes: kego.constants.TYPE_MATPLOTLIB_AXES | None = None,
    figure: kego.constants.TYPE_MATPLOTLIB_FIGURES | None = None,
    axes_colorbar: kego.constants.TYPE_MATPLOTLIB_AXES | None = None,
) -> dict:
    """
    Plots 2d histogram including histograms in margins and facet

    Log (axis) types included "false", "symlog", "log"
    Norm takes care of colorbar scale ("log", "linear"/None, "symlog")
    Allows for marginal 1D histograms on top and to left,
        set `marginal_x="histogram"` and `marginal_y="histogram"`
    Can annotate values on 2d histogram, set `annotate="True"`
    Allows splitting of 2d histogram into unique values of variable
        along rows (facet_row) and/or columns (facet_column)
        (Cannot be used with `marginal_x`/`marginal_y`)

    Parameters:
    ----------
    x_key_or_values:
        Values binned along x-axis, array(N) or column name found in `df`
    y_key_or_values:
        Values binned along y-axis, array(N) or column name found in `df`
    df:
        Dataframe used for `x`, `y`, `facet_column`, `facet_row`
    bin_edges:
        Bin edges provided as [x_edges, y_edges],
        if `None` computed based on data and `xlim`/`ylim` provided and `n_bins`.
    xlim:
        Limits of x-axis and bin_edges, if None determined from (min, max) of x-values
    ylim:
        Limits of y-axis and bin_edges, if None determined from (min, max) of y-values
    log:
        Type of bins ("false", "symlog", "log")
    filename:
        Figure saved to file if provided
    n_bins:
        Number of bins if `bin_edges` is None,
        can be provided as (x-axis n_bins, y-axis n_bins) or n_bins used for both x-axis/y-axis
    n_bins_linear:
        Only relevant if `log="symlog"`, Number of bins for linear part of bins (around zero)
    norm:
        Scaling of colorbar (None/"linear", "log", "symlog")
    norm_symlog_linear_threshold:
        Required when `norm="symlog"`.
        Threshold value below which symlog of norm becomes linear.
    axis_symlog_linear_threshold:
        Required when `log="symlog"`.
        Threshold value below which symlog of axis becomes linear.
    label_x:
        Label of x-axis
    label_y:
        Label of y-axis
    label_colorbar:
        Label of colorbar
    font_size:
        Size of font
    annotate:
        Whether to show number of samples on plot
    annotate_round_to_base:
        Round annotated values to this base integer
    annotate_color:
        Color of samples to be annotated
    vmin:
        Minimum value of colormap
    vmax:
        Maximum value of colormap
    facet_column:
        Assign all samples belonging to unique value from this field to histogram as columns
    facet_row:
        Assign all samples belonging to unique value from this field to histogram as rows
    facet_show_label_x:
        Which x-labels to show along rows provided as list of integers or use presets "off"/"all"/"minimal".
    facet_show_label_y:
        Which y-labels to show along columns provided as list of integers or use presets "off"/"all"/"minimal".
    marginal_x:
        Add additional plot next to histogram plot ("histogram"/"default"/None)
    marginal_x_label_x:
        X-axis label for plot `marginal_x`
    marginal_x_label_y:
        Y-axis label for plot `marginal_x`
    marginal_x_show_xticks:
        Wether to show ticks along the x-axis for plot `marginal_x`
    marginal_y:
        Add additional plot on top of histogram plot ("histogram"/"default"/None)
    marginal_y_label_x:
        X-axis label for plot `marginal_y`
    marginal_y_label_y:
        Y-axis label for plot `marginal_y`
    marginal_y_show_yticks:
        Wether to show ticks along the y-axis for plot `marginal_y`
    marginal_color:
        Color of histogram bars in marginal plots
    figure_size:
        Size of figure (height, width)
    colormap:
        Matplotlib colormap name
    colorbar_width:
        Width of the colorbars
    spacing_x:
        Spacing between axes along horizontal direction
    spacing_y:
        Spacing between axes along vertical direction
    spacing_colorbar:
        Spacing between axes and colorbar axes
    title:
        Title of figure
    convert_zeros_to_nan:
        Whether to convert all zero values in histogram 2d to nan.
        This will exclude these values from being colored according to colormap but render them white.
    axes:
        Plots 2d histogram on specified axes.
        Excludes usage of `marginal_x`, `marginal_y`, `facet_column` and/or `facet_row`.
    figure:
        Used as matplotlib figure for plot of 2d histogram when `axes` specified.
    axes_colorbar:
        Used as axes for colorbar of 2d histogram when `axes` specified.

    Returns
    -------
    Dictionary of axes `axes_2d_histograms_{i_row}{i_column}`
    and histograms `2d_histograms_{i_row}{i_column}`
    """

    axes_style = kego.plotting.histogram_2d_utils._resolve_axes_style(axes)

    (
        no_marginal_plots,
        marginal_x,
        marginal_y,
        spacing_x,
        spacing_y,
    ) = kego.plotting.histogram_2d_utils._resolve_defaults(
        facet_column,
        facet_row,
        marginal_x,
        marginal_y,
        spacing_x,
        spacing_y,
        axes_style,
    )
    kego.plotting.histogram_2d_utils._check_parameter_consistency(
        facet_column, facet_row, marginal_x, marginal_y, df
    )

    (
        column_masks,
        n_facet_column_values,
        facet_column_title,
    ) = kego.plotting.histogram_2d_utils._prepare_facet(df, facet_column)
    (
        row_masks,
        n_facet_row_values,
        facet_row_title,
    ) = kego.plotting.histogram_2d_utils._prepare_facet(df, facet_row)

    (
        n_columns,
        n_rows,
    ) = kego.plotting.histogram_2d_utils._set_number_rows_and_columns(
        marginal_x, marginal_y, n_facet_column_values, n_facet_row_values
    )

    (
        xaxis_labels,
        yaxis_labels,
    ) = kego.plotting.histogram_2d_utils._set_axis_labels(
        n_columns,
        n_rows,
        facet_show_label_x,
        facet_show_label_y,
        label_x,
        label_y,
        facet_column,
        facet_row,
    )

    masks = kego.plotting.histogram_2d_utils._set_masks(
        column_masks, row_masks, n_columns, n_rows
    )

    titles = kego.plotting.histogram_2d_utils._determine_axes_titles(
        facet_column,
        facet_row,
        title,
        facet_column_title,
        facet_row_title,
        n_columns,
        n_rows,
    )

    (
        skip_row_col,
        widths_along_x,
        heights_along_y,
        colorbar_include_row_col,
        colorbar_off,
    ) = kego.plotting.histogram_2d_utils._prepare_axes_grid_creation(
        marginal_x, marginal_y
    )
    if axes_style == "axes_grid":
        (
            fig,
            axes_grid,
            axes_colorbar_grid,
        ) = kego.plotting.utils_plotting.create_axes_grid(
            n_columns=n_columns,
            n_rows=n_rows,
            figure_size=figure_size,
            skip_row_column=skip_row_col,
            colorbar_width=colorbar_width,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            widths_along_x=widths_along_x,
            heights_along_y=heights_along_y,
            colorbar_include_row_columns=colorbar_include_row_col,
            spacing_colorbar=spacing_colorbar,
            colorbar_off=colorbar_off,
            left=0.08,
            bottom=0.1,
        )
    else:
        fig, axes = kego.plotting.utils_plotting.create_figure_axes(
            figure=figure, axes=axes
        )
    results_dic = {}
    for i_row in range(n_rows):
        for i_column in range(n_columns):
            axes_marginal = None
            if axes_style == "axes_grid":
                axes = axes_grid[i_row, i_column]
                axes_colorbar = axes_colorbar_grid[i_row, i_column]
                if not no_marginal_plots:
                    axes = axes_grid[1][0]
                    axes_colorbar = axes_colorbar_grid[1][1]
                    axes_marginal = axes_grid
            mask = masks[i_row, i_column]
            title = titles[i_row, i_column]
            _ax, _hist = _plot_histogram_2d(
                x_key_or_values,
                y_key_or_values,
                axes=axes,
                axes_colorbar=axes_colorbar,
                df=df,
                bin_edges=bin_edges,
                xlim=xlim,
                ylim=ylim,
                log=log,
                n_bins=n_bins,
                n_bins_linear=n_bins_linear,
                norm=norm,
                norm_symlog_linear_threshold=norm_symlog_linear_threshold,
                linear_thresh=axis_symlog_linear_threshold,
                label_x=xaxis_labels[i_row, i_column],
                label_y=yaxis_labels[i_row, i_column],
                label_colorbar=label_colorbar,
                font_size=font_size,
                annotate=annotate,
                annotate_round_to_base=annotate_round_to_base,
                annotate_color=annotate_color,
                vmin=vmin,
                vmax=vmax,
                marginal_x=marginal_x,
                marginal_x_label_x=marginal_x_label_x,
                marginal_x_label_y=marginal_x_label_y,
                marginal_x_show_xticks=marginal_x_show_xticks,
                marginal_y=marginal_y,
                marginal_y_label_x=marginal_y_label_x,
                marginal_y_label_y=marginal_y_label_y,
                marginal_y_show_yticks=marginal_y_show_yticks,
                marginal_color=marginal_color,
                axes_marginal=axes_marginal,
                colormap=colormap,
                title=title,
                mask=mask,
                convert_zeros_to_nan=convert_zeros_to_nan,
            )
            if no_marginal_plots:
                results_dic[f"axes_2d_histograms_{i_row}{i_column}"] = _ax
                results_dic[f"2d_histograms_{i_row}{i_column}"] = _hist
            else:
                results_dic["axes_2d_histograms_10"] = _ax
                results_dic["2d_histograms_10"] = _hist
                results_dic["axes_histograms_00"] = axes_grid[0, 0]
                results_dic["axes_histograms_11"] = axes_grid[1, 1]
            if not no_marginal_plots:
                break
            add_annotation_line(style=annotate_line, axes=axes)
        if not no_marginal_plots:
            break
    kego.plotting.figures.save_figure(fig, filename)
    return results_dic


def add_annotation_line(
    style: Literal["unity"] | None, axes, alpha: float = 0.4, plot_style: str = "--k"
):
    if style is None:
        return
    if style == "unity":
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        axes.plot(xlim, ylim, plot_style, label="unity", alpha=alpha)


def _get_label(label: str | None) -> str:
    if label is not None and label:
        return label
    else:
        return ""


def _get_values_and_axis_label_from_dataframe(
    df: pd.DataFrame | None, key_or_values: np.ndarray | str, label: str | None
) -> tuple[np.ndarray, str | None]:
    if isinstance(key_or_values, str):
        if df is not None:
            if label is None:
                label = key_or_values
            if label == "off":
                label = None
            values = df[key_or_values].values
        else:
            raise ValueError(
                f"{df=}, if {key_or_values=} given as string, dataset required"
            )
    else:
        values = key_or_values
    return values, label


def _plot_histogram_2d(
    x_key_or_values: np.ndarray | str,
    y_key_or_values: np.ndarray | str,
    axes: kego.constants.TYPE_MATPLOTLIB_AXES,
    axes_colorbar: kego.constants.TYPE_MATPLOTLIB_AXES,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    df: pd.DataFrame | None = None,
    bin_edges: tuple[np.ndarray, np.ndarray] | None = None,
    log: str | tuple[str, str] = "false",
    fig: kego.constants.TYPE_MATPLOTLIB_FIGURES | None = None,
    n_bins: int | tuple[int, int] = 60,
    n_bins_linear: int = 10,
    norm: str | None = None,
    norm_symlog_linear_threshold: float | None = None,
    linear_thresh: float | tuple[float, float] = 1e-9,
    label_x: str | None = None,
    label_y: str | None = None,
    label_colorbar: str | None = None,
    font_size: float = kego.constants.DEFAULT_FONTSIZE_SMALL,
    annotate: bool = False,
    annotate_round_to_base: int | None = None,
    annotate_color: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    marginal_x: str | None = "histogram",
    marginal_x_label_x: str | None = None,
    marginal_x_label_y: str | None = "N",
    marginal_x_show_xticks: bool = False,
    marginal_y: str | None = "histogram",
    marginal_y_label_x: str | None = "N",
    marginal_y_label_y: str | None = None,
    marginal_y_show_yticks: bool = False,
    marginal_color: str | None = None,
    colormap: str = kego.constants.DEFAULT_COLORMAP,
    title: None | str = None,
    axes_marginal: list | None = None,
    mask: np.ndarray | None = None,
    convert_zeros_to_nan: bool = True,
) -> tuple[kego.constants.TYPE_MATPLOTLIB_AXES, np.ndarray]:
    """
    Plots 2d histogram

    Parameters:
    ----------
    axes_marginal:
        Matplotlib axes to plot margins into
    mask:
        Mask x/y values (require same shape as x/y)

    See definition of `plot_histogram_2d` for explanation of parameters

    Returns
    -------
    Axes, histogram values
    """

    x_values, label_x = _get_values_and_axis_label_from_dataframe(
        df, x_key_or_values, label_x
    )
    y_values, label_y = _get_values_and_axis_label_from_dataframe(
        df, y_key_or_values, label_y
    )

    x_values, y_values = _get_xy_values(x_values, y_values, mask)

    xlim, ylim, log, n_bins, n_bins_linear, linear_thresh = _prepare_parameters(  # type: ignore
        xlim, ylim, log, n_bins, n_bins_linear, linear_thresh
    )

    if bin_edges is None:
        bin_edges_x = get_bin_edges(
            data=x_values,
            n_bins=n_bins[0],
            n_bins_linear=n_bins_linear[0],  # type: ignore
            symlog_linear_threshold=linear_thresh[0],
            log=log[0],
            vmin=xlim[0],
            vmax=xlim[1],
        )
        bin_edges_y = get_bin_edges(
            data=y_values,
            n_bins=n_bins[1],
            n_bins_linear=n_bins_linear[1],  # type: ignore
            symlog_linear_threshold=linear_thresh[1],
            log=log[1],
            vmin=ylim[0],
            vmax=ylim[1],
        )
    else:
        bin_edges_x, bin_edges_y = bin_edges

    kego.checks.validate_array(bin_edges_x, name="bin_edges_x")  # type: ignore
    kego.checks.validate_array(bin_edges_y, name="bin_edges_y")  # type: ignore

    norm_object = kego.plotting.utils_plotting.get_norm(
        norm,
        vmin=vmin,
        vmax=vmax,
        norm_symlog_linear_threshold=norm_symlog_linear_threshold,
    )
    H, bin_edges_x, bin_edges_y = np.histogram2d(
        x_values,
        y_values,
        bins=[np.array(bin_edges_x), np.array(bin_edges_y)],
    )
    H_plot = H.T
    X, Y = np.meshgrid(bin_edges_x, bin_edges_y)
    plot = kego.plotting.colormesh._plot_colormesh(
        axes,
        X,
        Y,
        H_plot,
        norm_object,
        colormap,
        linewidth=0,
        rasterized=True,
        convert_zeros_to_nan=convert_zeros_to_nan,
    )
    if annotate:
        kego.plotting.utils_plotting.annotate_values(
            H=H_plot,
            axes=axes,
            size_x=len(bin_edges_x) - 1,
            size_y=len(bin_edges_y) - 1,
            color=annotate_color,
            round_to_base=annotate_round_to_base,
            font_size=font_size,
        )
    kego.plotting.utils_plotting.plot_colorbar(
        plot, cax=axes_colorbar, label=label_colorbar, font_size=font_size
    )

    xlim, ylim = _find_axis_limits(xlim, ylim, bin_edges_x, bin_edges_y)

    kego.plotting.utils_plotting.set_x_log(
        axes, log[0], axis_symlog_linear_threshold=linear_thresh[0]
    )
    kego.plotting.utils_plotting.set_y_log(
        axes, log[1], axis_symlog_linear_threshold=linear_thresh[1]
    )
    kego.plotting.axes_utils.set_x_lim(axes, xlim)
    kego.plotting.axes_utils.set_y_lim(axes, ylim)

    kego.plotting.utils_plotting.set_title(axes, title, font_size=font_size)

    if marginal_x == "histogram" and axes_marginal is not None:
        plot_histogram(
            x_values,
            bin_edges=bin_edges_x,
            xlim=xlim,
            ylim=None,
            axes=axes_marginal[0][0],
            log=log[0],
            figure=fig,
            n_bins=n_bins[0],
            n_bins_linear=n_bins_linear[0],  # type: ignore
            symlog_linear_threshold=linear_thresh[0],
            label_x=_get_label(marginal_x_label_x),
            label_y=_get_label(marginal_x_label_y),
            font_size=font_size,
            color=marginal_color,
            tight_layout=False,
        )
        if not marginal_x_show_xticks:
            kego.plotting.utils_plotting.remove_tick_labels(axes_marginal[0][0], "x")
    if marginal_y == "histogram" and axes_marginal is not None:
        plot_histogram(
            y_values,
            bin_edges=bin_edges_y,
            xlim=None,
            ylim=ylim,
            axes=axes_marginal[1][1],
            log=log[1],
            figure=fig,
            n_bins=n_bins[1],
            n_bins_linear=n_bins_linear[1],  # type: ignore
            symlog_linear_threshold=linear_thresh[1],
            label_x=_get_label(marginal_y_label_x),
            label_y=_get_label(marginal_y_label_y),
            font_size=font_size,
            vertical=True,
            color=marginal_color,
            tight_layout=False,
            rotation_x_labels=270,
        )
        if not marginal_y_show_yticks:
            kego.plotting.utils_plotting.remove_tick_labels(axes_marginal[1][1], "y")
    kego.plotting.utils_plotting.set_axis_tick_labels(
        axes, font_size=font_size, axis="x"
    )
    kego.plotting.utils_plotting.set_axis_tick_labels(
        axes, font_size=font_size, axis="y"
    )
    kego.plotting.utils_plotting.set_axes_label(axes, label_x, "x", font_size=font_size)
    kego.plotting.utils_plotting.set_axes_label(axes, label_y, "y", font_size=font_size)
    return axes, H


def _prepare_parameters(
    xlim: tuple[float | None, float | None] | None,
    ylim: tuple[float | None, float | None] | None,
    log: str | tuple[str, str],
    n_bins: int | tuple[int, int],
    n_bins_linear: int | tuple[int, int],
    linear_thresh: float | tuple[float, float],
) -> tuple[
    tuple[float | None, float | None],
    tuple[float | None, float | None],
    tuple[str, str],
    tuple[int, int],
    tuple[int, int],
    tuple[float, float],
]:
    log = kego.lists.to_nlength_tuple(log)  # type: ignore
    n_bins = kego.lists.to_nlength_tuple(n_bins)  # type: ignore
    n_bins_linear = kego.lists.to_nlength_tuple(n_bins_linear)  # type: ignore
    xlim = kego.lists.to_nlength_tuple(xlim)  # type: ignore
    ylim = kego.lists.to_nlength_tuple(ylim)  # type: ignore
    n_bins = tuple(
        [i + 1 if i is not None else i for i in n_bins]  # type: ignore
    )  # using bin edges later, where n_edges = n_bins + 1
    n_bins_linear = tuple(
        [i + 1 if i is not None else i for i in n_bins_linear]  # type: ignore
    )  # using bin edges later, where n_edges = n_bins + 1
    linear_thresh = kego.lists.to_nlength_tuple(linear_thresh)  # type: ignore
    return xlim, ylim, log, n_bins, n_bins_linear, linear_thresh


def _get_xy_values(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    x = np.ndarray.flatten(x)  # type: ignore
    kego.checks.validate_array(x)
    y = np.ndarray.flatten(y)  # type: ignore
    kego.checks.validate_array(y)
    if x.shape != y.shape:
        raise ValueError(
            f"x and y need to be of same shape: {np.shape(x)} != {np.shape(y)}"
        )
    if mask is not None:
        x = x[mask]
        y = y[mask]
    return x, y


def plot_histogram(
    key_or_values: np.ndarray | str,
    df: pd.DataFrame | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    log: str | tuple[str, str] = "false",
    symlog_linear_threshold: float | None = None,
    bin_edges: np.ndarray | None = None,
    n_bins: int = 60,
    n_bins_linear: int = 10,
    label_x: str | None = None,
    label_y: str | None = "counts",
    figure: kego.constants.TYPE_MATPLOTLIB_FIGURES | None = None,
    axes: kego.constants.TYPE_MATPLOTLIB_AXES | None = None,
    filename: str | pathlib.Path | None = None,
    font_size: float = kego.constants.DEFAULT_FONTSIZE_LARGE,
    vertical: bool = False,
    alpha: float = 1,
    color: str | None = None,
    horizontal_line: int | None = None,
    tight_layout: bool = True,
    rotation_x_labels: int = 0,
    rotation_y_labels: int = 0,
    title: str = "",
) -> kego.constants.TYPE_MATPLOTLIB_AXES:
    """
    plots 1d histogram

    Log (axis) types included "false", "symlog", "log"
    Norm takes care of colorbar scale ("log", "linear"/None, "symlog")
    Can include a horizontal line at value `horizontal_line`

    Parameters:
    ----------
    x_key_or_values:
        Values binned along x-axis, array(N) or column name found in `df`
    df:
        Dataframe used for `key_or_values`
    xlim:
        Limits of x-axis and bin_edges, if None determined from (min, max) of x-values
    ylim:
        Limits of y-axis and bin_edges, if None determined from (min, max) of y-values
    log:
        Type of bins ("false", "symlog", "log")
    symlog_linear_threshold:
        Required when `log="symlog"`.
        Threshold value below which symlog becomes linear.
    bin_edges:
        Bin edges, if `None` computed based on data `xlim` provided and `n_bins`.
    n_bins:
        Number of bins if `bin_edges` is None.
    n_bins_linear:
        Only relevant if `log="symlog"`, Number of bins for linear part of bins (around zero)
    label_x:
        Label of x-axis
    label_y:
        Label of y-axis
    figure:
        Matplotlib figure
    axes:
        Matplotlib axes
    filename:
        Figure saved to file if provided
    font_size:
        Size of font
    vertical:
        Plot bars along vertical axis instead of default horizontal direction
    alpha:
        Alpha value of bars
    color:
        Color of bars
    horizontal_line:
        Plot horizontal at specified value (not shown if None)
    tight_layout:
        Adjusts padding around axis and removes unnecessary white space
    rotation_x_labels:
        Angle of rotation of x-axis labels
    rotation_y_labels:
        Angle of rotation of y-axis labels
    title:
        Title of figure

    Returns
    -------
    Axes of bar plot
    """
    _log = kego.lists.to_nlength_tuple(log, 2)
    _xlim = kego.lists.to_nlength_tuple(xlim, 2)
    _ylim = kego.lists.to_nlength_tuple(ylim, 2)

    values: np.ndarray
    if df is not None and isinstance(key_or_values, str):
        if label_x is None:
            label_x = key_or_values
        values = df[key_or_values].values
    elif isinstance(key_or_values, str):
        raise ValueError(
            f"Need to specify {df=} when specifying key name {key_or_values=}."
        )
    else:
        values = np.array(key_or_values)
    values = flatten_array(values)
    figure, axes = kego.plotting.utils_plotting.create_figure_axes(
        figure=figure, axes=axes
    )

    if bin_edges is None:
        bin_edges, symlog_linear_threshold = get_bin_edges(
            data=values,
            n_bins=n_bins,
            n_bins_linear=n_bins_linear,
            symlog_linear_threshold=symlog_linear_threshold,
            log=_log[0],
            return_symlog_linear_threshold=True,
            vmin=_xlim[0],
            vmax=_xlim[1],
        )
    (
        hist,
        bin_edges,
    ) = np.histogram(values, bins=bin_edges)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0

    _ = plot_bar(
        bin_centers,
        hist,
        width_bars=np.diff(bin_edges),
        xlim=_xlim,  # type: ignore
        ylim=_ylim,  # type: ignore
        axes=axes,
        log=_log,  # type: ignore
        symlog_linear_threshold=symlog_linear_threshold,
        label_x=label_x,
        label_y=label_y,
        vertical=vertical,
        alpha=alpha,
        color=color,
        rotation_x_labels=rotation_x_labels,
        rotation_y_labels=rotation_y_labels,
        font_size=font_size,
    )
    if horizontal_line is not None:
        axes.axhline(horizontal_line)

    if tight_layout:
        figure.tight_layout()
    kego.plotting.utils_plotting.set_title(axes, title, font_size=font_size)
    kego.plotting.figures.save_figure(figure, filename)
    return axes


def plot_bar(
    bin_centers: np.ndarray,
    height_bars: np.ndarray,
    width_bars: np.ndarray,
    axes: kego.constants.TYPE_MATPLOTLIB_AXES,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    log: str | tuple[str, str] = "false",
    symlog_linear_threshold: None | float = None,
    label_x: None | str = None,
    label_y: None | str = None,
    vertical: bool = False,
    alpha: float = 1,
    color: str | None = None,
    font_size: float = kego.constants.DEFAULT_FONTSIZE_LARGE,
    replace_x_labels_at: Sequence | None = None,
    replace_x_labels_with: Sequence | None = None,
    replace_y_labels_at: Sequence | None = None,
    replace_y_labels_with: Sequence | None = None,
    rotation_x_labels: int = 0,
    rotation_y_labels: int = 0,
) -> matplotlib.container.BarContainer:
    """
    plots 1d bar plot

    Parameters:
    ----------
    bin_centers:
        Center position of bins, array(N)
    height_bars:
        Bar heights at corresponding `bin_centers`, array(N)
    width_bars:
        Bar widths, array(N)
    axes:
        Matplotlib axes
    xlim:
        Limits of x-axis and bin_edges, if None determined from (min, max) of x-values
    ylim:
        Limits of y-axis and bin_edges, if None determined from (min, max) of y-values
    log:
        Type of bins ("false", "symlog", "log")
    symlog_linear_threshold:
        Required when `log="symlog"`.
        Threshold value below which symlog becomes linear.
    label_x:
        Label of x-axis
    label_y:
        Label of y-axis
    vertical:
        Plot bars along vertical axis instead of default horizontal direction
    alpha:
        Alpha value of bars
    color:
        Color of bars
    font_size:
        Size of font
    replace_x_labels_at:
        Tick values of x-axis to be replaced
    replace_x_labels_with:
        Replacement values for tick values of x-axis
    replace_y_labels_at:
        Tick values of y-axis to be replaced
    replace_y_labels_with:
        Replacement values for tick values of y-axis
    rotation_x_labels:
        Angle of rotation of x-axis labels
    rotation_y_labels:
        Angle of rotation of y-axis labels

    Returns
    -------
    Bar plot object
    """
    _log = kego.lists.to_nlength_tuple(log)
    _xlim = kego.lists.to_nlength_tuple(xlim)
    _ylim = kego.lists.to_nlength_tuple(ylim)

    if vertical:
        plot = axes.barh(
            bin_centers,
            height_bars,
            height=width_bars,
            edgecolor="black",
            alpha=alpha,
            color=color,
        )
    else:
        plot = axes.bar(
            bin_centers,
            height_bars,
            width=width_bars,
            edgecolor="black",
            alpha=alpha,
            color=color,
        )

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
    return plot


def flatten_array(x):
    x = np.asarray(x)
    x = np.ndarray.flatten(x)
    return x


def get_bin_edges(
    vmin: float | None = None,
    vmax: float | None = None,
    data: np.ndarray | None = None,
    n_bins: int = 60,
    n_bins_linear: int = 10,
    log: str = "false",
    symlog_linear_threshold: float | None = None,
    return_symlog_linear_threshold: bool = False,
) -> tuple[np.ndarray, float | None] | np.ndarray:
    """
    Computes bin edges for plots

    Log types included `"false"`, `"symlog"`, `"log"`
    Parameters:
    ----------
    vmin:
        Minimum value of bin edge
    vmax:
        Maximum value of bin edge
    data:
        Used to compute `vmin`/`vmax` if None
    n_bins:
        Number of bins
    n_bins_linear:
        Only relevant if `log="symlog"`, Number of bins for linear part of bins (around zero)
    log:
        Type of bins ("false", "symlog", "log")
    symlog_linear_threshold:
        Required when `log="symlog"`.
        Threshold value below which symlog becomes linear.
    return_symlog_linear_threshold:
        Return symlog_linear_threshold in addition to bin edges

    Returns
    -------
    Bin edges, Optionally(symlog_linear_threshold)
    """
    if data is not None and vmin is None:
        vmin = data.min()
    if data is not None and vmax is None:
        vmax = data.max()
    if vmin is None or vmax is None:
        raise ValueError(
            f"Need to specify vmin {vmin} and {vmax} or provide data: {data}!"
        )
    if vmin > vmax:
        raise ValueError(f"{vmin=} > {vmax=}")
    if vmin <= 0 and log == "log":
        raise ValueError(f"For {log=}, cannot have {vmin=} <= 0")
    if vmin == vmax:
        vmin -= vmax / 2
        vmax += vmax / 2
        n_bins = 2
    if log == "symlog":
        if symlog_linear_threshold is None:
            abs_max = abs(vmax)
            abs_min = abs(vmin)
            symlog_linear_threshold = (
                abs_min if abs_min < abs_max else abs_max if abs_max != 0 else abs_min
            )
            if symlog_linear_threshold == 0:
                raise ValueError(
                    f"{symlog_linear_threshold=}, which is not allowed. Please set manually to different value!"
                )
            logger.info(
                f"Setting: linear_thresh: {symlog_linear_threshold} with vmin: {vmin}"
                " and vmax: {vmax}!"
            )

        bins = _get_bin_edges_symlog(
            vmin,
            vmax,
            symlog_linear_threshold,
            n_bins=n_bins,
            n_bins_linear=n_bins_linear,
        )
    elif log == "log":
        bins = 10 ** np.linspace(np.log10(vmin), np.log10(vmax), n_bins)
    else:
        bins = np.linspace(vmin, vmax, n_bins)

    if return_symlog_linear_threshold:
        return bins, symlog_linear_threshold
    else:
        return bins


def _get_bin_edges_symlog(
    vmin: float,
    vmax: float,
    linear_thresh: float,
    n_bins: int = 60,
    n_bins_linear: int = 10,
) -> np.ndarray:
    """
    Computes symmetrical logarithmic bins

    Bins have same absolute vmin/vmax if vmin is negative
    Parameters:
    ----------
    vmin:
        Minimum value of bin edge
    vmax:
        Maximum value of bin edge
    linear_thresh:
        Threshold value below which symlog becomes linear.
    n_bins:
        Number of bins for logarithmic part of bins
    n_bins_linear:
        Number of bins for linear part of bins (around zero)

    Returns
    -------
    symmetrical bin edges
    """
    if isinstance(vmin, np.datetime64) or vmin > 0:
        bins = 10 ** np.linspace(np.log10(vmin), np.log10(vmax), n_bins)
    elif vmin == 0:
        bins = np.hstack(
            (
                np.linspace(0, linear_thresh, n_bins_linear),
                10 ** np.linspace(np.log10(linear_thresh), np.log10(vmax), n_bins),
            )
        )
    else:
        bins = np.hstack(
            (
                -(
                    10
                    ** np.linspace(
                        np.log10(vmax),
                        np.log10(linear_thresh),
                        n_bins // 2,
                        endpoint=False,
                    )
                ),
                np.linspace(
                    -linear_thresh, linear_thresh, n_bins_linear, endpoint=False
                ),
                10 ** np.linspace(np.log10(linear_thresh), np.log10(vmax), n_bins // 2),
            )
        )
    return bins


def _find_axis_limits(
    xlim: tuple[float | None, float | None],
    ylim: tuple[float | None, float | None],
    bin_edges_x: np.ndarray,
    bin_edges_y: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if xlim[0] is None:
        xlim = (bin_edges_x[0], xlim[1])
    if xlim[1] is None:
        xlim = (xlim[0], bin_edges_x[-1])
    if ylim[0] is None:
        ylim = (bin_edges_y[0], ylim[1])
    if ylim[1] is None:
        ylim = (ylim[0], bin_edges_y[-1])
    return xlim, ylim  # type: ignore
