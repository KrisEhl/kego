import pathlib
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

import kego.plotting.axes_utils
import kego.plotting.figures
import kego.plotting.utils_plotting

try:
    import sklearn.metrics
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{e}\n"
        f"Optional dependencies missing, install via"
        " `poetry install --extras=train`"
    )


def plot_prediction_certainty(
    truth: np.ndarray,
    prediction_probabilities: np.ndarray,
    n_true_labels: int = 2,
    n_bins: int = 10,
    label_x: str = "True label",
    label_y: str = "Prediction probability for 'raining'",
    label_colorbar: str = "Number of Tweets",
    filename: str | pathlib.Path | None = None,
    font_size: int = 12,
    vmin=None,
    vmax=None,
    cmap="viridis",
    figure_size: tuple[float, float] | None = None,
    return_matrix: bool = False,
    axes: Optional[plt.axes] = None,
    **kwargs,
) -> tuple[plt.Axes, tuple[plt.Axes, np.ndarray]]:
    """
    Compute 2d histogram of true labels (x-axis) and
    binned prediction probabilities (y-axis)

    Parameters:
    ----------
    truth: true labels
    prediction_probabilities: probabilities for labels
    n_true_labels: number of different labels
    n_bins: number of bins for prediction probabilities
    label_x: label of x-axis
    label_y: label of y-axis
    label_colorbar: label of colorbar axis
    filename: save figure to this file, not saved if `None`
    font_size: font size of all labels of plot
    vmin: Minimum value for colorbar
    vmax: Maximum value for colorbar
    cmap: Colormap used for confusion matrix
    figure_size: size of figure [width, height]
    return_matrix: return 2d histogram matrix and axes
    axes: Matplotlib axes object

    Returns
    -------
    axes of plot
    """
    results_dic = kego.plotting.histograms.plot_histogram_2d(
        truth,
        prediction_probabilities,
        n_bins=(n_true_labels, n_bins),
        label_x=label_x,
        label_colorbar=label_colorbar,
        label_y=label_y,
        annotate=True,
        filename=filename,
        vmin=vmin,
        vmax=vmax,
        colormap=cmap,
        font_size=font_size,
        axes=axes,
        figure_size=figure_size,
        **kwargs,
    )
    if return_matrix:
        return results_dic["axes_2d_histograms_10"], results_dic["2d_histograms_10"]
    return results_dic["axes_2d_histograms_10"]


def classification_report(
    truth: Sequence,
    prediction: Sequence,
    output_dict: bool = True,
    label: str = "raining",
) -> tuple[str, Mapping]:
    """
    Compute classification report, returns precision, recall, f1-score
    and respective averages

    report returned as dict or string (see `output_dict`)
    Parameters:
    ----------
    truth: true labels
    prediction: predicted labels
    output_dict: return report as dictionary rather than string
    label: Label name of the classification

    Returns
    -------
    report
    """
    target_names = [f"not {label}", label]
    report = sklearn.metrics.classification_report(
        truth,
        prediction,
        target_names=target_names,
        output_dict=output_dict,
    )
    return report


def check_prediction(
    truth: Sequence,
    prediction: Sequence,
    output_dict: bool = True,
    filename: str | pathlib.Path | None = None,
    label: str = "raining",
    font_size: int = 14,
):
    plot_confusion_matrix(
        truth,
        prediction,
        filename=filename,
        overplot_round_base=2,
        tick_labels_x=(f"not {label}", f"{label}"),
        tick_labels_y=(f"{label}", f"not {label}"),
        font_size=font_size,
    )

    report = classification_report(
        truth,
        prediction,
        label=label,
        output_dict=output_dict,
    )
    return report


def plot_confusion_matrix(
    truth: Sequence,
    prediction: Sequence,
    filename: str | pathlib.Path | None = None,
    normalize: str = "all",
    font_size: int = 14,
    figure_size: tuple[float, float] | None = None,
    label_x: str | None = "Predicted label",
    label_y: str | None = "True label",
    vmin: float | None = None,
    vmax: float | None = None,
    text_color: str = "firebrick",
    colormap: str = "Blues",
    overplot_round_base: int | None = 2,
    tick_labels_x: tuple | None = ("not raining", "raining"),
    tick_labels_y: tuple | None = ("raining", "not raining"),
    colorbar_label: str | None = "Fraction",
    colorbar_width: float = 0.05,
    axes: plt.Axes | None = None,
    ax_colorbar: object | None = None,
):
    """
    Computes and plots confusion matrix

    Parameters:
    ----------
    truth: True labels
    prediction: Predicted labels
    filename: Save figure to this file, not saved if `None`
    normalize: Normalize style of confusion matrix
    output_dict: Return classification report as dictionary
    font_size: Font size of all labels of plot
    figure_size: Size of figure [width, height], default [6, 6]

    Returns
    -------
    Confusion matrix figure
    """
    if figure_size is None:
        figure_size = (6, 6)
    cm = sklearn.metrics.confusion_matrix(truth, prediction, normalize=normalize)
    if axes is None:
        fig, axes, ax_colorbar = kego.plotting.utils_plotting.create_axes_grid(
            1,
            1,
            figure_size=figure_size,
            unravel=True,
            colorbar_off=False,
            colorbar_width=colorbar_width,
        )
    else:
        fig = axes.get_figure()
    norm = kego.plotting.utils_plotting.get_norm("linear", vmin=vmin, vmax=vmax)
    xedges = [0, 0.5, 1]
    yedges = [0, 0.5, 1]
    X, Y = np.meshgrid(xedges, yedges)
    mesh = kego.plotting.colormesh._plot_colormesh(
        axes,
        X,
        Y,
        cm[::-1],
        norm,
        colormap,
        linewidth=0,
        rasterized=True,
    )
    kego.plotting.utils_plotting.plot_colorbar(
        mesh, cax=ax_colorbar, label=colorbar_label, font_size=font_size
    )
    kego.plotting.utils_plotting.annotate_values(
        np.array(cm),
        axes,
        2,
        2,
        color=text_color,
        round_to_base=overplot_round_base,
        font_size=font_size,
    )
    kego.plotting.utils_plotting.set_axis_tick_labels(
        axes=axes, values=[0.25, 0.75], labels=tick_labels_x, axis="x"
    )
    kego.plotting.utils_plotting.set_axis_tick_labels(
        axes=axes, values=[0.25, 0.75], labels=tick_labels_y, axis="y"
    )
    kego.plotting.axes_utils.set_axes(
        axes,
        xlim=[0, 1],
        ylim=[0, 1],
        fontsize=font_size,
        label_x=label_x,
        label_y=label_y,
    )

    kego.plotting.figures.save_figure(fig, filename)
    return fig


def plot_roc(
    truth: Sequence,
    prediction_probabilities: Sequence,
    filename: str | pathlib.Path | None = None,
    font_size: int = 12,
    figure_size: Optional[Sequence] = None,
    axes: plt.Axes | None = None,
    fig: object | None = None,
    label_x: str | None = "False Positive Rate",
    label_y: str | None = "True Positive Rate",
    return_rates: bool = False,
) -> tuple[plt.Axes] | tuple[plt.Axes, Sequence, Sequence]:
    """
    Plots Receiver Operating Characteristic curve

    Can optionally return true positive and false positive rate
    (if `return_rates=True`)
    Parameters:
    ----------
    truth: true labels
    prediction_probabilities: probabilities for labels
    filename: save figure to this file, not saved if `None`
    font_size: font size of all labels of plot
    figure_size: size of figure [width, height], default: [6, 6]
    return_rates: return true positive and false positive rate

    Returns
    -------
    axes of plot
    """
    if figure_size is None:
        figure_size = [6, 6]
    false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(
        truth, prediction_probabilities, drop_intermediate=False
    )
    roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
    fig, axes = kego.plotting.utils_plotting.create_figure_axes(
        figure=fig, axes=axes, figure_size=figure_size, font_size=font_size
    )
    lw = 2
    axes.plot(
        false_positive_rate,
        true_positive_rate,
        lw=lw,
        label="ROC curve (area = %.04f)" % roc_auc,
    )

    axes.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    kego.plotting.axes_utils.set_axes(
        axes,
        xlim=[0, 1],
        ylim=[0, 1],
        fontsize=font_size,
        label_x=label_x,
        label_y=label_y,
    )

    axes.legend(loc="lower right")
    kego.plotting.figures.save_figure(fig, filename)
    if return_rates:
        return axes, true_positive_rate, false_positive_rate
    return axes
