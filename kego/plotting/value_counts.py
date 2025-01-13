import numpy as np
import pandas as pd

import kego.plotting


def plot_value_counts(
    df: pd.DataFrame,
    rotation_x_labels: int = 90,
    rotation_y_labels: int = 0,
    font_size: int = 12,
    figure_size: tuple | None = None,
):
    for column in df.columns:
        figure, axes, _ = kego.plotting.utils_plotting.create_axes_grid(
            n_columns=1, n_rows=1, unravel=True, figure_size=figure_size
        )
        value_counts = df[column].value_counts()
        values = value_counts.index
        counts = value_counts.values
        bin_edges = np.arange(len(values) + 1)
        width_bars = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + width_bars

        replace_x_labels_at = None
        replace_x_labels_with = None
        if len(values) < 100:
            replace_x_labels_at = bin_centers
            replace_x_labels_with = values

            kego.plotting.histograms.plot_bar(
                bin_centers=bin_centers,
                height_bars=counts,
                width_bars=width_bars,
                replace_x_labels_at=replace_x_labels_at,
                replace_x_labels_with=replace_x_labels_with,
                rotation_x_labels=rotation_x_labels,
                rotation_y_labels=rotation_y_labels,
                axes=axes,
                font_size=font_size,
            )
        else:
            kego.plotting.histograms.plot_histogram(
                counts,
                axes=axes,
                font_size=font_size,
            )

    return figure
