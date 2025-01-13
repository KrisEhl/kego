from typing import Literal

import numpy as np
import pandas as pd

import kego.checks
import kego.constants


def _resolve_defaults(
    facet_column: str | None,
    facet_row: str | None,
    marginal_x: str | None,
    marginal_y: str | None,
    spacing_x: float | None,
    spacing_y: float | None,
    axes_style: kego.constants.AXES_STYLE_TYPE,
) -> tuple[bool, None | str, None | str, float, float]:
    if axes_style == "axes_single" and (
        facet_column is not None or facet_row is not None
    ):
        raise ValueError(
            f"Cannot specify {facet_column=} or {facet_row=} when axes is given!"
        )
    if marginal_x == "default":
        if facet_column is None and facet_row is None and axes_style == "axes_grid":
            marginal_x = "histogram"
        else:
            marginal_x = None
    if marginal_y == "default":
        if facet_column is None and facet_row is None and axes_style == "axes_grid":
            marginal_y = "histogram"
        else:
            marginal_y = None
    no_marginal_plots = marginal_x is None and marginal_y is None
    if facet_column is not None and spacing_x is None:
        spacing_x = 0.1
    else:
        spacing_x = 0.03
    if facet_row is not None and spacing_y is None:
        spacing_y = 0.08
    else:
        spacing_y = 0.03
    return no_marginal_plots, marginal_x, marginal_y, spacing_x, spacing_y


def _resolve_axes_style(
    axes: kego.constants.TYPE_MATPLOTLIB_AXES | None,
) -> kego.constants.AXES_STYLE_TYPE:
    axes_style: Literal["axes_single", "axes_grid"]
    if axes is not None:
        axes_style = "axes_single"
    else:
        axes_style = "axes_grid"
    return axes_style


def _check_parameter_consistency(
    facet_column: str | None,
    facet_row: str | None,
    marginal_x: str | None,
    marginal_y: str | None,
    df: pd.DataFrame | None,
):
    if (bool(marginal_x) + bool(marginal_y)) not in [0, 2]:
        raise NotImplementedError(
            f"{bool(marginal_x)=} anbool(d) {marginal_y=} both have to be on/off"
        )
    if (bool(marginal_x) or bool(marginal_y)) and (facet_column or facet_row):
        raise NotImplementedError(
            f"Cannot use ({marginal_x=} or {marginal_y}) at the same time as ({facet_column=} or {facet_row})"
        )
    if (facet_column or facet_row) and df is None:
        raise NotImplementedError(
            f"Cannnot use {facet_column=} and/or {facet_row}, when {df=} is not given!"
        )


def _prepare_facet(
    df: pd.DataFrame | None, facet: str | None
) -> tuple[np.ndarray | None, int, np.ndarray | None]:
    if facet is not None and isinstance(df, pd.DataFrame) and facet in df:
        facet_values = np.unique(df[facet].values)
        masks = np.array([df[facet] == val for val in facet_values])
        n_facet_values = len(facet_values)
        facet_title = np.array([f"{facet} == {val}" for val in facet_values])
    else:
        facet_values, masks, n_facet_values, facet_title = None, None, 0, None
    return masks, n_facet_values, facet_title


def _set_number_rows_and_columns(
    marginal_x: str | None,
    marginal_y: str | None,
    n_facet_column_values: int | None,
    n_facet_row_values: int | None,
) -> tuple[int, int]:
    n_columns = 1
    if n_facet_column_values:
        n_columns = n_facet_column_values
    elif marginal_x is not None:
        n_columns = 2

    n_rows = 1
    if n_facet_row_values:
        n_rows = n_facet_row_values
    elif marginal_y is not None:
        n_rows = 2
    return n_columns, n_rows


def _set_axis_labels(
    n_columns: int,
    n_rows: int,
    facet_show_label_x: list[int] | str = "minimal",
    facet_show_label_y: list[int] | str = "minimal",
    label_x: str | None = None,
    label_y: str | None = None,
    facet_column: str | None = None,
    facet_row: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if facet_column is not None:
        i_last_row = n_rows - 1
        xaxis_labels = np.full((n_rows, n_columns), "off", dtype=object)
        for i_row in range(n_rows):
            for i_column in range(n_columns):
                if isinstance(facet_show_label_x, str):
                    if facet_show_label_x == "all":
                        xaxis_labels[i_row, i_column] = label_x
                    elif facet_show_label_x == "minimal":
                        if i_row == i_last_row:
                            xaxis_labels[i_row, i_column] = label_x
                else:
                    kego.checks.assert_all_same_type(facet_show_label_x, int)
                    if i_row in facet_show_label_x:
                        xaxis_labels[i_row, i_column] = label_x
    else:
        xaxis_labels = np.full((n_rows, n_columns), label_x, dtype=object)
    if facet_row is not None:
        i_first_colum = 0
        yaxis_labels = np.full((n_rows, n_columns), "off", dtype=object)
        for i_row in range(n_rows):
            for i_column in range(n_columns):
                if isinstance(facet_show_label_y, str):
                    if facet_show_label_y == "all":
                        yaxis_labels[i_row, i_column] = label_y
                    elif facet_show_label_y == "minimal":
                        if i_column == i_first_colum:
                            yaxis_labels[i_row, i_column] = label_y
                else:
                    kego.checks.assert_all_same_type(facet_show_label_y, int)
                    if i_column in facet_show_label_y:
                        yaxis_labels[i_row, i_column] = label_y
    else:
        yaxis_labels = np.full((n_rows, n_columns), label_y, dtype=object)
    return xaxis_labels, yaxis_labels


def _set_masks(
    column_masks: np.ndarray | None,
    row_masks: np.ndarray | None,
    n_columns: int,
    n_rows: int,
) -> np.ndarray:
    masks = np.full((n_rows, n_columns), None, dtype=object)
    for i_row in range(n_rows):
        for i_column in range(n_columns):
            if column_masks is not None:
                masks[i_row, i_column] = _add_masks(
                    masks[i_row, i_column], column_masks[i_column]
                )
            if row_masks is not None:
                masks[i_row, i_column] = _add_masks(
                    masks[i_row, i_column], row_masks[i_row]
                )
    return masks


def _determine_axes_titles(
    facet_column: str | None,
    facet_row: str | None,
    title: str,
    facet_column_title: np.ndarray | None,
    facet_row_title: np.ndarray | None,
    n_columns: int,
    n_rows: int,
) -> np.ndarray:
    titles = np.full((n_rows, n_columns), title, dtype=object)
    for i_row in range(n_rows):
        for i_column in range(n_columns):
            if facet_row and facet_row_title is not None:
                titles[i_row, i_column] = _add_titles(
                    [titles[i_row, i_column], facet_row_title[i_row]]
                )
            if facet_column and facet_column_title is not None:
                titles[i_row, i_column] = _add_titles(
                    [titles[i_row, i_column], facet_column_title[i_column]]
                )
    return titles


def _prepare_axes_grid_creation(
    marginal_x: str | None, marginal_y: str | None
) -> tuple[
    list[tuple[int, int]] | None,
    list[float] | None,
    list[float] | None,
    list[tuple[int, int]] | None,
    bool,
]:
    skip_row_col: list[tuple[int, int]] | None = None
    widths_along_x: list[float] | None = None
    heights_along_y: list[float] | None = None
    colorbar_include_row_col: list[tuple[int, int]] | None = None

    colorbar_off = False
    if marginal_x is not None and marginal_y is not None:
        skip_row_col = [
            (0, 1),
        ]
        widths_along_x = [0.2, 0.1]
        heights_along_y = [0.1, 0.2]
        colorbar_include_row_col = [
            (1, 1),
        ]
    return (
        skip_row_col,
        widths_along_x,
        heights_along_y,
        colorbar_include_row_col,
        colorbar_off,
    )


def _add_titles(titles: list, delimeter: str = ", "):
    return f"{delimeter}".join([t for t in titles if len(t)])


def _add_masks(mask: np.ndarray | None, to_add: np.ndarray | None) -> np.ndarray | None:
    if to_add is None:
        return mask
    if mask is None:
        return to_add
    else:
        return mask & to_add
