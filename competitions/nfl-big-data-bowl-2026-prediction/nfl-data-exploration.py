import os
import pathlib

import numpy as np
import polars as pl
from IPython.display import display

import kego.plotting

PATH_COMPETITION = pathlib.Path("../../data/nfl/nfl-big-data-bowl-2026-prediction")
PATH_TRAIN = PATH_COMPETITION / "train"
PATH_TEST = PATH_COMPETITION / "test.csv"
print(f"{os.listdir(PATH_COMPETITION)}")
print(f"{os.listdir(PATH_COMPETITION/"train")}")

filename_suffix = "_2023_w09.csv"
input = pl.read_csv(PATH_TRAIN / f"input{filename_suffix}")
output = pl.read_csv(PATH_TRAIN / f"output{filename_suffix}")
print(f"{input.columns}")
print(f"{output.columns}")
display(input)
display(output)

display(
    input.filter(
        (pl.col("game_id") == 2023110200)
        & (pl.col("play_id") == 108)
        & (pl.col("nfl_id") == 47899)
    )
)
display(
    output.filter(
        (pl.col("game_id") == 2023110200)
        & (pl.col("play_id") == 108)
        & (pl.col("nfl_id") == 47899)
    )
)


def plot_histogram_all(df):
    n_columns = len(df.columns)
    figure, axes, axes_colorbar = kego.plotting.create_axes_grid(
        n_columns=2,
        n_rows=int(np.ceil(n_columns / 2)),
        unravel=True,
        spacing_x=0.05,
        spacing_y=0.02,
    )
    for axes, column in zip(axes, df.columns):
        kego.plotting.plot_histogram(
            column, df=df, axes=axes, font_size=4, title=column
        )

    figure.savefig("histograms_all.pdf")


plot_histogram_all(input)
