import logging
from dataclasses import dataclass
from typing import Literal

from ...checks import all_same_type
from ...lists import flatten_list
from .config_plot import ConfigPlotStyle

logger = logging.getLogger(__name__)


def _extend_grid(
    grid: list[list],
    new_x: int | None = None,
    new_y: int | None = None,
    fill_value=None,
):
    current_y = len(grid)
    if new_y is None:
        new_y = current_y
    current_x = len(grid[0])
    if new_x is None:
        new_x = current_x

    for row in grid:
        row.extend([fill_value] * (new_x - current_x))

    for _ in range(new_y - current_y):
        grid.append([fill_value] * new_x)

    return grid


def _from_xy_to_i(xy: tuple[int, int], ny: int) -> int:
    """Convert tuple index, (1,2), to wrapped index, 1 + 2 * 3 (assuming nx=2, ny=3)"""
    x, y = xy
    return x + y * ny


def _from_i_to_xy(i: int, nx: int) -> tuple[int, int]:
    """Convert wrapped index, 1 + 2 * 3, to tuple index, (1,2) (assuming nx=2, ny=3)"""
    y = i // nx
    x = i % nx
    return x, y


class Grid:
    def __init__(
        self,
        nx: int | None = None,
        ny: int | None = None,
        grid: None | list[list] = None,
    ) -> None:
        if nx is not None and ny is not None:
            self.grid = [[None for _ in range(nx)] for _ in range(ny)]
        elif grid is not None:
            if not isinstance(grid, list):
                raise TypeError(f"{grid=} should be of type list[list]!")
            if not all(isinstance(element, list) for element in grid):
                raise TypeError(f"{grid=} should be of type list[list]!")
            self.grid = grid

    @property
    def nx(self):
        return len(self.grid[0])

    @property
    def ny(self):
        return len(self.grid)

    @property
    def _empty_entries_i(self):
        return [i for i in range(len(self.flatten())) if self[i] is None]

    def __str__(self) -> str:
        return (
            f"<Scaffold {self.nx},{self.ny}>\n[\n "
            + "\n ".join([str(x).center(12) for x in self.grid])
            + "\n]"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, xy: tuple[int, int] | int):
        if isinstance(xy, tuple):
            x, y = xy
            if x + 1 > self.nx or y + 1 > self.ny:
                raise IndexError(
                    f"Grid index ({x},{y}) out of range of grid (nx={self.nx}, ny={self.ny})!"
                )
            return self.grid[y][x]
        elif isinstance(xy, int):
            return self.flatten()[xy]

    def __setitem__(self, xy: tuple[int, int] | int, value) -> None:
        if isinstance(xy, int):
            xy = _from_i_to_xy(i=xy, nx=self.nx)
        x, y = xy
        self.extend_grid(x=x + 1, y=y + 1)
        self.grid[y][x] = value

    def transpose(self):
        return Grid(grid=[list(row) for row in zip(*self.grid)])

    def flatten(self) -> list:
        return flatten_list(self.grid)

    def extend_grid(self, x: int | None = None, y: int | None = None):
        if x is not None and self.nx < x:
            self.grid = _extend_grid(grid=self.grid, new_x=x)
        if y is not None and self.ny < y:
            self.grid = _extend_grid(grid=self.grid, new_y=y)
        return self

    def __iter__(self):
        return iter(self.grid)


@dataclass
class Scaffold:
    entries: Grid

    @property
    def nx(self):
        return self.entries.nx

    @property
    def ny(self):
        return self.entries.ny

    @property
    def empty_entries(self):
        return self.entries._empty_entries_i

    @classmethod
    def from_nx_ny(cls, nx, ny):
        return cls(entries=Grid(nx=nx, ny=ny))

    def __str__(self) -> str:
        return str(self.entries)

    def __repr__(self) -> str:
        return self.__str__()

    def set(
        self, confif_plot: ConfigPlotStyle, x: int | None = None, y: int | None = None
    ):
        if x is None and y is None:
            self._set_in_next_empty(config_plot=confif_plot)
        elif x is not None and y is not None:
            self._set_in_specific(x=x, y=y, config_plot=confif_plot)
        else:
            raise ValueError(f"Need to specify [{x=} and {y=}] or [{confif_plot=}]")

    @property
    def _next_empty_entry(self):
        empty_entries = self.empty_entries
        if not len(empty_entries):
            self.entries.extend_grid(y=self.ny + 1)
        return self.empty_entries[0]

    def _set_in_next_empty(self, config_plot: ConfigPlotStyle):
        # NOTE: should these entries be treated differently than "set specific" and be moved when specific requires their spot?
        i = self._next_empty_entry
        self.entries[i] = config_plot
        return self

    def _set_in_specific(self, x: int, y: int, config_plot: ConfigPlotStyle):
        # NOTE: should entries always be extended?
        self.entries[x, y] = config_plot
        return self
