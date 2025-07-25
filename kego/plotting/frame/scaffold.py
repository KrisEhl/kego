import logging
from dataclasses import dataclass
from typing import Literal

from ...checks import all_same_type
from ...lists import flatten_list
from .config_plot import ConfigPlot

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
    """Convert tuple index, e.g. (1,2), to wrapped index, i.e. 1 + 2 * 3 (assuming nx=2, ny=3)"""
    x, y = xy
    return x + y * ny


def _from_i_to_xy(i: int, nx: int, ny: int):
    """Convert tuple index, e.g. (1,2), to wrapped index, i.e. 1 + 2 * 3 (assuming nx=2, ny=3)"""
    return (i % ny,)


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

    def transpose(self):
        return Grid(grid=[list(row) for row in zip(*self.grid)])

    def __str__(self) -> str:
        return (
            f"<Scaffold {self.nx},{self.ny}>\n[\n "
            + "\n ".join([str(x).center(12) for x in self.grid])
            + "\n]"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def nx(self):
        return len(self.grid[0])

    @property
    def ny(self):
        return len(self.grid)

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

    def __setitem__(self, xy: tuple[int, int], value) -> None:
        x, y = xy
        self.extend_grid(x=x + 1, y=y + 1)
        self.grid[y][x] = value

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

    @property
    def empty_entries_i(self):
        return [i for i in range(len(self.flatten())) if self[i] is None]


@dataclass
class Scaffold:
    entries: Grid

    def __str__(self) -> str:
        return (
            f"<Scaffold {self.nx},{self.ny}>\n[\n "
            + "\n ".join([str(x).center(12) for x in self.entries])
            + "\n]"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def nx(self):
        return self.entries.nx

    @property
    def ny(self):
        return self.entries.ny

    @property
    def empty_entries(self):
        return self.entries.empty_entries_i

    @property
    def next_empty_entry(self):
        empty_entries = self.empty_entries
        if not len(empty_entries):
            self.entries.extend_grid(x=self.ny + 1)
        return empty_entries[0]

    def set(self, confif_plot: ConfigPlot, x: int | None = None, y: int | None = None):
        if x is None and y is None:
            self.set_in_next_empty(config_plot=confif_plot)
        elif x is not None and y is not None:
            self.set_in_specific(x=x, y=y, config_plot=confif_plot)

    def set_in_next_empty(self, config_plot: ConfigPlot):
        # NOTE: should these entries be treated differently than "set specific" and be moved when specific requires their spot?
        self.next_empty_entry

    def set_in_specific(self, x: int, y: int, config_plot: ConfigPlot):
        # self.extend_grid(x=x + 1, y=y + 1)
        entry_current = self.entries[y][x]
        if entry_current is not None:
            logger.debug(
                f"Location {x=} and {y} already taken ({entry_current})! Will be overwritten by {config_plot}."
            )
        self.entries[y][x] = config_plot
        return self

    @classmethod
    def from_nx_ny(cls, nx, ny):
        return cls(scaffold_entries=[[None for _ in range(nx)] for _ in range(ny)])
