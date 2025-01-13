import json

import arckit
import drawsvg
import numpy as np
from IPython.core.display import SVG, display


def show(drawing: drawsvg.Drawing):
    display(SVG(drawing.as_svg()))


def show_grid(grid):
    for sub_grid in grid:
        print(sub_grid)


def get_solution(task: arckit.data.Task, train_solutions):
    solution = train_solutions[task.id]
    return task, solution


def show_task(task: arckit.data.Task, train_solutions):
    drawing = arckit.vis.draw_task(task, include_test=True)
    solution = train_solutions[task.id]
    show(drawing)
    show_grid(solution[0])
    show(arckit.vis.draw_grid(np.array(solution[0])))
    return drawing
