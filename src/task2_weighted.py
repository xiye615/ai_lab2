"""Task 2: minimum-cost path through weighted terrain."""

from __future__ import annotations

from functools import partial
from typing import Any, Sequence

from .astar import (
    AStarResult,
    Coordinate,
    astar_grid,
    free_goal_manhattan,
    manhattan,
    zero_heuristic,
)


SAMPLE_START: Coordinate = (0, 0)
SAMPLE_GOAL: Coordinate = (4, 4)
SAMPLE_GRID: list[list[Any]] = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]


def _minimum_numeric_cost(grid: Sequence[Sequence[Any]]) -> float:
    costs = [
        float(cell)
        for row in grid
        for cell in row
        if isinstance(cell, (int, float)) or (isinstance(cell, str) and cell.isdigit())
    ]
    return min(costs, default=1.0)


def solve_weighted_terrain(
    start: Coordinate,
    goal: Coordinate,
    grid: Sequence[Sequence[Any]],
    *,
    heuristic_name: str = "adjusted_manhattan",
) -> AStarResult:
    minimum_cost = _minimum_numeric_cost(grid)
    heuristics = {
        "zero": zero_heuristic,
        "manhattan": manhattan,
        "adjusted_manhattan": partial(
            free_goal_manhattan, minimum_step_cost=minimum_cost
        ),
    }
    if heuristic_name not in heuristics:
        raise ValueError(f"unknown heuristic: {heuristic_name}")

    def cost_to_enter(_current: Coordinate, neighbour: Coordinate, cell: Any) -> float:
        if neighbour == goal or cell == "E":
            return 0.0
        if cell == "S":
            return minimum_cost
        try:
            return float(cell)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid terrain cost at {neighbour}: {cell!r}") from error

    return astar_grid(
        start,
        goal,
        grid,
        heuristic=heuristics[heuristic_name],
        cost_to_enter=cost_to_enter,
        is_blocked=lambda _coordinate, cell: cell == "X",
    )


def minimum_terrain_cost(
    start: Coordinate, goal: Coordinate, grid: Sequence[Sequence[Any]]
) -> int:
    """Assignment-facing function returning only the required integer cost."""

    result = solve_weighted_terrain(start, goal, grid)
    return int(result.cost) if result.found else -1

