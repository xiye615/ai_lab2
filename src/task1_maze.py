"""Task 1: shortest path in an unweighted maze."""

from __future__ import annotations

from typing import Sequence

from .astar import AStarResult, Coordinate, astar_grid, manhattan, zero_heuristic


SAMPLE_START: Coordinate = (0, 0)
SAMPLE_GOAL: Coordinate = (3, 4)
SAMPLE_GRID: list[list[str]] = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]


def solve_maze(
    start: Coordinate,
    goal: Coordinate,
    grid: Sequence[Sequence[str]],
    *,
    heuristic_name: str = "manhattan",
) -> AStarResult:
    heuristics = {"zero": zero_heuristic, "manhattan": manhattan}
    if heuristic_name not in heuristics:
        raise ValueError(f"unknown heuristic: {heuristic_name}")
    return astar_grid(
        start,
        goal,
        grid,
        heuristic=heuristics[heuristic_name],
        cost_to_enter=lambda _current, _neighbour, _cell: 1.0,
        is_blocked=lambda _coordinate, cell: cell == "X",
    )


def maze_path(start: Coordinate, goal: Coordinate, grid: Sequence[Sequence[str]]) -> list[Coordinate]:
    """Assignment-facing function returning only the required coordinate list."""

    return solve_maze(start, goal, grid).path

