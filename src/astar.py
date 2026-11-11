"""Reusable A* search for rectangular four-neighbour grids."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from time import perf_counter
from typing import Any, Callable, Sequence

Coordinate = tuple[int, int]
Grid = Sequence[Sequence[Any]]
Heuristic = Callable[[Coordinate, Coordinate], float]
CostFunction = Callable[[Coordinate, Coordinate, Any], float]
BlockedFunction = Callable[[Coordinate, Any], bool]


@dataclass
class AStarStats:
    """Instrumentation collected during one A* run."""

    expanded_nodes: int = 0
    generated_nodes: int = 0
    stale_pops: int = 0
    reopened_nodes: int = 0
    neighbours_considered: int = 0
    max_frontier: int = 0
    runtime_ms: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass
class AStarResult:
    """Path, optimal cost and metrics returned by A*."""

    path: list[Coordinate]
    cost: float
    stats: AStarStats

    @property
    def found(self) -> bool:
        return bool(self.path)


def zero_heuristic(_: Coordinate, __: Coordinate) -> float:
    return 0.0


def manhattan(node: Coordinate, goal: Coordinate) -> float:
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def free_goal_manhattan(
    node: Coordinate, goal: Coordinate, minimum_step_cost: float = 1.0
) -> float:
    """Admissible lower bound when the final move into the goal is free."""

    return max(manhattan(node, goal) - 1.0, 0.0) * minimum_step_cost


def _validate_grid(grid: Grid) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    columns = len(grid[0])
    if any(len(row) != columns for row in grid):
        raise ValueError("grid must be rectangular")
    return len(grid), columns


def _reconstruct_path(
    came_from: dict[Coordinate, Coordinate], goal: Coordinate
) -> list[Coordinate]:
    path = [goal]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def astar_grid(
    start: Coordinate,
    goal: Coordinate,
    grid: Grid,
    *,
    heuristic: Heuristic,
    cost_to_enter: CostFunction,
    is_blocked: BlockedFunction,
    neighbour_order: Sequence[Coordinate] = ((-1, 0), (0, 1), (1, 0), (0, -1)),
) -> AStarResult:
    """Run A* on a rectangular grid.

    The heap may contain stale entries. They are skipped using ``g_score`` so a
    better route discovered later remains correct even with an inconsistent
    heuristic. ``generated_nodes`` counts successful heap insertions, while
    ``expanded_nodes`` counts valid heap removals.
    """

    started = perf_counter()
    rows, columns = _validate_grid(grid)
    for name, node in (("start", start), ("goal", goal)):
        if not (0 <= node[0] < rows and 0 <= node[1] < columns):
            raise ValueError(f"{name} is outside the grid: {node}")
        if is_blocked(node, grid[node[0]][node[1]]):
            raise ValueError(f"{name} is blocked: {node}")

    stats = AStarStats(generated_nodes=1, max_frontier=1)
    sequence = count()
    start_h = float(heuristic(start, goal))
    frontier: list[tuple[float, float, int, float, Coordinate]] = [
        (start_h, start_h, next(sequence), 0.0, start)
    ]
    g_score: dict[Coordinate, float] = {start: 0.0}
    expanded_at_cost: dict[Coordinate, float] = {}
    came_from: dict[Coordinate, Coordinate] = {}

    while frontier:
        _, _, _, current_g, current = heappop(frontier)
        if current_g != g_score.get(current, inf):
            stats.stale_pops += 1
            continue

        stats.expanded_nodes += 1
        expanded_at_cost[current] = current_g
        if current == goal:
            stats.runtime_ms = (perf_counter() - started) * 1000.0
            return AStarResult(
                path=_reconstruct_path(came_from, goal),
                cost=current_g,
                stats=stats,
            )

        for row_delta, column_delta in neighbour_order:
            stats.neighbours_considered += 1
            neighbour = (current[0] + row_delta, current[1] + column_delta)
            row, column = neighbour
            if not (0 <= row < rows and 0 <= column < columns):
                continue
            cell = grid[row][column]
            if is_blocked(neighbour, cell):
                continue

            step_cost = float(cost_to_enter(current, neighbour, cell))
            if step_cost < 0:
                raise ValueError("A* requires non-negative edge costs")
            tentative_g = current_g + step_cost
            if tentative_g >= g_score.get(neighbour, inf):
                continue

            if neighbour in expanded_at_cost and tentative_g < expanded_at_cost[neighbour]:
                stats.reopened_nodes += 1
            came_from[neighbour] = current
            g_score[neighbour] = tentative_g
            neighbour_h = float(heuristic(neighbour, goal))
            heappush(
                frontier,
                (
                    tentative_g + neighbour_h,
                    neighbour_h,
                    next(sequence),
                    tentative_g,
                    neighbour,
                ),
            )
            stats.generated_nodes += 1
            stats.max_frontier = max(stats.max_frontier, len(frontier))

    stats.runtime_ms = (perf_counter() - started) * 1000.0
    return AStarResult(path=[], cost=inf, stats=stats)

