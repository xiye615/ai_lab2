"""Task 4: treasure allocation combining A* and adversarial search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .astar import AStarStats, Coordinate, astar_grid, manhattan
from .game_search import (
    DecisionResult,
    OrderActions,
    alpha_beta_decision,
    minimax_decision,
)

TREASURE_GRID: list[list[Any]] = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]

LOCATIONS: dict[str, Coordinate] = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}
TREASURE_VALUES: dict[str, int] = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
LOCATION_ORDER: tuple[str, ...] = ("A", "B", "T1", "T2", "T3", "T4")
TREASURE_ORDER: tuple[str, ...] = ("T1", "T2", "T3", "T4")

DistanceMatrix = dict[str, dict[str, int]]


@dataclass(frozen=True)
class TreasureState:
    position_a: str = "A"
    position_b: str = "B"
    remaining: tuple[str, ...] = TREASURE_ORDER
    score_a: int = 0
    score_b: int = 0
    current_player: str = "A"

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


def _treasure_cell_cost(_current: Coordinate, _neighbour: Coordinate, cell: Any) -> float:
    if cell in LOCATIONS:
        return 1.0
    return float(cell)


def compute_distance_matrix(
    grid: Sequence[Sequence[Any]] = TREASURE_GRID,
) -> tuple[DistanceMatrix, AStarStats]:
    """Compute all directed distances between A, B and the four treasures."""

    matrix: DistanceMatrix = {name: {} for name in LOCATION_ORDER}
    aggregate = AStarStats()
    for source in LOCATION_ORDER:
        for target in LOCATION_ORDER:
            if source == target:
                matrix[source][target] = 0
                continue
            result = astar_grid(
                LOCATIONS[source],
                LOCATIONS[target],
                grid,
                heuristic=manhattan,
                cost_to_enter=_treasure_cell_cost,
                is_blocked=lambda _coordinate, cell: cell == "X",
            )
            if not result.found:
                raise ValueError(f"no route between {source} and {target}")
            matrix[source][target] = int(result.cost)
            for field_name in (
                "expanded_nodes",
                "generated_nodes",
                "stale_pops",
                "reopened_nodes",
                "neighbours_considered",
            ):
                setattr(
                    aggregate,
                    field_name,
                    getattr(aggregate, field_name) + getattr(result.stats, field_name),
                )
            aggregate.max_frontier = max(
                aggregate.max_frontier, result.stats.max_frontier
            )
            aggregate.runtime_ms += result.stats.runtime_ms
    return matrix, aggregate


def apply_treasure_action(
    state: TreasureState, target: str, distances: DistanceMatrix
) -> TreasureState:
    if target not in state.remaining:
        raise ValueError(f"treasure is not available: {target}")
    remaining = tuple(item for item in state.remaining if item != target)
    if state.current_player == "A":
        gain = TREASURE_VALUES[target] - distances[state.position_a][target]
        return TreasureState(
            position_a=target,
            position_b=state.position_b,
            remaining=remaining,
            score_a=state.score_a + gain,
            score_b=state.score_b,
            current_player="B",
        )
    gain = TREASURE_VALUES[target] - distances[state.position_b][target]
    return TreasureState(
        position_a=state.position_a,
        position_b=target,
        remaining=remaining,
        score_a=state.score_a,
        score_b=state.score_b + gain,
        current_player="A",
    )


def _action_order(
    mode: str, distances: DistanceMatrix
) -> OrderActions[TreasureState, str]:
    if mode not in {"left_to_right", "best_first", "worst_first"}:
        raise ValueError(f"unknown action order: {mode}")

    def order(
        state: TreasureState, actions: Sequence[str], maximizing: bool
    ) -> Sequence[str]:
        if mode == "left_to_right":
            return actions
        scored = [
            (apply_treasure_action(state, action, distances).utility, action)
            for action in actions
        ]
        reverse = maximizing if mode == "best_first" else not maximizing
        return [action for _, action in sorted(scored, reverse=reverse)]

    return order


def solve_treasure_game(
    depth: int,
    *,
    algorithm: str = "alpha_beta",
    order: str = "left_to_right",
    distances: DistanceMatrix | None = None,
    use_cache: bool = False,
) -> DecisionResult[str]:
    if distances is None:
        distances, _ = compute_distance_matrix()
    callbacks = {
        "is_terminal": lambda state: not state.remaining,
        "evaluate": lambda state: float(state.utility),
        "legal_actions": lambda state: list(state.remaining),
        "apply_action": lambda state, action: apply_treasure_action(
            state, action, distances
        ),
        "is_maximizing": lambda state: state.current_player == "A",
        "order_actions": _action_order(order, distances),
    }
    initial = TreasureState()
    if algorithm == "minimax":
        return minimax_decision(initial, depth, **callbacks)
    if algorithm == "alpha_beta":
        return alpha_beta_decision(
            initial,
            depth,
            use_transposition_table=use_cache,
            transposition_key=lambda state: (
                state.position_a,
                state.position_b,
                state.remaining,
                state.current_player,
            ),
            transposition_offset=lambda state: float(state.utility),
            **callbacks,
        )
    raise ValueError(f"unknown game-search algorithm: {algorithm}")

