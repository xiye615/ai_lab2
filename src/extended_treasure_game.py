"""Reusable larger treasure-allocation scenario for extension experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .game_search import DecisionResult, OrderActions, alpha_beta_decision


@dataclass(frozen=True)
class AllocationProblem:
    values: dict[str, int]
    distances: dict[str, dict[str, int]]
    treasure_order: tuple[str, ...]
    start_a: str = "A"
    start_b: str = "B"


@dataclass(frozen=True)
class AllocationState:
    position_a: str
    position_b: str
    remaining: tuple[str, ...]
    score_a: int = 0
    score_b: int = 0
    current_player: str = "A"

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


def apply_action(
    state: AllocationState, target: str, problem: AllocationProblem
) -> AllocationState:
    remaining = tuple(item for item in state.remaining if item != target)
    if state.current_player == "A":
        gain = problem.values[target] - problem.distances[state.position_a][target]
        return AllocationState(
            target,
            state.position_b,
            remaining,
            state.score_a + gain,
            state.score_b,
            "B",
        )
    gain = problem.values[target] - problem.distances[state.position_b][target]
    return AllocationState(
        state.position_a,
        target,
        remaining,
        state.score_a,
        state.score_b + gain,
        "A",
    )


def _order_actions(
    mode: str, problem: AllocationProblem
) -> OrderActions[AllocationState, str]:
    if mode not in {"left_to_right", "best_first", "worst_first"}:
        raise ValueError(f"unknown action order: {mode}")

    def order(
        state: AllocationState, actions: Sequence[str], maximizing: bool
    ) -> Sequence[str]:
        if mode == "left_to_right":
            return actions
        scored = [
            (apply_action(state, action, problem).utility, action) for action in actions
        ]
        reverse = maximizing if mode == "best_first" else not maximizing
        return [action for _, action in sorted(scored, reverse=reverse)]

    return order


def solve_allocation(
    problem: AllocationProblem,
    depth: int,
    *,
    order: str = "left_to_right",
    use_cache: bool = False,
) -> DecisionResult[str]:
    initial = AllocationState(
        problem.start_a, problem.start_b, problem.treasure_order
    )
    return alpha_beta_decision(
        initial,
        depth,
        is_terminal=lambda state: not state.remaining,
        evaluate=lambda state: float(state.utility),
        legal_actions=lambda state: list(state.remaining),
        apply_action=lambda state, action: apply_action(state, action, problem),
        is_maximizing=lambda state: state.current_player == "A",
        order_actions=_order_actions(order, problem),
        use_transposition_table=use_cache,
        transposition_key=lambda state: (
            state.position_a,
            state.position_b,
            state.remaining,
            state.current_player,
        ),
        transposition_offset=lambda state: float(state.utility),
    )


def build_six_treasure_problem() -> AllocationProblem:
    """Create a deterministic 9x9 open-grid scenario with six treasures."""

    positions = {
        "A": (0, 0),
        "B": (8, 8),
        "T1": (0, 8),
        "T2": (2, 2),
        "T3": (4, 4),
        "T4": (8, 0),
        "T5": (2, 6),
        "T6": (6, 2),
    }
    labels = tuple(positions)
    distances = {
        source: {
            target: abs(positions[source][0] - positions[target][0])
            + abs(positions[source][1] - positions[target][1])
            for target in labels
        }
        for source in labels
    }
    return AllocationProblem(
        values={"T1": 9, "T2": 6, "T3": 7, "T4": 7, "T5": 8, "T6": 5},
        distances=distances,
        treasure_order=("T1", "T2", "T3", "T4", "T5", "T6"),
    )

