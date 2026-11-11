"""Reusable Minimax and Alpha-Beta search plus the Task 3 tree wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import inf
from time import perf_counter
from typing import Callable, Generic, Hashable, Sequence, TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass
class GameSearchStats:
    nodes_visited: int = 0
    leaf_evaluations: int = 0
    cutoffs: int = 0
    pruned_branches: int = 0
    cache_hits: int = 0
    cache_stores: int = 0
    runtime_ms: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass
class DecisionResult(Generic[ActionT]):
    best_action: ActionT | None
    value: float
    stats: GameSearchStats


@dataclass
class _TranspositionEntry(Generic[ActionT]):
    value: float
    flag: str
    best_action: ActionT | None


IsTerminal = Callable[[StateT], bool]
Evaluate = Callable[[StateT], float]
LegalActions = Callable[[StateT], Sequence[ActionT]]
ApplyAction = Callable[[StateT, ActionT], StateT]
IsMaximizing = Callable[[StateT], bool]
OrderActions = Callable[[StateT, Sequence[ActionT], bool], Sequence[ActionT]]


def _identity_order(
    _state: StateT, actions: Sequence[ActionT], _maximizing: bool
) -> Sequence[ActionT]:
    return actions


def minimax_decision(
    state: StateT,
    depth: int,
    *,
    is_terminal: IsTerminal[StateT],
    evaluate: Evaluate[StateT],
    legal_actions: LegalActions[StateT, ActionT],
    apply_action: ApplyAction[StateT, ActionT],
    is_maximizing: IsMaximizing[StateT],
    order_actions: OrderActions[StateT, ActionT] = _identity_order,
) -> DecisionResult[ActionT]:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    stats = GameSearchStats()
    started = perf_counter()

    def visit(current: StateT, remaining_depth: int) -> tuple[float, ActionT | None]:
        stats.nodes_visited += 1
        if remaining_depth == 0 or is_terminal(current):
            stats.leaf_evaluations += 1
            return float(evaluate(current)), None

        maximizing = is_maximizing(current)
        actions = list(order_actions(current, legal_actions(current), maximizing))
        if not actions:
            stats.leaf_evaluations += 1
            return float(evaluate(current)), None

        best_value = -inf if maximizing else inf
        best_action: ActionT | None = None
        for action in actions:
            child_value, _ = visit(apply_action(current, action), remaining_depth - 1)
            if (maximizing and child_value > best_value) or (
                not maximizing and child_value < best_value
            ):
                best_value = child_value
                best_action = action
        return best_value, best_action

    value, best_action = visit(state, depth)
    stats.runtime_ms = (perf_counter() - started) * 1000.0
    return DecisionResult(best_action=best_action, value=value, stats=stats)


def alpha_beta_decision(
    state: StateT,
    depth: int,
    *,
    is_terminal: IsTerminal[StateT],
    evaluate: Evaluate[StateT],
    legal_actions: LegalActions[StateT, ActionT],
    apply_action: ApplyAction[StateT, ActionT],
    is_maximizing: IsMaximizing[StateT],
    order_actions: OrderActions[StateT, ActionT] = _identity_order,
    use_transposition_table: bool = False,
    transposition_key: Callable[[StateT], Hashable] | None = None,
    transposition_offset: Callable[[StateT], float] | None = None,
) -> DecisionResult[ActionT]:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    stats = GameSearchStats()
    started = perf_counter()
    table: dict[tuple[Hashable, int], _TranspositionEntry[ActionT]] = {}

    def cache_key(current: StateT, remaining_depth: int) -> tuple[Hashable, int]:
        state_key = transposition_key(current) if transposition_key else current
        try:
            hash(state_key)
        except TypeError as error:
            raise TypeError(
                "transposition table requires a hashable state or transposition_key"
            ) from error
        return state_key, remaining_depth

    def offset(current: StateT) -> float:
        return float(transposition_offset(current)) if transposition_offset else 0.0

    def visit(
        current: StateT, remaining_depth: int, alpha: float, beta: float
    ) -> tuple[float, ActionT | None]:
        stats.nodes_visited += 1
        alpha_original, beta_original = alpha, beta
        key: tuple[Hashable, int] | None = None
        if use_transposition_table:
            key = cache_key(current, remaining_depth)
            entry = table.get(key)
            if entry is not None:
                stats.cache_hits += 1
                cached_value = entry.value + offset(current)
                if entry.flag == "EXACT":
                    return cached_value, entry.best_action
                if entry.flag == "LOWER":
                    alpha = max(alpha, cached_value)
                else:
                    beta = min(beta, cached_value)
                if alpha >= beta:
                    return cached_value, entry.best_action

        if remaining_depth == 0 or is_terminal(current):
            stats.leaf_evaluations += 1
            value = float(evaluate(current))
            if use_transposition_table and key is not None:
                table[key] = _TranspositionEntry(value - offset(current), "EXACT", None)
                stats.cache_stores += 1
            return value, None

        maximizing = is_maximizing(current)
        actions = list(order_actions(current, legal_actions(current), maximizing))
        if not actions:
            stats.leaf_evaluations += 1
            value = float(evaluate(current))
            if use_transposition_table and key is not None:
                table[key] = _TranspositionEntry(value - offset(current), "EXACT", None)
                stats.cache_stores += 1
            return value, None

        best_value = -inf if maximizing else inf
        best_action: ActionT | None = None
        for index, action in enumerate(actions):
            child_value, _ = visit(
                apply_action(current, action), remaining_depth - 1, alpha, beta
            )
            if (maximizing and child_value > best_value) or (
                not maximizing and child_value < best_value
            ):
                best_value = child_value
                best_action = action

            if maximizing:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)
            if alpha >= beta:
                pruned = len(actions) - index - 1
                if pruned > 0:
                    stats.cutoffs += 1
                    stats.pruned_branches += pruned
                    break

        if use_transposition_table and key is not None:
            if best_value <= alpha_original:
                flag = "UPPER"
            elif best_value >= beta_original:
                flag = "LOWER"
            else:
                flag = "EXACT"
            table[key] = _TranspositionEntry(
                best_value - offset(current), flag, best_action
            )
            stats.cache_stores += 1
        return best_value, best_action

    value, best_action = visit(state, depth, -inf, inf)
    stats.runtime_ms = (perf_counter() - started) * 1000.0
    return DecisionResult(best_action=best_action, value=value, stats=stats)


@dataclass
class TreeNode:
    value: float | None = None
    children: list["TreeNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass(frozen=True)
class _TreeState:
    node: TreeNode
    maximizing: bool


def _tree_value(node: TreeNode, maximizing: bool) -> float:
    if node.is_leaf:
        if node.value is None:
            raise ValueError("leaf node must have a value")
        return float(node.value)
    values = [_tree_value(child, not maximizing) for child in node.children]
    return max(values) if maximizing else min(values)


def _tree_order(mode: str) -> OrderActions[_TreeState, int]:
    if mode not in {"left_to_right", "best_first", "worst_first"}:
        raise ValueError(f"unknown tree order: {mode}")

    def order(state: _TreeState, actions: Sequence[int], maximizing: bool) -> Sequence[int]:
        if mode == "left_to_right":
            return actions
        reverse = maximizing if mode == "best_first" else not maximizing
        return sorted(
            actions,
            key=lambda index: _tree_value(state.node.children[index], not maximizing),
            reverse=reverse,
        )

    return order


def _tree_callbacks() -> dict[str, Callable]:
    return {
        "is_terminal": lambda state: state.node.is_leaf,
        "evaluate": lambda state: (
            float(state.node.value)
            if state.node.value is not None
            else (_ for _ in ()).throw(ValueError("leaf node must have a value"))
        ),
        "legal_actions": lambda state: list(range(len(state.node.children))),
        "apply_action": lambda state, action: _TreeState(
            state.node.children[action], not state.maximizing
        ),
        "is_maximizing": lambda state: state.maximizing,
    }


def _tree_depth(node: TreeNode) -> int:
    return 0 if node.is_leaf else 1 + max(_tree_depth(child) for child in node.children)


def minimax_tree(root: TreeNode, *, order: str = "left_to_right") -> DecisionResult[int]:
    return minimax_decision(
        _TreeState(root, True),
        _tree_depth(root),
        order_actions=_tree_order(order),
        **_tree_callbacks(),
    )


def alpha_beta_tree(
    root: TreeNode, *, order: str = "left_to_right"
) -> DecisionResult[int]:
    return alpha_beta_decision(
        _TreeState(root, True),
        _tree_depth(root),
        order_actions=_tree_order(order),
        **_tree_callbacks(),
    )


def build_sample_tree() -> TreeNode:
    """Build independent nodes equivalent to the tree shown in the PDF."""

    max1 = TreeNode(children=[TreeNode(3), TreeNode(5)])
    max2_left = TreeNode(children=[TreeNode(6), TreeNode(9)])
    max2_right = TreeNode(children=[TreeNode(6), TreeNode(9)])
    max3 = TreeNode(children=[TreeNode(1), TreeNode(2)])
    return TreeNode(
        children=[
            TreeNode(children=[max1, max2_left]),
            TreeNode(children=[max2_right, max3]),
        ]
    )

