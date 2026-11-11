"""Run the required baseline comparisons and save reproducible raw data."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.game_search import alpha_beta_tree, build_sample_tree, minimax_tree
from src.task1_maze import SAMPLE_GOAL as MAZE_GOAL
from src.task1_maze import SAMPLE_GRID as MAZE_GRID
from src.task1_maze import SAMPLE_START as MAZE_START
from src.task1_maze import solve_maze
from src.task2_weighted import SAMPLE_GOAL as WEIGHTED_GOAL
from src.task2_weighted import SAMPLE_GRID as WEIGHTED_GRID
from src.task2_weighted import SAMPLE_START as WEIGHTED_START
from src.task2_weighted import solve_weighted_terrain
from src.treasure_game import LOCATION_ORDER, compute_distance_matrix, solve_treasure_game

DATA_DIRECTORY = ROOT / "results" / "data"
REPETITIONS = 50


def repeated(run: Callable, repetitions: int = REPETITIONS):
    results = [run() for _ in range(repetitions)]
    median_ms = statistics.median(result.stats.runtime_ms for result in results)
    return results[-1], median_ms


def empty_row() -> dict[str, object]:
    return {
        "category": "",
        "scenario": "",
        "algorithm": "",
        "configuration": "",
        "depth": "",
        "result": "",
        "cost_or_value": "",
        "expanded_or_visited_nodes": "",
        "generated_nodes": "",
        "leaf_evaluations": "",
        "cutoffs": "",
        "pruned_branches": "",
        "max_frontier": "",
        "runtime_ms_median": "",
    }


def main() -> None:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for heuristic in ("zero", "manhattan"):
        result, median_ms = repeated(
            lambda heuristic=heuristic: solve_maze(
                MAZE_START, MAZE_GOAL, MAZE_GRID, heuristic_name=heuristic
            )
        )
        row = empty_row()
        row.update(
            category="astar",
            scenario="task1_pdf_maze",
            algorithm="A*",
            configuration=heuristic,
            result=str(result.path),
            cost_or_value=result.cost,
            expanded_or_visited_nodes=result.stats.expanded_nodes,
            generated_nodes=result.stats.generated_nodes,
            max_frontier=result.stats.max_frontier,
            runtime_ms_median=round(median_ms, 6),
        )
        rows.append(row)

    for heuristic in ("zero", "adjusted_manhattan", "manhattan"):
        result, median_ms = repeated(
            lambda heuristic=heuristic: solve_weighted_terrain(
                WEIGHTED_START,
                WEIGHTED_GOAL,
                WEIGHTED_GRID,
                heuristic_name=heuristic,
            )
        )
        row = empty_row()
        row.update(
            category="astar",
            scenario="task2_pdf_weighted",
            algorithm="A*",
            configuration=heuristic,
            result=str(result.path),
            cost_or_value=result.cost,
            expanded_or_visited_nodes=result.stats.expanded_nodes,
            generated_nodes=result.stats.generated_nodes,
            max_frontier=result.stats.max_frontier,
            runtime_ms_median=round(median_ms, 6),
        )
        rows.append(row)

    sample_tree = build_sample_tree()
    tree_runs = [("minimax", "left_to_right", lambda: minimax_tree(sample_tree))]
    tree_runs.extend(
        (
            "alpha_beta",
            order,
            lambda order=order: alpha_beta_tree(sample_tree, order=order),
        )
        for order in ("left_to_right", "best_first", "worst_first")
    )
    for algorithm, order, run in tree_runs:
        result, median_ms = repeated(run)
        row = empty_row()
        row.update(
            category="game_tree",
            scenario="task3_pdf_tree",
            algorithm=algorithm,
            configuration=order,
            result=result.best_action,
            cost_or_value=result.value,
            expanded_or_visited_nodes=result.stats.nodes_visited,
            leaf_evaluations=result.stats.leaf_evaluations,
            cutoffs=result.stats.cutoffs,
            pruned_branches=result.stats.pruned_branches,
            runtime_ms_median=round(median_ms, 6),
        )
        rows.append(row)

    distances, _ = compute_distance_matrix()
    for depth in (2, 4):
        configurations = [("minimax", "left_to_right")]
        configurations.extend(
            ("alpha_beta", order)
            for order in ("left_to_right", "best_first", "worst_first")
        )
        for algorithm, order in configurations:
            result, median_ms = repeated(
                lambda depth=depth, algorithm=algorithm, order=order: solve_treasure_game(
                    depth,
                    algorithm=algorithm,
                    order=order,
                    distances=distances,
                )
            )
            row = empty_row()
            row.update(
                category="treasure_game",
                scenario="task4_pdf_map",
                algorithm=algorithm,
                configuration=order,
                depth=depth,
                result=result.best_action,
                cost_or_value=result.value,
                expanded_or_visited_nodes=result.stats.nodes_visited,
                leaf_evaluations=result.stats.leaf_evaluations,
                cutoffs=result.stats.cutoffs,
                pruned_branches=result.stats.pruned_branches,
                runtime_ms_median=round(median_ms, 6),
            )
            rows.append(row)

    csv_path = DATA_DIRECTORY / "baseline_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(empty_row()))
        writer.writeheader()
        writer.writerows(rows)

    matrix_path = DATA_DIRECTORY / "distance_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["from/to", *LOCATION_ORDER])
        for source in LOCATION_ORDER:
            writer.writerow([source, *(distances[source][target] for target in LOCATION_ORDER)])

    print(f"Wrote {csv_path}")
    print(f"Wrote {matrix_path}")
    print(f"Recorded {len(rows)} comparison rows")


if __name__ == "__main__":
    main()

