"""Run improvement ablations and self-designed extension experiments."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extended_treasure_game import build_six_treasure_problem, solve_allocation
from src.task2_weighted import solve_weighted_terrain
from src.treasure_game import compute_distance_matrix, solve_treasure_game

OUTPUT_DIRECTORY = ROOT / "results" / "data"
REPETITIONS = 100
CONFIGURATIONS = (
    ("baseline", "left_to_right", False),
    ("ordering", "best_first", False),
    ("cache", "left_to_right", True),
    ("ordering_cache", "best_first", True),
)


def median_run(run: Callable):
    results = [run() for _ in range(REPETITIONS)]
    return results[-1], statistics.median(item.stats.runtime_ms for item in results)


def game_row(
    scenario: str, depth: int, name: str, result, median_ms: float
) -> dict[str, object]:
    return {
        "experiment": "game_ablation",
        "scenario": scenario,
        "depth": depth,
        "configuration": name,
        "best_action": result.best_action,
        "value": result.value,
        "visited_nodes": result.stats.nodes_visited,
        "leaf_evaluations": result.stats.leaf_evaluations,
        "cutoffs": result.stats.cutoffs,
        "pruned_branches": result.stats.pruned_branches,
        "cache_hits": result.stats.cache_hits,
        "cache_stores": result.stats.cache_stores,
        "runtime_ms_median": round(median_ms, 6),
        "path": "",
        "cost": "",
        "expanded_nodes": "",
    }


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    distances, _ = compute_distance_matrix()
    for name, order, use_cache in CONFIGURATIONS:
        result, median_ms = median_run(
            lambda order=order, use_cache=use_cache: solve_treasure_game(
                4,
                algorithm="alpha_beta",
                order=order,
                distances=distances,
                use_cache=use_cache,
            )
        )
        rows.append(game_row("pdf_four_treasures", 4, name, result, median_ms))

    problem = build_six_treasure_problem()
    for depth in (4, 6):
        for name, order, use_cache in CONFIGURATIONS:
            result, median_ms = median_run(
                lambda depth=depth, order=order, use_cache=use_cache: solve_allocation(
                    problem, depth, order=order, use_cache=use_cache
                )
            )
            rows.append(
                game_row("open_grid_six_treasures", depth, name, result, median_ms)
            )

    counterexample = [["S", 1, 1], [1, 4, 1], [0, 0, "E"]]
    for heuristic in ("zero", "adjusted_manhattan", "manhattan"):
        result, median_ms = median_run(
            lambda heuristic=heuristic: solve_weighted_terrain(
                (0, 0), (2, 2), counterexample, heuristic_name=heuristic
            )
        )
        rows.append(
            {
                "experiment": "astar_counterexample",
                "scenario": "zero_cost_terrain",
                "depth": "",
                "configuration": heuristic,
                "best_action": "",
                "value": "",
                "visited_nodes": "",
                "leaf_evaluations": "",
                "cutoffs": "",
                "pruned_branches": "",
                "cache_hits": "",
                "cache_stores": "",
                "runtime_ms_median": round(median_ms, 6),
                "path": str(result.path),
                "cost": result.cost,
                "expanded_nodes": result.stats.expanded_nodes,
            }
        )

    fields = list(rows[0])
    csv_path = OUTPUT_DIRECTORY / "enhancement_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")
    print(f"Recorded {len(rows)} rows")


if __name__ == "__main__":
    main()

