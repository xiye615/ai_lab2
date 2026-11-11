"""Generate compact figures for the complete report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "data" / "enhancement_results.csv"
OUTPUT_DIRECTORY = ROOT / "results" / "figures"
LABELS = ["Baseline", "Ordering", "Cache", "Both"]
COLORS = ("#315B7D", "#A9C4D8")


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def game_rows(rows: list[dict[str, str]], scenario: str, depth: int):
    selected = [
        row
        for row in rows
        if row["scenario"] == scenario and row["depth"] == str(depth)
    ]
    order = {name: index for index, name in enumerate(
        ("baseline", "ordering", "cache", "ordering_cache")
    )}
    return sorted(selected, key=lambda row: order[row["configuration"]])


def grouped_bars(axis, rows, title: str) -> None:
    positions = list(range(4))
    width = 0.36
    visited = [int(row["visited_nodes"]) for row in rows]
    leaves = [int(row["leaf_evaluations"]) for row in rows]
    bars_a = axis.bar(
        [position - width / 2 for position in positions],
        visited,
        width,
        color=COLORS[0],
        label="Visited nodes",
    )
    bars_b = axis.bar(
        [position + width / 2 for position in positions],
        leaves,
        width,
        color=COLORS[1],
        label="Leaf evaluations",
    )
    axis.bar_label(bars_a, padding=2, fontsize=8)
    axis.bar_label(bars_b, padding=2, fontsize=8)
    axis.set_xticks(positions, LABELS, rotation=12)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)


def improvement_figure(rows: list[dict[str, str]]) -> None:
    four = game_rows(rows, "pdf_four_treasures", 4)
    six = game_rows(rows, "open_grid_six_treasures", 6)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    grouped_bars(axes[0], four, "Four treasures at H = 4")
    grouped_bars(axes[1], six, "Six treasures at H = 6")
    axes[0].set_ylabel("Count")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(
        OUTPUT_DIRECTORY / "improvement_ablation_complete.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(figure)


def extension_figure(rows: list[dict[str, str]]) -> None:
    h4 = game_rows(rows, "open_grid_six_treasures", 4)
    h6 = game_rows(rows, "open_grid_six_treasures", 6)
    counterexample = [
        row for row in rows if row["experiment"] == "astar_counterexample"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    positions = list(range(4))
    width = 0.36
    bars_h4 = axes[0].bar(
        [position - width / 2 for position in positions],
        [int(row["visited_nodes"]) for row in h4],
        width,
        color=COLORS[1],
        label="H = 4",
    )
    bars_h6 = axes[0].bar(
        [position + width / 2 for position in positions],
        [int(row["visited_nodes"]) for row in h6],
        width,
        color=COLORS[0],
        label="H = 6",
    )
    axes[0].bar_label(bars_h4, padding=2, fontsize=8)
    axes[0].bar_label(bars_h6, padding=2, fontsize=8)
    axes[0].set_xticks(positions, LABELS, rotation=12)
    axes[0].set_ylabel("Visited nodes")
    axes[0].set_title("Six-treasure depth scaling")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].spines[["top", "right"]].set_visible(False)

    names = ["h = 0", "Adjusted", "Raw Manhattan"]
    costs = [float(row["cost"]) for row in counterexample]
    bars = axes[1].bar(names, costs, color=[COLORS[1], COLORS[0], "#D58A4A"])
    axes[1].bar_label(bars, padding=3)
    axes[1].set_ylabel("Returned path cost")
    axes[1].set_title("Zero-cost terrain counterexample")
    axes[1].set_ylim(0, max(costs) + 1)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY / "extension_results_complete.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    improvement_figure(rows)
    extension_figure(rows)
    print(f"Wrote complete-report figures to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()

