import unittest

from src.extended_treasure_game import build_six_treasure_problem, solve_allocation
from src.task2_weighted import solve_weighted_terrain


class ExtensionTests(unittest.TestCase):
    def test_zero_cost_counterexample(self):
        grid = [["S", 1, 1], [1, 4, 1], [0, 0, "E"]]
        optimal = solve_weighted_terrain(
            (0, 0), (2, 2), grid, heuristic_name="adjusted_manhattan"
        )
        raw = solve_weighted_terrain(
            (0, 0), (2, 2), grid, heuristic_name="manhattan"
        )
        self.assertEqual(optimal.cost, 1)
        self.assertEqual(raw.cost, 3)

    def test_six_treasure_optimizations_preserve_result(self):
        problem = build_six_treasure_problem()
        results = [
            solve_allocation(problem, 6, order=order, use_cache=cache)
            for order, cache in (
                ("left_to_right", False),
                ("best_first", False),
                ("left_to_right", True),
                ("best_first", True),
            )
        ]
        self.assertEqual({(result.best_action, result.value) for result in results}, {("T2", 6)})
        self.assertGreater(results[-1].stats.cache_hits, 0)


if __name__ == "__main__":
    unittest.main()

