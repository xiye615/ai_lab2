import unittest

from src.treasure_game import LOCATION_ORDER, compute_distance_matrix, solve_treasure_game


class TreasureGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.distances, _ = compute_distance_matrix()

    def test_distance_matrix_is_six_by_six(self):
        self.assertEqual(set(self.distances), set(LOCATION_ORDER))
        for row in self.distances.values():
            self.assertEqual(set(row), set(LOCATION_ORDER))

    def test_pdf_depth_two_result(self):
        for algorithm in ("minimax", "alpha_beta"):
            result = solve_treasure_game(
                2, algorithm=algorithm, distances=self.distances
            )
            self.assertEqual(result.best_action, "T2")
            self.assertEqual(result.value, -1)

    def test_pdf_depth_four_result(self):
        for algorithm in ("minimax", "alpha_beta"):
            result = solve_treasure_game(
                4, algorithm=algorithm, distances=self.distances
            )
            self.assertEqual(result.best_action, "T3")
            self.assertEqual(result.value, 3)

    def test_ordering_preserves_the_decision_value(self):
        values = {
            solve_treasure_game(
                4, algorithm="alpha_beta", order=order, distances=self.distances
            ).value
            for order in ("left_to_right", "best_first", "worst_first")
        }
        self.assertEqual(values, {3})

    def test_transposition_table_preserves_results(self):
        for depth, target, value in ((2, "T2", -1), (4, "T3", 3)):
            result = solve_treasure_game(
                depth,
                algorithm="alpha_beta",
                order="best_first",
                distances=self.distances,
                use_cache=True,
            )
            self.assertEqual(result.best_action, target)
            self.assertEqual(result.value, value)
            self.assertGreater(result.stats.cache_stores, 0)


if __name__ == "__main__":
    unittest.main()

