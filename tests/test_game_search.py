import unittest

from src.game_search import alpha_beta_tree, build_sample_tree, minimax_tree


class GameTreeTests(unittest.TestCase):
    def test_pdf_sample_value_is_five(self):
        root = build_sample_tree()
        self.assertEqual(minimax_tree(root).value, 5)
        self.assertEqual(alpha_beta_tree(root).value, 5)

    def test_alpha_beta_visits_no_more_nodes_than_minimax(self):
        root = build_sample_tree()
        baseline = minimax_tree(root)
        pruned = alpha_beta_tree(root)
        self.assertLessEqual(pruned.stats.nodes_visited, baseline.stats.nodes_visited)
        self.assertGreater(pruned.stats.pruned_branches, 0)

    def test_ordering_does_not_change_value(self):
        root = build_sample_tree()
        values = {
            alpha_beta_tree(root, order=order).value
            for order in ("left_to_right", "best_first", "worst_first")
        }
        self.assertEqual(values, {5})


if __name__ == "__main__":
    unittest.main()

