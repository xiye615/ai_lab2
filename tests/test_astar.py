import math
import unittest

from src.task1_maze import SAMPLE_GOAL as MAZE_GOAL
from src.task1_maze import SAMPLE_GRID as MAZE_GRID
from src.task1_maze import SAMPLE_START as MAZE_START
from src.task1_maze import solve_maze
from src.task2_weighted import SAMPLE_GOAL as WEIGHTED_GOAL
from src.task2_weighted import SAMPLE_GRID as WEIGHTED_GRID
from src.task2_weighted import SAMPLE_START as WEIGHTED_START
from src.task2_weighted import minimum_terrain_cost, solve_weighted_terrain


class MazeTests(unittest.TestCase):
    def test_pdf_sample_returns_a_shortest_path(self):
        result = solve_maze(MAZE_START, MAZE_GOAL, MAZE_GRID)
        self.assertEqual(result.path[0], MAZE_START)
        self.assertEqual(result.path[-1], MAZE_GOAL)
        self.assertEqual(result.cost, 7)
        self.assertEqual(len(result.path), 8)

    def test_zero_and_manhattan_agree(self):
        zero = solve_maze(MAZE_START, MAZE_GOAL, MAZE_GRID, heuristic_name="zero")
        informed = solve_maze(
            MAZE_START, MAZE_GOAL, MAZE_GRID, heuristic_name="manhattan"
        )
        self.assertEqual(zero.cost, informed.cost)

    def test_unreachable_maze_returns_empty_path(self):
        grid = [["S", "X", "E"], ["X", "X", "X"]]
        result = solve_maze((0, 0), (0, 2), grid)
        self.assertEqual(result.path, [])
        self.assertTrue(math.isinf(result.cost))


class WeightedTerrainTests(unittest.TestCase):
    def test_pdf_sample_cost_is_seven(self):
        self.assertEqual(
            minimum_terrain_cost(WEIGHTED_START, WEIGHTED_GOAL, WEIGHTED_GRID), 7
        )

    def test_admissible_heuristic_matches_uniform_cost(self):
        zero = solve_weighted_terrain(
            WEIGHTED_START, WEIGHTED_GOAL, WEIGHTED_GRID, heuristic_name="zero"
        )
        adjusted = solve_weighted_terrain(
            WEIGHTED_START,
            WEIGHTED_GOAL,
            WEIGHTED_GRID,
            heuristic_name="adjusted_manhattan",
        )
        self.assertEqual(zero.cost, adjusted.cost)

    def test_goal_entry_is_free(self):
        result = solve_weighted_terrain((0, 0), (0, 1), [["S", "E"]])
        self.assertEqual(result.cost, 0)


if __name__ == "__main__":
    unittest.main()

