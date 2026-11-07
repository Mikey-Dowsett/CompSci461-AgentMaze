import unittest

from agent import (astar_search, bfs_search, best_first_search, dfs_search,
                   id_dfs_search)
from maze import Maze


class SearchMetricsTests(unittest.TestCase):
    def test_bfs_metrics_and_path_cost(self):
        maze = Maze([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        result = bfs_search(maze)

        self.assertEqual(result.solution_depth, 4)
        self.assertEqual(result.path_cost, 4)
        self.assertEqual(result.metrics.nodes_generated, 8)
        self.assertEqual(result.metrics.nodes_expanded, len(result.explored))
        self.assertGreaterEqual(result.metrics.peak_frontier_plus_explored,
                                result.metrics.peak_frontier)
        self.assertGreaterEqual(result.metrics.branching_max,
                                result.metrics.branching_avg)

    def test_movement_modes_apply_to_uninformed_searches(self):
        maze = Maze([[0, 0], [0, 0]])

        self.assertEqual(bfs_search(maze).solution_depth, 2)
        self.assertEqual(bfs_search(maze, diagonal=True).solution_depth, 1)
        self.assertEqual(dfs_search(maze, diagonal=True).solution_depth, 1)
        self.assertEqual(id_dfs_search(maze, diagonal=True).solution_depth, 1)

    def test_astar_is_shortest_on_orthogonal_unit_grid(self):
        maze = Maze([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        bfs = bfs_search(maze)
        astar = astar_search(maze, heuristic="manhattan")
        greedy = best_first_search(maze, heuristic="manhattan")

        self.assertEqual(astar.solution_depth, bfs.solution_depth)
        self.assertIsNotNone(greedy.path)
        self.assertGreater(astar.metrics.nodes_generated, 0)

    def test_iddfs_matches_bfs_solution_depth_on_seeded_mazes(self):
        for seed in range(8):
            maze, _ = Maze.generate_solvable(5, 5, 0.2, seed=seed)
            self.assertEqual(id_dfs_search(maze).solution_depth,
                             bfs_search(maze).solution_depth)

    def test_trace_captures_frontier_nodes(self):
        maze = Maze([[0, 0], [0, 0]])
        result = bfs_search(maze, capture_trace=True)

        self.assertEqual(len(result.events), len(result.explored))
        self.assertIn("frontier_nodes", result.events[0])
        self.assertTrue(result.events[0]["frontier_nodes"])

    def test_dfs_walk_uses_legal_diagonal_steps(self):
        maze = Maze([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        result = dfs_search(maze, diagonal=True)

        for current, following in zip(result.walk, result.walk[1:]):
            self.assertIn(following, maze.neighbors(current, diagonal=True))

    def test_start_is_goal_has_zero_cost(self):
        maze = Maze([[0]], start=(0, 0), goal=(0, 0))
        result = bfs_search(maze)

        self.assertEqual(result.solution_depth, 0)
        self.assertEqual(result.path_cost, 0)
        self.assertEqual(result.metrics.nodes_expanded, 1)


if __name__ == "__main__":
    unittest.main()
