import unittest

from batch_runner import (DEFAULT_DENSITIES, DEFAULT_REPETITIONS,
                          DEFAULT_SIZES, _method_configurations, run_profiled)
from maze import Maze


class BatchProtocolTests(unittest.TestCase):
    def test_default_protocol_has_twelve_graph_parameter_combinations(self):
        self.assertEqual(len(DEFAULT_SIZES) * len(DEFAULT_DENSITIES), 12)
        self.assertGreaterEqual(DEFAULT_REPETITIONS, 5)

    def test_profiled_worker_returns_runtime_and_memory_metrics(self):
        maze = Maze([[0, 0], [0, 0]])
        result = run_profiled(maze, {
            "algorithm": "bfs", "movement": "orthogonal", "heuristic": None,
        })

        self.assertGreater(result["runtime_ns"], 0)
        self.assertGreaterEqual(result["python_peak_bytes"], 0)
        self.assertEqual(result["metrics"]["nodes_expanded"], 4)

    def test_suite_covers_methods_movement_and_applicable_heuristics(self):
        configs = _method_configurations()
        self.assertEqual(len(configs), 18)
        for algorithm in ("bfs", "dfs", "iddfs"):
            variants = [row for row in configs if row["algorithm"] == algorithm]
            self.assertEqual({row["movement"] for row in variants},
                             {"orthogonal", "diagonal"})
            self.assertTrue(all(row["heuristic"] is None for row in variants))
        informed = [row for row in configs
                    if row["algorithm"] in {"best_first", "astar"}]
        self.assertEqual(len(informed), 12)
        self.assertEqual({row["heuristic"] for row in informed},
                         {"manhattan", "euclidean", "chebyshev"})


if __name__ == "__main__":
    unittest.main()
