from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from llm4ad.baselines.aco import (  # noqa: E402
    ACOParameters,
    solve_bpp_aco,
    solve_cvrp_aco,
    solve_tsp_aco,
)


FAST = ACOParameters(ants=4, iterations=5, evaporation=0.2)


class ACOBaselineTests(unittest.TestCase):
    def test_tsp_returns_deterministic_permutation(self):
        coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        first = solve_tsp_aco(distances, FAST, seed=7)
        second = solve_tsp_aco(distances, FAST, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(set(first.tour), set(range(4)))
        self.assertAlmostEqual(first.cost, 4.0)

    def test_cvrp_routes_are_complete_and_capacity_feasible(self):
        coordinates = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        demands = np.array([0, 1, 1, 1, 1])
        solution = solve_cvrp_aco(distances, demands, capacity=2, parameters=FAST, seed=3)
        visited = [node for route in solution.routes for node in route[1:-1]]
        self.assertEqual(sorted(visited), [1, 2, 3, 4])
        for route in solution.routes:
            self.assertEqual(route[0], 0)
            self.assertEqual(route[-1], 0)
            self.assertLessEqual(sum(demands[node] for node in route), 2)

    def test_bpp_finds_two_exact_bins(self):
        solution = solve_bpp_aco(
            np.array([6, 4, 6, 4]),
            capacity=10,
            parameters=FAST,
            seed=11,
            position_buckets=4,
        )
        self.assertEqual(solution.bin_count, 2)
        self.assertEqual(solution.lower_bound, 2)
        self.assertEqual(sorted(solution.order), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()

