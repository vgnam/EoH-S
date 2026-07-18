from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from llm4ad.task.optimization.aco_pheromone_set import (  # noqa: E402
    CVRPACOPheromoneEvaluation,
    CVRP_TEMPLATE_PROGRAM,
    TSPACOPheromoneEvaluation,
    TSP_TEMPLATE_PROGRAM,
)
from llm4ad.base import LLM, SampleTrimmer, SecureEvaluator  # noqa: E402
from llm4ad.method.eohs import EoHS  # noqa: E402


def template_callable(program: str):
    namespace = {}
    exec(program, namespace)
    return namespace["update_pheromone"]


class StaticACOLLM(LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        return "{Use the classical all-ant deposit rule.}\n" + TSP_TEMPLATE_PROGRAM


class InvalidLLM(LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        return "not a function"


class ACOPheromoneEvaluationTests(unittest.TestCase):
    def test_sample_trimmer_accepts_multiline_function_signature(self):
        sampled = """{thought}
def update_pheromone(
        pheromone,
        ant_tours):
    return pheromone
"""
        body = SampleTrimmer.trim_preface_of_function(sampled)
        self.assertEqual(body.strip(), "return pheromone")

    def test_tsp_ant_system_template_is_deterministic(self):
        coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        distances = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2
        )
        evaluation = TSPACOPheromoneEvaluation(
            instances=[(coordinates, distances, 4.0)],
            n_ants=4,
            iterations=4,
            seed=7,
            safe_evaluate=False,
        )
        update = template_callable(TSP_TEMPLATE_PROGRAM)
        first = evaluation.evaluate_program(TSP_TEMPLATE_PROGRAM, update)
        second = evaluation.evaluate_program(TSP_TEMPLATE_PROGRAM, update)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertTrue(np.isfinite(first[0]))

    def test_cvrp_ant_system_template_returns_finite_utility(self):
        coordinates = np.array(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float
        )
        distances = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2
        )
        demands = np.array([0, 1, 1, 1, 1])
        evaluation = CVRPACOPheromoneEvaluation(
            instances=[(coordinates, distances, demands, 2, 8.0)],
            n_ants=4,
            iterations=4,
            seed=11,
            safe_evaluate=False,
        )
        scores = evaluation.evaluate_program(
            CVRP_TEMPLATE_PROGRAM,
            template_callable(CVRP_TEMPLATE_PROGRAM),
        )
        self.assertEqual(len(scores), 1)
        self.assertTrue(np.isfinite(scores[0]))

    def test_invalid_pheromone_update_is_rejected(self):
        coordinates = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        distances = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2
        )
        evaluation = TSPACOPheromoneEvaluation(
            instances=[(coordinates, distances, 4.0)],
            n_ants=2,
            iterations=2,
            safe_evaluate=False,
        )

        def invalid_update(pheromone, *_args):
            return -np.ones_like(pheromone)

        self.assertIsNone(evaluation.evaluate_program("", invalid_update))

    def test_template_runs_through_secure_evaluator(self):
        coordinates = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        distances = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2
        )
        evaluation = TSPACOPheromoneEvaluation(
            instances=[(coordinates, distances, 4.0)],
            n_ants=2,
            iterations=2,
            timeout_seconds=30,
        )
        scores = SecureEvaluator(evaluation).evaluate_program(TSP_TEMPLATE_PROGRAM)
        self.assertIsNotNone(scores)
        self.assertEqual(len(scores), 1)

    def test_eohs_can_search_the_tsp_aco_function_contract(self):
        coordinates = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        distances = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2
        )
        evaluation = TSPACOPheromoneEvaluation(
            instances=[(coordinates, distances, 4.0)],
            n_ants=2,
            iterations=2,
            timeout_seconds=30,
        )
        method = EoHS(
            llm=StaticACOLLM(),
            evaluation=evaluation,
            max_sample_nums=2,
            max_generations=1,
            pop_size=1,
            num_samplers=1,
            num_evaluators=1,
        )
        method.run()
        self.assertGreaterEqual(len(method._population.population), 1)
        self.assertIsNotNone(method._population.population[0].score)

    def test_eohs_invalid_response_consumes_budget_and_terminates(self):
        coordinates = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        distances = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2
        )
        evaluation = TSPACOPheromoneEvaluation(
            instances=[(coordinates, distances, 4.0)],
            n_ants=1,
            iterations=1,
            timeout_seconds=30,
        )
        method = EoHS(
            llm=InvalidLLM(),
            evaluation=evaluation,
            max_sample_nums=1,
            max_generations=1,
            pop_size=1,
            num_samplers=1,
            num_evaluators=1,
        )
        method.run()
        self.assertEqual(method._tot_sample_nums, 1)
        self.assertEqual(len(method._population.population), 0)


if __name__ == "__main__":
    unittest.main()
