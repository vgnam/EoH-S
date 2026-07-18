from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "examples" / "training"))

from aco_pheromone_common import (  # noqa: E402
    evaluate_portfolio,
    instance_descriptor,
    make_evaluation,
    normalize_instance,
)
from llm4ad.base import LLM, TextFunctionProgramConverter  # noqa: E402
from llm4ad.method.ow_cahd import OWCAHD, OWCAHDConfig  # noqa: E402
from llm4ad.task.optimization.aco_pheromone_set import TSP_TEMPLATE_PROGRAM  # noqa: E402


class StaticACOLLM(LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        return "{Use classical Ant System deposition.}\n" + TSP_TEMPLATE_PROGRAM


class ACOTrainingCommonTests(unittest.TestCase):
    def test_raw_coordinate_lists_are_not_mistaken_for_full_instances(self):
        tsp_coordinates = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        cvrp_coordinates = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.2],
            [0.2, 0.5],
        ]
        self.assertEqual(len(normalize_instance(tsp_coordinates, "tsp")[0]), 3)
        self.assertEqual(len(normalize_instance(cvrp_coordinates, "cvrp")[0]), 5)
        self.assertEqual(instance_descriptor(tsp_coordinates, "tsp").shape, (8,))
        self.assertEqual(instance_descriptor(cvrp_coordinates, "cvrp").shape, (11,))

    def test_parallel_portfolio_post_evaluation(self):
        coordinates = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        distances = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2
        )
        instance = (coordinates, distances, 4.0)
        function = TextFunctionProgramConverter.text_to_function(TSP_TEMPLATE_PROGRAM)
        utility, records = evaluate_portfolio(
            "tsp",
            [function, copy.deepcopy(function)],
            [instance],
            {
                "n_ants": 2,
                "iterations": 2,
                "n_runs": 1,
                "seed": 7,
                "timeout_seconds": 30,
            },
            workers=2,
        )
        self.assertEqual(len(utility), 1)
        self.assertTrue(np.isfinite(utility[0]))
        self.assertTrue(all(record["valid"] for record in records))

    def test_ow_cahd_can_use_tsp_aco_evaluation_factory(self):
        coordinates = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        distances = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2
        )
        instance = (coordinates, distances, 4.0)
        aco_config = {
            "n_ants": 1,
            "iterations": 1,
            "n_runs": 1,
            "seed": 7,
            "timeout_seconds": 30,
        }
        method = OWCAHD(
            llm=StaticACOLLM(),
            descriptor=lambda item: instance_descriptor(item, "tsp"),
            evaluation_factory=lambda replay: make_evaluation(
                "tsp", replay, aco_config
            ),
            validity_fn=lambda _item: True,
            config=OWCAHDConfig(
                portfolio_size=1,
                sleep_instances_per_round=1,
                min_sleep_per_regime=1,
                auto_accept_regime=True,
                max_sample_nums=2,
                max_generations=1,
                pop_size=1,
                num_samplers=1,
                num_evaluators=1,
            ),
        )
        result = method.step([instance], round_id=0)
        self.assertGreater(result.eohs_total_samples_used, 0)
        self.assertEqual(len(result.portfolio), 1)


if __name__ == "__main__":
    unittest.main()
