from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.jssp_construct.get_instance import GetData
from .template import task_description, template_program

__all__ = ["FSSPEvaluation"]


class FSSPEvaluation(Evaluation):
    def __init__(self, timeout_seconds=120, n_instance=16, n_jobs=50,
                 n_machines=10, instances=None, return_list=False, **kwargs):
        super().__init__(template_program=template_program,
                         task_description=task_description,
                         use_numba_accelerate=False,
                         timeout_seconds=timeout_seconds)
        self.return_list = return_list
        if instances is None:
            self._datasets = GetData(n_instance, n_jobs, n_machines).generate_instances()
        else:
            self._datasets = list(instances)
        self.n_instance = len(self._datasets)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    @staticmethod
    def makespan(processing_times, sequence):
        values = np.asarray(processing_times, dtype=int)
        completion = np.zeros(values.shape[1], dtype=np.int64)
        for job in sequence:
            completion[0] += values[job, 0]
            for machine in range(1, values.shape[1]):
                completion[machine] = max(completion[machine], completion[machine - 1]) + values[job, machine]
        return int(completion[-1])

    def construct(self, processing_times, n_jobs, heuristic):
        remaining = list(range(n_jobs))
        sequence = []
        while remaining:
            selected = heuristic(sequence.copy(), remaining.copy(), processing_times)
            if selected not in remaining:
                raise ValueError("Heuristic must return one remaining job id.")
            selected = int(selected)
            sequence.append(selected)
            remaining.remove(selected)
        return sequence

    def evaluate(self, heuristic):
        scores = []
        for processing_times, n_jobs, _ in self._datasets[:self.n_instance]:
            sequence = self.construct(processing_times, int(n_jobs), heuristic)
            scores.append(-float(self.makespan(processing_times, sequence)))
        return scores if self.return_list else float(np.mean(scores))
