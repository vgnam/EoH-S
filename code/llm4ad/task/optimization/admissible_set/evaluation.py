# Module Name: ASPEvaluation
# Last Revision: 2025/2/14
# Description: Evaluates admissible sets for symmetric constant-weight optimization problems.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# 
# Parameters:
#   - dimension: int - The dimension of the problem space (default: 15).
#   - weight: int - The weight constraint for the admissible set (default: 10).
#   - timeout_seconds: int - Maximum allowed time (in seconds) for the evaluation process (default: 60).
# 
# References:
#   - Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, 
#     Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco JR Ruiz et al. 
#     "Mathematical discoveries from program search with large language models." 
#     Nature 625, no. 7995 (2024): 468-475.
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------


from __future__ import annotations

import itertools
import pickle
from pathlib import Path
from typing import Any, List, Tuple
import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.admissible_set.template import template_program, task_description

__all__ = ['ASPEvaluation']

class ASPEvaluation(Evaluation):
    """Evaluator for online bin packing problem."""

    def __init__(self, timeout_seconds=60, dimension=15, weight=10, dataset_file=None,
                 datasets=None, dataset_split=None, return_list=False, **kwargs):
        """
            Args:
                - 'dimension' (int): The dimension of tested case (default is 15).
                - 'weight' (int): The wight of tested case (default is 10).
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.dimension = dimension
        self.weight = weight
        self._return_list = return_list
        self._dataset_instances = None

        self.TRIPLES = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1), (2, 2, 2)]
        self.INT_TO_WEIGHT = [0, 1, 1, 2, 2, 3, 3]
        self.Optimal_Set_Length = {
            "n12w7": 792,
            "n15w10": 3003,
            "n21w15": 43596,
            "n24w17": 237984
        }

        if datasets is not None or dataset_file is not None or dataset_split is not None:
            self._load_dataset(dataset_file, datasets, dataset_split)

    @staticmethod
    def _repo_root():
        return Path(__file__).resolve().parents[5]

    def _resolve_path(self, path):
        path = Path(path)
        if path.is_absolute():
            return path
        return self._repo_root() / path

    def _load_manifest_file(self, path):
        path = self._resolve_path(path)
        with path.open("rb") as handle:
            manifest = pickle.load(handle)
        if not isinstance(manifest, dict) or "instances" not in manifest:
            raise ValueError(f"Admissible manifest {path} must contain an 'instances' field.")
        return manifest

    def _load_manifest_data(self, manifest):
        if isinstance(manifest, dict):
            instances = manifest.get("instances")
        elif (
            isinstance(manifest, (list, tuple))
            and manifest
            and all(isinstance(item, dict) and "dimension" in item for item in manifest)
        ):
            instances = manifest
        elif isinstance(manifest, (list, tuple)):
            instances = []
            for item in manifest:
                loaded = self._load_manifest_file(item)
                instances.extend(loaded["instances"])
        else:
            raise ValueError("Admissible dataset must be a manifest dict, instance list, or path list.")
        if not instances:
            raise ValueError("Admissible dataset contains no instances.")
        self._dataset_instances = [dict(instance) for instance in instances]

    def _load_dataset(self, dataset_file, datasets, dataset_split):
        if datasets is not None:
            self._load_manifest_data(datasets)
            return
        path = dataset_file
        if path is None and dataset_split is not None:
            split = "id" if dataset_split in {"test", "id"} else dataset_split
            if split not in {"train", "id", "ood"}:
                raise ValueError(f"Unsupported admissible dataset split: {dataset_split!r}")
            path = f"datasets/admissible/{split}/asp_{split}.pkl"
        if path is None:
            return
        manifest = self._load_manifest_file(path)
        self._load_manifest_data(manifest)


    def expand_admissible_set(self, pre_admissible_set: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        """Expands a pre-admissible set into an admissible set."""
        num_groups = len(pre_admissible_set[0])
        admissible_set_15_10 = []
        for row in pre_admissible_set:
            rotations = [[] for _ in range(num_groups)]
            for i in range(num_groups):
                x, y, z = self.TRIPLES[row[i]]
                rotations[i].append((x, y, z))
                if not x == y == z:
                    rotations[i].append((z, x, y))
                    rotations[i].append((y, z, x))
            product = list(itertools.product(*rotations))
            concatenated = [sum(xs, ()) for xs in product]
            admissible_set_15_10.extend(concatenated)
        return admissible_set_15_10


    def get_surviving_children(self, extant_elements, new_element, valid_children):
        """Returns the indices of `valid_children` that remain valid after adding `new_element` to `extant_elements`."""
        bad_triples = {(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5), (0, 6, 6), (1, 1, 1),
                    (1, 1, 2),
                    (1, 2, 2), (1, 2, 3), (1, 2, 4), (1, 3, 3), (1, 4, 4), (1, 5, 5), (1, 6, 6), (2, 2, 2),
                    (2, 3, 3),
                    (2, 4, 4), (2, 5, 5), (2, 6, 6), (3, 3, 3), (3, 3, 4), (3, 4, 4), (3, 4, 5), (3, 4, 6),
                    (3, 5, 5),
                    (3, 6, 6), (4, 4, 4), (4, 5, 5), (4, 6, 6), (5, 5, 5), (5, 5, 6), (5, 6, 6), (6, 6, 6)}

        # Compute.
        valid_indices = []
        for index, child in enumerate(valid_children):
            # Invalidate based on 2 elements from `new_element` and 1 element from a
            # potential child.
            if all(self.INT_TO_WEIGHT[x] <= self.INT_TO_WEIGHT[y]
                for x, y in zip(new_element, child)):
                continue
            # Invalidate based on 1 element from `new_element` and 2 elements from a
            # potential child.
            if all(self.INT_TO_WEIGHT[x] >= self.INT_TO_WEIGHT[y]
                for x, y in zip(new_element, child)):
                continue
            # Invalidate based on 1 element from `extant_elements`, 1 element from
            # `new_element`, and 1 element from a potential child.
            is_invalid = False
            for extant_element in extant_elements:
                if all(tuple(sorted((x, y, z))) in bad_triples
                    for x, y, z in zip(extant_element, new_element, child)):
                    is_invalid = True
                    break
            if is_invalid:
                continue

            valid_indices.append(index)
        return valid_indices


    def _evaluate_instance(self, priority: callable, dimension: int, weight: int, seed=None) -> int:
        """Generates a symmetric constant-weight admissible set I(n, w)."""
        num_groups = dimension // 3
        assert 3 * num_groups == dimension

        # Compute the scores of all valid (weight w) children.
        valid_children = []
        for child in itertools.product(range(7), repeat=num_groups):
            child_weight = sum(self.INT_TO_WEIGHT[x] for x in child)
            if child_weight == weight:
                valid_children.append(np.array(child, dtype=np.int32))

        valid_scores = np.array([
            priority(sum([self.TRIPLES[x] for x in xs], ()), dimension, weight) for xs in valid_children])

        tie_rng = np.random.default_rng(seed) if seed is not None else None

        # Greedy search guided by the scores.
        pre_admissible_set = np.empty((0, num_groups), dtype=np.int32)
        while valid_children:
            if seed is None:
                max_index = int(np.argmax(valid_scores))
            else:
                max_score = np.max(valid_scores)
                tied = np.flatnonzero(valid_scores == max_score)
                max_index = int(tie_rng.choice(tied))
            max_child = valid_children[max_index]
            surviving_indices = self.get_surviving_children(pre_admissible_set, max_child, valid_children)
            valid_children = [valid_children[i] for i in surviving_indices]
            valid_scores = valid_scores[surviving_indices]

            pre_admissible_set = np.concatenate([pre_admissible_set, max_child[None]], axis=0)

        admissible_set = np.array(self.expand_admissible_set(pre_admissible_set))

        return int(len(admissible_set) - self.Optimal_Set_Length[f"n{dimension}w{weight}"])

    def evaluate(self, priority: callable):
        if self._dataset_instances is not None:
            gaps = []
            for instance in self._dataset_instances:
                gaps.append(
                    self._evaluate_instance(
                        priority,
                        int(instance["dimension"]),
                        int(instance["weight"]),
                        instance.get("seed"),
                    )
                )
            if self._return_list:
                return gaps
            return float(np.mean(gaps))
        return self._evaluate_instance(priority, self.dimension, self.weight)


    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)


if __name__ == '__main__':
    def priority(el: tuple, n: int, w: int) -> float:
        """Design a novel algorithm to evaluate a vector for potential inclusion in a set
        Args:
            el: Candidate vectors for the admissible set.
            n: Number of dimensions and the length of a vector.
            w: Weight of each vector.

        Return:
            The priorities of `el`.
        """
        priorities = sum([abs(i) for i in el]) / n
        return priorities

    eval = ASPEvaluation()
    res = eval.evaluate_program('', priority)
    print(res)
