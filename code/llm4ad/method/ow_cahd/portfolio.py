from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np

from ...base import Function


def greedy_portfolio(candidates: list[Function], weights: np.ndarray, portfolio_size: int) -> list[Function]:
    if not candidates:
        return []
    n_instances = len(weights)
    valid: list[tuple[Function, np.ndarray]] = []
    for func in candidates:
        vector = score_vector(func, n_instances)
        if vector is not None and np.all(np.isfinite(vector)):
            valid.append((func, vector))
    if not valid:
        return []

    score_matrix = np.vstack([vector for _, vector in valid])
    mins = np.min(score_matrix, axis=0)
    maxs = np.max(score_matrix, axis=0)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    utilities = (score_matrix - mins) / denom

    selected: list[int] = []
    covered = np.zeros(n_instances, dtype=float)
    for _ in range(min(portfolio_size, len(valid))):
        best_idx = None
        best_value = -math.inf
        for idx in range(len(valid)):
            if idx in selected:
                continue
            candidate_cover = np.maximum(covered, utilities[idx])
            value = float(np.sum(weights * candidate_cover))
            if value > best_value:
                best_value = value
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        covered = np.maximum(covered, utilities[best_idx])
    return [copy.deepcopy(valid[idx][0]) for idx in selected]


def score_vector(func: Function, n_instances: int) -> Optional[np.ndarray]:
    score = func.score
    if score is None:
        return None
    if isinstance(score, (list, tuple, np.ndarray)):
        vector = np.asarray(score, dtype=float).ravel()
        if len(vector) != n_instances:
            return None
        return vector
    if math.isfinite(float(score)):
        return np.full(n_instances, float(score), dtype=float)
    return None

