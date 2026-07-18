"""Shared configuration and numerical helpers for ACO baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ACOParameters:
    """Parameters shared by the three Ant System implementations.

    The defaults follow the scale used by the ACO example in FeiLiu36/EoH.
    ``elitist_weight=0`` gives the original Ant System update. A positive value
    additionally reinforces the global-best solution.
    """

    ants: int = 20
    iterations: int = 100
    alpha: float = 1.0
    beta: float = 2.0
    evaporation: float = 0.1
    elitist_weight: float = 0.0
    pheromone_floor: float = 1e-12

    def validate(self) -> None:
        if self.ants < 1 or self.iterations < 1:
            raise ValueError("ants and iterations must both be positive")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if not 0 < self.evaporation < 1:
            raise ValueError("evaporation must be in (0, 1)")
        if self.elitist_weight < 0:
            raise ValueError("elitist_weight must be non-negative")
        if self.pheromone_floor <= 0:
            raise ValueError("pheromone_floor must be positive")


def weighted_choice(
    rng: np.random.Generator,
    candidates: np.ndarray,
    weights: np.ndarray,
) -> int:
    """Sample one candidate while safely handling underflow and invalid weights."""

    candidates = np.asarray(candidates, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if candidates.ndim != 1 or len(candidates) == 0 or weights.shape != candidates.shape:
        raise ValueError("candidates and weights must be non-empty one-dimensional arrays")

    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    largest = float(np.max(weights))
    if largest <= 0:
        return int(rng.choice(candidates))
    probabilities = weights / largest
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0:
        return int(rng.choice(candidates))
    return int(rng.choice(candidates, p=probabilities / total))


def validate_distance_matrix(distance_matrix: np.ndarray, minimum_size: int) -> np.ndarray:
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < minimum_size:
        raise ValueError(f"distance_matrix must be square with at least {minimum_size} nodes")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("distance_matrix must contain finite non-negative values")
    return matrix

