from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from .types import DescriptorFn, Instance, Regime


def bootstrap_regime_from_instances(
    name: str,
    instances: list[Instance],
    descriptor: DescriptorFn,
    *,
    description: str = "bootstrap empirical regime",
    covariance_reg: float = 1e-6,
) -> Regime:
    """Create an empirical regime by resampling observed instances."""

    if not instances:
        raise ValueError("Cannot bootstrap a regime from an empty instance list.")
    archive = list(instances)

    def generator(n: int, seed: int) -> list[Instance]:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(archive), size=n)
        return [copy.deepcopy(archive[int(i)]) for i in indices]

    regime = Regime(
        name=name,
        generator=generator,
        description=description,
        archive=archive,
    )
    fit_regime_observation_model(regime, descriptor, covariance_reg=covariance_reg)
    return regime


def fit_regime_observation_model(
    regime: Regime,
    descriptor: DescriptorFn,
    *,
    covariance_reg: float,
    samples: Optional[list[Instance]] = None,
) -> None:
    instances = samples if samples is not None else regime.archive
    if not instances:
        instances = regime.sample(1, seed=0)

    matrix = np.vstack([np.asarray(descriptor(instance), dtype=float).ravel() for instance in instances])
    regime.mu = np.mean(matrix, axis=0)
    if len(matrix) <= 1:
        cov = np.eye(matrix.shape[1], dtype=float)
    else:
        cov = np.cov(matrix, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
    regime.cov = np.asarray(cov, dtype=float) + covariance_reg * np.eye(matrix.shape[1])
    regime.n_fit = len(matrix)

