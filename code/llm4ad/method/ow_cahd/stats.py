from __future__ import annotations

import math

import numpy as np

from .types import DescriptorFn, Instance, Regime

try:
    from scipy.stats import chi2
except Exception:  # pragma: no cover - scipy is in requirements, keep fallback robust.
    chi2 = None


def mean_descriptor(instances: list[Instance], descriptor: DescriptorFn) -> np.ndarray:
    if not instances:
        raise ValueError("wake batch must contain at least one instance")
    return np.mean(
        np.vstack([np.asarray(descriptor(instance), dtype=float).ravel() for instance in instances]),
        axis=0,
    )


def mahalanobis_sq(phi: np.ndarray, regime: Regime) -> float:
    if regime.mixture_mus and len(regime.mixture_mus) == len(regime.mixture_covs):
        distances = [
            _mahalanobis_sq(phi, mu, cov)
            for mu, cov in zip(regime.mixture_mus, regime.mixture_covs)
        ]
        return min(distances) if distances else math.inf
    if regime.mu is None or regime.cov is None:
        return math.inf
    return _mahalanobis_sq(phi, regime.mu, regime.cov)


def gaussian_logpdf(phi: np.ndarray, regime: Regime) -> float:
    if regime.mixture_mus and len(regime.mixture_mus) == len(regime.mixture_covs):
        component_count = len(regime.mixture_mus)
        weights = regime.mixture_weights
        if weights is None or len(weights) != component_count:
            weights = np.full(component_count, 1.0 / component_count, dtype=float)
        else:
            weights = normalize(np.asarray(weights, dtype=float))
        component_logs = np.asarray(
            [
                _gaussian_logpdf(phi, mu, cov) + math.log(max(float(weight), 1e-300))
                for mu, cov, weight in zip(regime.mixture_mus, regime.mixture_covs, weights)
            ],
            dtype=float,
        )
        finite = component_logs[np.isfinite(component_logs)]
        if len(finite) == 0:
            return -math.inf
        maximum = float(np.max(finite))
        return maximum + math.log(float(np.sum(np.exp(finite - maximum))))
    if regime.mu is None or regime.cov is None:
        return -math.inf
    return _gaussian_logpdf(phi, regime.mu, regime.cov)


def _mahalanobis_sq(phi: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    delta = phi - mu
    inv_cov = np.linalg.pinv(cov)
    return float(delta.T @ inv_cov @ delta)


def _gaussian_logpdf(phi: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    d = len(phi)
    delta = phi - mu
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -math.inf
    inv_cov = np.linalg.pinv(cov)
    return float(-0.5 * (d * math.log(2.0 * math.pi) + logdet + delta.T @ inv_cov @ delta))


def normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(values, dtype=float) / len(values)
    return values / total


def novelty_threshold(descriptor_dim: int, alpha: float) -> float:
    if chi2 is None:
        z = 2.3263478740408408 if alpha <= 0.01 else 1.6448536269514722
        return descriptor_dim * (1.0 - 2.0 / (9.0 * descriptor_dim) + z * math.sqrt(2.0 / (9.0 * descriptor_dim))) ** 3
    return float(chi2.ppf(1.0 - alpha, descriptor_dim))


def rbf_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    xy = np.vstack([x, y])
    sq_dists = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
    positive = sq_dists[sq_dists > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    gamma = 1.0 / max(2.0 * bandwidth, 1e-12)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-gamma * np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())


def mmd_pvalue(x: np.ndarray, y: np.ndarray, permutations: int, seed: int) -> tuple[float, float]:
    observed = rbf_mmd2(x, y)
    if permutations <= 0:
        return observed, 0.0

    rng = np.random.default_rng(seed)
    combined = np.vstack([x, y])
    nx = len(x)
    exceed = 0
    for _ in range(permutations):
        perm = rng.permutation(len(combined))
        x_perm = combined[perm[:nx]]
        y_perm = combined[perm[nx:]]
        if rbf_mmd2(x_perm, y_perm) >= observed:
            exceed += 1
    pvalue = (exceed + 1.0) / (permutations + 1.0)
    return observed, float(pvalue)
