"""Position-based ACO baseline for offline one-dimensional bin packing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import ACOParameters


@dataclass(frozen=True)
class BPPACOSolution:
    order: tuple[int, ...]
    bins: tuple[tuple[int, ...], ...]
    bin_count: int
    lower_bound: int


def _best_fit_decode(
    items: np.ndarray,
    order: np.ndarray,
    capacity: int,
) -> tuple[tuple[int, ...], ...]:
    residuals = np.empty(len(items), dtype=int)
    bins: list[list[int]] = []
    bin_count = 0
    for item_index in order:
        item_size = int(items[item_index])
        after = residuals[:bin_count] - item_size
        feasible = after >= 0
        if not np.any(feasible):
            bins.append([int(item_index)])
            residuals[bin_count] = capacity - item_size
            bin_count += 1
        else:
            best_bin = int(np.argmin(np.where(feasible, after, capacity + 1)))
            bins[best_bin].append(int(item_index))
            residuals[best_bin] = after[best_bin]
    return tuple(tuple(bin_items) for bin_items in bins)


def _sample_order(
    rng: np.random.Generator,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    parameters: ACOParameters,
) -> np.ndarray:
    weights = np.power(pheromone, parameters.alpha) * np.power(
        heuristic, parameters.beta
    )
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    row_sums = weights.sum(axis=1, keepdims=True)
    invalid_rows = row_sums[:, 0] <= 0
    if np.any(invalid_rows):
        weights[invalid_rows] = 1.0
        row_sums = weights.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(weights / row_sums, axis=1)
    draws = rng.random((len(pheromone), 1))
    buckets = np.sum(draws > cumulative, axis=1)
    tie_breakers = rng.random(len(pheromone))
    return np.lexsort((tie_breakers, buckets)).astype(int)


def _deposit_order(
    pheromone: np.ndarray,
    order: np.ndarray,
    amount: float,
) -> None:
    n_items, n_buckets = pheromone.shape
    bucket_by_rank = np.minimum(
        n_buckets - 1,
        np.arange(n_items, dtype=int) * n_buckets // n_items,
    )
    np.add.at(pheromone, (order, bucket_by_rank), amount)


def solve_bpp_aco(
    items: np.ndarray,
    capacity: int,
    parameters: ACOParameters | None = None,
    seed: int = 0,
    position_buckets: int = 16,
) -> BPPACOSolution:
    """Solve offline BPP by learning an item order and applying Best Fit.

    This is intentionally an offline baseline: repeated ants inspect and reorder
    the complete item sequence. It must not be reported as an online OBP policy.
    """

    parameters = parameters or ACOParameters()
    parameters.validate()
    items = np.asarray(items, dtype=int)
    capacity = int(capacity)
    if items.ndim != 1 or len(items) == 0:
        raise ValueError("items must be a non-empty one-dimensional array")
    if capacity <= 0 or np.any(items <= 0) or np.any(items > capacity):
        raise ValueError("every item must be positive and no larger than capacity")
    if position_buckets < 2:
        raise ValueError("position_buckets must be at least 2")

    n_items = len(items)
    n_buckets = min(int(position_buckets), n_items)
    lower_bound = max(1, int(np.ceil(np.sum(items) / capacity)))
    rng = np.random.default_rng(seed)

    # Seed the incumbent with Best Fit Decreasing, a strong deterministic order.
    best_order = np.argsort(-items, kind="stable").astype(int)
    best_bins = _best_fit_decode(items, best_order, capacity)
    best_count = len(best_bins)

    pheromone = np.ones((n_items, n_buckets), dtype=float)
    item_ratios = items.astype(float) / capacity
    target_ratios = 1.0 - (np.arange(n_buckets, dtype=float) + 0.5) / n_buckets
    heuristic = 1.0 / (np.abs(item_ratios[:, None] - target_ratios[None, :]) + 0.05)

    for _ in range(parameters.iterations):
        ant_orders: list[np.ndarray] = []
        ant_counts: list[int] = []
        for _ant in range(parameters.ants):
            order = _sample_order(rng, pheromone, heuristic, parameters)
            bins = _best_fit_decode(items, order, capacity)
            count = len(bins)
            ant_orders.append(order)
            ant_counts.append(count)
            if count < best_count:
                best_order = order.copy()
                best_bins = bins
                best_count = count

        pheromone *= 1.0 - parameters.evaporation
        for order, count in zip(ant_orders, ant_counts):
            _deposit_order(pheromone, order, lower_bound / count)
        if parameters.elitist_weight:
            _deposit_order(
                pheromone,
                best_order,
                parameters.elitist_weight * lower_bound / best_count,
            )
        np.maximum(pheromone, parameters.pheromone_floor, out=pheromone)

    return BPPACOSolution(
        order=tuple(int(index) for index in best_order),
        bins=best_bins,
        bin_count=int(best_count),
        lower_bound=int(lower_bound),
    )
