"""EoH-compatible ACO pheromone evaluators for TSP and CVRP."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from llm4ad.base import Evaluation

from .template import (
    CVRP_TASK_DESCRIPTION,
    CVRP_TEMPLATE_PROGRAM,
    TSP_TASK_DESCRIPTION,
    TSP_TEMPLATE_PROGRAM,
)

__all__ = ["TSPACOPheromoneEvaluation", "CVRPACOPheromoneEvaluation"]


def _weighted_choice(
    rng: np.random.Generator,
    candidates: np.ndarray,
    weights: np.ndarray,
) -> int:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    largest = float(np.max(weights)) if len(weights) else 0.0
    if largest <= 0:
        return int(rng.choice(candidates))
    probabilities = weights / largest
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0:
        return int(rng.choice(candidates))
    return int(rng.choice(candidates, p=probabilities / total))


def _validate_aco_parameters(
    n_ants: int,
    iterations: int,
    alpha: float,
    beta: float,
    rho: float,
    n_runs: int,
) -> None:
    if n_ants < 1 or iterations < 1 or n_runs < 1:
        raise ValueError("n_ants, iterations, and n_runs must be positive")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    if not 0 < rho < 1:
        raise ValueError("rho must be in (0, 1)")


def _load_dataset_instances(paths: Iterable[str | Path]) -> list[Any]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    instances: list[Any] = []
    for path in paths:
        with Path(path).open("rb") as handle:
            dataset = pickle.load(handle)
        if isinstance(dataset, dict) and "instances" in dataset:
            loaded = dataset["instances"]
        else:
            loaded = dataset
        if isinstance(loaded, dict):
            loaded = list(loaded.values())
        instances.extend(list(loaded))
    return instances


def _validate_distance_matrix(matrix: Any, size: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError("distance matrix shape does not match the instance")
    if np.any(matrix < 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("distance matrix must be finite and non-negative")
    return matrix


def _apply_update(
    update_fn: Callable,
    pheromone: np.ndarray,
    args: tuple[Any, ...],
    floor: float,
    ceiling: float,
) -> np.ndarray:
    updated = np.asarray(update_fn(pheromone.copy(), *args), dtype=float)
    if updated.shape != pheromone.shape or not np.all(np.isfinite(updated)):
        raise ValueError("pheromone update returned an invalid matrix")
    if np.any(updated < 0):
        raise ValueError("pheromone update returned negative values")
    updated = np.clip(updated, floor, ceiling)
    np.fill_diagonal(updated, floor)
    return updated


class TSPACOPheromoneEvaluation(Evaluation):
    """Evaluate one pheromone update rule on Euclidean TSP instances."""

    def __init__(
        self,
        datasets: Iterable[str | Path] | None = None,
        instances: Iterable[Any] | None = None,
        *,
        n_ants: int = 8,
        iterations: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        n_runs: int = 1,
        seed: int = 2026,
        max_instances: int | None = None,
        pheromone_floor: float = 1e-12,
        pheromone_ceiling: float = 1e12,
        timeout_seconds: int | float = 600,
        return_list: bool = True,
        safe_evaluate: bool = True,
    ):
        super().__init__(
            template_program=TSP_TEMPLATE_PROGRAM,
            task_description=TSP_TASK_DESCRIPTION,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
            safe_evaluate=safe_evaluate,
        )
        _validate_aco_parameters(n_ants, iterations, alpha, beta, rho, n_runs)
        if pheromone_floor <= 0 or pheromone_ceiling <= pheromone_floor:
            raise ValueError("pheromone bounds must satisfy 0 < floor < ceiling")
        raw = list(instances or [])
        if datasets is not None:
            raw.extend(_load_dataset_instances(datasets))
        if max_instances is not None:
            raw = raw[: int(max_instances)]
        if not raw:
            raise ValueError("TSP ACO evaluation requires at least one instance")
        self.instances = [self._normalize_instance(instance) for instance in raw]
        self.n_ants = int(n_ants)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.n_runs = int(n_runs)
        self.seed = int(seed)
        self.pheromone_floor = float(pheromone_floor)
        self.pheromone_ceiling = float(pheromone_ceiling)
        self.return_list = bool(return_list)

    @staticmethod
    def _normalize_instance(instance: Any) -> tuple[np.ndarray, np.ndarray, float]:
        if not isinstance(instance, (tuple, list)) or len(instance) != 3:
            raise ValueError("TSP ACO instances must be (coordinates, distances, baseline)")
        coordinates = np.asarray(instance[0], dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 3:
            raise ValueError("TSP coordinates must have shape (n>=3, 2)")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("TSP coordinates must be finite")
        distances = _validate_distance_matrix(instance[1], len(coordinates))
        baseline = float(instance[2])
        if baseline <= 0 or not math.isfinite(baseline):
            raise ValueError("TSP baseline must be finite and positive")
        return coordinates, distances, baseline

    @staticmethod
    def _tour_cost(tour: np.ndarray, distances: np.ndarray) -> float:
        return float(np.sum(distances[tour, np.roll(tour, -1)]))

    def _construct_tour(
        self,
        rng: np.random.Generator,
        pheromone: np.ndarray,
        heuristic: np.ndarray,
    ) -> np.ndarray:
        n = len(pheromone)
        tour = np.empty(n, dtype=int)
        tour[0] = int(rng.integers(n))
        unvisited = np.ones(n, dtype=bool)
        unvisited[tour[0]] = False
        for position in range(1, n):
            current = tour[position - 1]
            candidates = np.flatnonzero(unvisited)
            weights = (
                np.power(pheromone[current, candidates], self.alpha)
                * np.power(heuristic[current, candidates], self.beta)
            )
            tour[position] = _weighted_choice(rng, candidates, weights)
            unvisited[tour[position]] = False
        return tour

    def _run_once(
        self,
        distances: np.ndarray,
        update_fn: Callable,
        seed: int,
    ) -> float:
        with np.errstate(divide="ignore"):
            heuristic = np.where(distances > 0, 1.0 / distances, 0.0)
        np.fill_diagonal(heuristic, 0.0)
        pheromone = np.ones_like(distances, dtype=float)
        best_tour: np.ndarray | None = None
        best_cost = float("inf")
        rng = np.random.default_rng(seed)
        for iteration in range(self.iterations):
            ant_tours = [
                self._construct_tour(rng, pheromone, heuristic)
                for _ in range(self.n_ants)
            ]
            costs = np.asarray(
                [self._tour_cost(tour, distances) for tour in ant_tours],
                dtype=float,
            )
            iteration_best = int(np.argmin(costs))
            if costs[iteration_best] < best_cost:
                best_cost = float(costs[iteration_best])
                best_tour = ant_tours[iteration_best].copy()
            pheromone = _apply_update(
                update_fn,
                pheromone,
                (
                    ant_tours,
                    costs,
                    best_tour.copy(),
                    best_cost,
                    self.rho,
                    iteration,
                    self.iterations,
                ),
                self.pheromone_floor,
                self.pheromone_ceiling,
            )
        return best_cost

    def evaluate_program(
        self,
        program_str: str,
        callable_func: Callable,
    ) -> list[float] | float | None:
        scores = []
        try:
            for instance_id, (_coords, distances, baseline) in enumerate(self.instances):
                costs = [
                    self._run_once(
                        distances,
                        callable_func,
                        self.seed + instance_id * 1009 + run_id,
                    )
                    for run_id in range(self.n_runs)
                ]
                scores.append((baseline - float(np.mean(costs))) / baseline)
        except Exception:
            return None
        return scores if self.return_list else float(np.mean(scores))


class CVRPACOPheromoneEvaluation(Evaluation):
    """Evaluate one pheromone update rule on capacity-feasible CVRP ACO."""

    def __init__(
        self,
        datasets: Iterable[str | Path] | None = None,
        instances: Iterable[Any] | None = None,
        *,
        n_ants: int = 8,
        iterations: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        n_runs: int = 1,
        seed: int = 2026,
        max_instances: int | None = None,
        pheromone_floor: float = 1e-12,
        pheromone_ceiling: float = 1e12,
        timeout_seconds: int | float = 600,
        return_list: bool = True,
        safe_evaluate: bool = True,
    ):
        super().__init__(
            template_program=CVRP_TEMPLATE_PROGRAM,
            task_description=CVRP_TASK_DESCRIPTION,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
            safe_evaluate=safe_evaluate,
        )
        _validate_aco_parameters(n_ants, iterations, alpha, beta, rho, n_runs)
        if pheromone_floor <= 0 or pheromone_ceiling <= pheromone_floor:
            raise ValueError("pheromone bounds must satisfy 0 < floor < ceiling")
        raw = list(instances or [])
        if datasets is not None:
            raw.extend(_load_dataset_instances(datasets))
        if max_instances is not None:
            raw = raw[: int(max_instances)]
        if not raw:
            raise ValueError("CVRP ACO evaluation requires at least one instance")
        self.instances = [self._normalize_instance(instance) for instance in raw]
        self.n_ants = int(n_ants)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.n_runs = int(n_runs)
        self.seed = int(seed)
        self.pheromone_floor = float(pheromone_floor)
        self.pheromone_ceiling = float(pheromone_ceiling)
        self.return_list = bool(return_list)

    @staticmethod
    def _normalize_instance(
        instance: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        if not isinstance(instance, (tuple, list)) or len(instance) != 5:
            raise ValueError(
                "CVRP ACO instances must be (coordinates, distances, demands, capacity, baseline)"
            )
        coordinates = np.asarray(instance[0], dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
            raise ValueError("CVRP coordinates must have shape (n>=2, 2)")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("CVRP coordinates must be finite")
        distances = _validate_distance_matrix(instance[1], len(coordinates))
        demands = np.asarray(instance[2], dtype=int)
        capacity = int(instance[3])
        baseline = float(instance[4])
        if demands.shape != (len(coordinates),) or demands[0] != 0:
            raise ValueError("CVRP demands must align with nodes and depot demand must be zero")
        if np.any(demands[1:] <= 0) or capacity < int(np.max(demands)):
            raise ValueError("CVRP customer demands/capacity are invalid")
        if baseline <= 0 or not math.isfinite(baseline):
            raise ValueError("CVRP baseline must be finite and positive")
        return coordinates, distances, demands, capacity, baseline

    def _construct_solution(
        self,
        rng: np.random.Generator,
        pheromone: np.ndarray,
        heuristic: np.ndarray,
        demands: np.ndarray,
        capacity: int,
    ) -> list[np.ndarray]:
        unvisited = np.ones(len(demands), dtype=bool)
        unvisited[0] = False
        routes: list[np.ndarray] = []
        while np.any(unvisited):
            route = [0]
            current = 0
            remaining = capacity
            while True:
                feasible = np.flatnonzero(unvisited & (demands <= remaining))
                if len(feasible) == 0:
                    break
                weights = (
                    np.power(pheromone[current, feasible], self.alpha)
                    * np.power(heuristic[current, feasible], self.beta)
                )
                next_node = _weighted_choice(rng, feasible, weights)
                route.append(next_node)
                unvisited[next_node] = False
                remaining -= int(demands[next_node])
                current = next_node
            route.append(0)
            routes.append(np.asarray(route, dtype=int))
        return routes

    @staticmethod
    def _solution_cost(routes: list[np.ndarray], distances: np.ndarray) -> float:
        return float(
            sum(np.sum(distances[route[:-1], route[1:]]) for route in routes)
        )

    @staticmethod
    def _copy_solution(routes: list[np.ndarray]) -> list[np.ndarray]:
        return [route.copy() for route in routes]

    def _run_once(
        self,
        distances: np.ndarray,
        demands: np.ndarray,
        capacity: int,
        update_fn: Callable,
        seed: int,
    ) -> float:
        with np.errstate(divide="ignore"):
            heuristic = np.where(distances > 0, 1.0 / distances, 0.0)
        np.fill_diagonal(heuristic, 0.0)
        pheromone = np.ones_like(distances, dtype=float)
        best_solution: list[np.ndarray] | None = None
        best_cost = float("inf")
        rng = np.random.default_rng(seed)
        for iteration in range(self.iterations):
            ant_solutions = [
                self._construct_solution(rng, pheromone, heuristic, demands, capacity)
                for _ in range(self.n_ants)
            ]
            costs = np.asarray(
                [self._solution_cost(routes, distances) for routes in ant_solutions],
                dtype=float,
            )
            iteration_best = int(np.argmin(costs))
            if costs[iteration_best] < best_cost:
                best_cost = float(costs[iteration_best])
                best_solution = self._copy_solution(ant_solutions[iteration_best])
            pheromone = _apply_update(
                update_fn,
                pheromone,
                (
                    ant_solutions,
                    costs,
                    self._copy_solution(best_solution),
                    best_cost,
                    self.rho,
                    iteration,
                    self.iterations,
                ),
                self.pheromone_floor,
                self.pheromone_ceiling,
            )
        return best_cost

    def evaluate_program(
        self,
        program_str: str,
        callable_func: Callable,
    ) -> list[float] | float | None:
        scores = []
        try:
            for instance_id, (_coords, distances, demands, capacity, baseline) in enumerate(
                self.instances
            ):
                costs = [
                    self._run_once(
                        distances,
                        demands,
                        capacity,
                        callable_func,
                        self.seed + instance_id * 1009 + run_id,
                    )
                    for run_id in range(self.n_runs)
                ]
                scores.append((baseline - float(np.mean(costs))) / baseline)
        except Exception:
            return None
        return scores if self.return_list else float(np.mean(scores))
