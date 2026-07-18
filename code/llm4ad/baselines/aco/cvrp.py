"""Capacity-feasible Ant System baseline for CVRP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import ACOParameters, validate_distance_matrix, weighted_choice


@dataclass(frozen=True)
class CVRPACOSolution:
    routes: tuple[tuple[int, ...], ...]
    cost: float


def _construct_routes(
    rng: np.random.Generator,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    parameters: ACOParameters,
) -> tuple[tuple[int, ...], ...]:
    unvisited = np.ones(len(demands), dtype=bool)
    unvisited[0] = False
    routes: list[tuple[int, ...]] = []

    while np.any(unvisited):
        remaining_capacity = capacity
        current = 0
        route = [0]
        while True:
            feasible = np.flatnonzero(unvisited & (demands <= remaining_capacity))
            if len(feasible) == 0:
                break
            desirability = (
                np.power(pheromone[current, feasible], parameters.alpha)
                * np.power(heuristic[current, feasible], parameters.beta)
            )
            next_node = weighted_choice(rng, feasible, desirability)
            route.append(next_node)
            unvisited[next_node] = False
            remaining_capacity -= int(demands[next_node])
            current = next_node
        route.append(0)
        routes.append(tuple(route))
    return tuple(routes)


def _routes_cost(routes: tuple[tuple[int, ...], ...], distance_matrix: np.ndarray) -> float:
    return float(
        sum(
            distance_matrix[source, target]
            for route in routes
            for source, target in zip(route, route[1:])
        )
    )


def _deposit_routes(
    pheromone: np.ndarray,
    routes: tuple[tuple[int, ...], ...],
    amount: float,
) -> None:
    for route in routes:
        sources = np.asarray(route[:-1], dtype=int)
        targets = np.asarray(route[1:], dtype=int)
        np.add.at(pheromone, (sources, targets), amount)
        np.add.at(pheromone, (targets, sources), amount)


def solve_cvrp_aco(
    distance_matrix: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    parameters: ACOParameters | None = None,
    seed: int = 0,
) -> CVRPACOSolution:
    """Solve one depot-at-index-zero CVRP instance with Ant System."""

    parameters = parameters or ACOParameters()
    parameters.validate()
    distance_matrix = validate_distance_matrix(distance_matrix, minimum_size=2)
    demands = np.asarray(demands, dtype=int)
    capacity = int(capacity)
    if demands.shape != (len(distance_matrix),):
        raise ValueError("demands must have one entry per distance-matrix node")
    if demands[0] != 0 or np.any(demands[1:] <= 0):
        raise ValueError("depot demand must be zero and customer demands must be positive")
    if capacity <= 0 or np.any(demands[1:] > capacity):
        raise ValueError("capacity must be positive and fit every customer")

    rng = np.random.default_rng(seed)
    with np.errstate(divide="ignore"):
        heuristic = np.where(distance_matrix > 0, 1.0 / distance_matrix, 0.0)
    np.fill_diagonal(heuristic, 0.0)
    pheromone = np.ones_like(distance_matrix, dtype=float)
    best_routes: tuple[tuple[int, ...], ...] | None = None
    best_cost = float("inf")

    for _ in range(parameters.iterations):
        ant_routes: list[tuple[tuple[int, ...], ...]] = []
        ant_costs: list[float] = []
        for _ant in range(parameters.ants):
            routes = _construct_routes(
                rng, pheromone, heuristic, demands, capacity, parameters
            )
            cost = _routes_cost(routes, distance_matrix)
            ant_routes.append(routes)
            ant_costs.append(cost)
            if cost < best_cost:
                best_routes = routes
                best_cost = cost

        pheromone *= 1.0 - parameters.evaporation
        for routes, cost in zip(ant_routes, ant_costs):
            if cost > 0:
                _deposit_routes(pheromone, routes, 1.0 / cost)
        if parameters.elitist_weight and best_routes is not None and best_cost > 0:
            _deposit_routes(
                pheromone,
                best_routes,
                parameters.elitist_weight / best_cost,
            )
        np.maximum(pheromone, parameters.pheromone_floor, out=pheromone)

    if best_routes is None:
        raise RuntimeError("ACO failed to construct CVRP routes")
    return CVRPACOSolution(best_routes, float(best_cost))

