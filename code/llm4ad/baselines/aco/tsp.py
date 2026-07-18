"""Classical Ant System baseline for the symmetric TSP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import ACOParameters, validate_distance_matrix, weighted_choice


@dataclass(frozen=True)
class TSPACOSolution:
    tour: tuple[int, ...]
    cost: float


def _tour_cost(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    return float(np.sum(distance_matrix[tour, np.roll(tour, -1)]))


def _construct_tour(
    rng: np.random.Generator,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    parameters: ACOParameters,
) -> np.ndarray:
    n_nodes = len(pheromone)
    start = int(rng.integers(n_nodes))
    tour = np.empty(n_nodes, dtype=int)
    tour[0] = start
    unvisited = np.ones(n_nodes, dtype=bool)
    unvisited[start] = False

    for position in range(1, n_nodes):
        current = tour[position - 1]
        candidates = np.flatnonzero(unvisited)
        desirability = (
            np.power(pheromone[current, candidates], parameters.alpha)
            * np.power(heuristic[current, candidates], parameters.beta)
        )
        next_node = weighted_choice(rng, candidates, desirability)
        tour[position] = next_node
        unvisited[next_node] = False
    return tour


def _deposit_tour(pheromone: np.ndarray, tour: np.ndarray, amount: float) -> None:
    sources = tour
    targets = np.roll(tour, -1)
    np.add.at(pheromone, (sources, targets), amount)
    np.add.at(pheromone, (targets, sources), amount)


def solve_tsp_aco(
    distance_matrix: np.ndarray,
    parameters: ACOParameters | None = None,
    seed: int = 0,
) -> TSPACOSolution:
    """Solve one symmetric TSP instance with the Ant System update rule."""

    parameters = parameters or ACOParameters()
    parameters.validate()
    distance_matrix = validate_distance_matrix(distance_matrix, minimum_size=3)
    rng = np.random.default_rng(seed)

    with np.errstate(divide="ignore"):
        heuristic = np.where(distance_matrix > 0, 1.0 / distance_matrix, 0.0)
    np.fill_diagonal(heuristic, 0.0)
    pheromone = np.ones_like(distance_matrix, dtype=float)
    best_tour: np.ndarray | None = None
    best_cost = float("inf")

    for _ in range(parameters.iterations):
        ant_tours: list[np.ndarray] = []
        ant_costs: list[float] = []
        for _ant in range(parameters.ants):
            tour = _construct_tour(rng, pheromone, heuristic, parameters)
            cost = _tour_cost(tour, distance_matrix)
            ant_tours.append(tour)
            ant_costs.append(cost)
            if cost < best_cost:
                best_tour = tour.copy()
                best_cost = cost

        pheromone *= 1.0 - parameters.evaporation
        for tour, cost in zip(ant_tours, ant_costs):
            if cost > 0:
                _deposit_tour(pheromone, tour, 1.0 / cost)
        if parameters.elitist_weight and best_tour is not None and best_cost > 0:
            _deposit_tour(
                pheromone,
                best_tour,
                parameters.elitist_weight / best_cost,
            )
        np.maximum(pheromone, parameters.pheromone_floor, out=pheromone)

    if best_tour is None:
        raise RuntimeError("ACO failed to construct a TSP tour")
    return TSPACOSolution(tuple(int(node) for node in best_tour), float(best_cost))

