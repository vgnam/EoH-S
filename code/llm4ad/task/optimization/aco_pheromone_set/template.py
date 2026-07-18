"""Heuristic templates for EoH-S/OW-CAHD-guided ACO."""

TSP_TEMPLATE_PROGRAM = '''import numpy as np

def update_pheromone(pheromone: np.ndarray, ant_tours: list,
                     tour_costs: np.ndarray, best_tour: np.ndarray,
                     best_cost: float, rho: float, iteration: int,
                     max_iterations: int) -> np.ndarray:
    """Update TSP pheromone after one complete ant-colony iteration."""
    updated = (1.0 - rho) * pheromone
    n = pheromone.shape[0]
    for tour, cost in zip(ant_tours, tour_costs):
        if cost <= 0:
            continue
        amount = 1.0 / float(cost)
        for index in range(n):
            source = int(tour[index])
            target = int(tour[(index + 1) % n])
            updated[source, target] += amount
            updated[target, source] += amount
    return updated
'''

TSP_TASK_DESCRIPTION = '''
Design the pheromone-update heuristic inside Ant Colony Optimisation for a
symmetric Euclidean Traveling Salesman Problem. The ACO harness fixes ant tour
construction: transition desirability is pheromone**alpha times
(1/distance)**beta. Your function is called once after all ants finish an
iteration. Return a finite, non-negative pheromone matrix with exactly the same
shape. Lower-cost tours are better. You may use iteration progress, all ant
tours, the global-best tour, rank weighting, elitism, adaptive evaporation, or
MMAS-style bounds. Do not change the function signature and do not perform file,
network, subprocess, or global-random-state operations.
'''


CVRP_TEMPLATE_PROGRAM = '''import numpy as np

def update_pheromone(pheromone: np.ndarray, ant_solutions: list,
                     solution_costs: np.ndarray, best_solution: list,
                     best_cost: float, rho: float, iteration: int,
                     max_iterations: int) -> np.ndarray:
    """Update CVRP pheromone after one complete ant-colony iteration."""
    updated = (1.0 - rho) * pheromone
    for routes, cost in zip(ant_solutions, solution_costs):
        if cost <= 0:
            continue
        amount = 1.0 / float(cost)
        for route in routes:
            for source, target in zip(route[:-1], route[1:]):
                source = int(source)
                target = int(target)
                updated[source, target] += amount
                updated[target, source] += amount
    return updated
'''

CVRP_TASK_DESCRIPTION = '''
Design the pheromone-update heuristic inside Ant Colony Optimisation for a
symmetric Euclidean Capacitated Vehicle Routing Problem. Node 0 is the depot.
The ACO harness fixes capacity-feasible construction: an ant can select only an
unvisited customer whose demand fits its remaining vehicle capacity, and every
route starts and ends at the depot. Each ant solution is a list of such routes.
Return a finite, non-negative pheromone matrix with exactly the same shape.
Lower total route cost is better. You may use iteration progress, ranked ant
solutions, global-best reinforcement, adaptive evaporation, route structure,
or MMAS-style bounds. Do not change the function signature and do not perform
file, network, subprocess, or global-random-state operations.
'''
