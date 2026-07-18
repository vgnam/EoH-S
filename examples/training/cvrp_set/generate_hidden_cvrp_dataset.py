from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
TSP_SCRIPT_DIR = SCRIPT_DIR.parent / "tsp_set"
if str(TSP_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TSP_SCRIPT_DIR))

from generate_hidden_tsp_dataset import (  # noqa: E402
    MIXED_ID_REGIME,
    MIXED_OOD_REGIME,
    OOD_FAMILIES,
    TRAIN_FAMILIES,
    regime_pool,
    sample_hidden_regime,
)


DEMAND_VALUES = np.arange(1, 10, dtype=int)
DEMAND_PROBABILITIES = np.linspace(5.0, 1.0, len(DEMAND_VALUES))
DEMAND_PROBABILITIES /= DEMAND_PROBABILITIES.sum()


def pairwise_distances(coords):
    coords = np.asarray(coords, dtype=float)
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


def route_cost(distance_matrix, customers):
    if not customers:
        return 0.0
    route = [0, *customers, 0]
    return float(
        sum(distance_matrix[route[idx], route[idx + 1]] for idx in range(len(route) - 1))
    )


def improve_route_2opt(distance_matrix, customers):
    route = list(customers)
    if len(route) < 3:
        return route
    improved = True
    while improved:
        improved = False
        best_cost = route_cost(distance_matrix, route)
        for left in range(len(route) - 1):
            for right in range(left + 1, len(route)):
                candidate = route[:left] + list(reversed(route[left : right + 1])) + route[right + 1 :]
                candidate_cost = route_cost(distance_matrix, candidate)
                if candidate_cost + 1e-12 < best_cost:
                    route = candidate
                    best_cost = candidate_cost
                    improved = True
        # Restart after a full improving pass so route endpoints are reconsidered.
    return route


def clarke_wright_reference(distance_matrix, demands, capacity):
    """Return a deterministic feasible CVRP reference from savings + route 2-opt."""
    demands = np.asarray(demands, dtype=int)
    capacity = int(capacity)
    customers = list(range(1, len(demands)))
    routes = {customer: [customer] for customer in customers}
    loads = {customer: int(demands[customer]) for customer in customers}
    route_of = {customer: customer for customer in customers}
    savings = sorted(
        (
            (
                float(distance_matrix[0, left] + distance_matrix[0, right] - distance_matrix[left, right]),
                left,
                right,
            )
            for left in customers
            for right in range(left + 1, len(demands))
        ),
        reverse=True,
    )

    for _saving, left, right in savings:
        left_key = route_of[left]
        right_key = route_of[right]
        if left_key == right_key or loads[left_key] + loads[right_key] > capacity:
            continue
        left_route = routes[left_key]
        right_route = routes[right_key]
        if left not in (left_route[0], left_route[-1]) or right not in (right_route[0], right_route[-1]):
            continue
        if left_route[-1] != left:
            left_route = list(reversed(left_route))
        if right_route[0] != right:
            right_route = list(reversed(right_route))
        merged = left_route + right_route
        routes[left_key] = merged
        loads[left_key] += loads[right_key]
        for customer in right_route:
            route_of[customer] = left_key
        del routes[right_key]
        del loads[right_key]

    improved_routes = [improve_route_2opt(distance_matrix, route) for route in routes.values()]
    return float(sum(route_cost(distance_matrix, route) for route in improved_routes))


def sample_demands_and_capacity(rng, n_customers):
    customer_demands = rng.choice(
        DEMAND_VALUES,
        size=int(n_customers),
        p=DEMAND_PROBABILITIES,
    ).astype(int)
    demands = np.concatenate(([0], customer_demands))
    target_routes = max(2, int(round(math.sqrt(n_customers))))
    capacity = max(
        int(np.max(customer_demands)),
        int(math.ceil(float(np.sum(customer_demands)) * 1.12 / target_routes)),
    )
    return demands, capacity


def make_cvrp_instance(customer_coords, rng, *, demands=None, capacity=None, baseline=None):
    customer_coords = np.clip(np.asarray(customer_coords, dtype=float), 0.0, 1.0)
    depot = rng.uniform(0.45, 0.55, size=(1, 2))
    coords = np.vstack([depot, customer_coords])
    if demands is None or capacity is None:
        demands, capacity = sample_demands_and_capacity(rng, len(customer_coords))
    demands = np.asarray(demands, dtype=int)
    capacity = int(capacity)
    distance_matrix = pairwise_distances(coords)
    if baseline is None:
        baseline = clarke_wright_reference(distance_matrix, demands, capacity)
    return coords, distance_matrix, demands, capacity, float(baseline)


def _instance_regimes(rng, regime, instances_per_size):
    pool = regime_pool(regime)
    if regime in (MIXED_ID_REGIME, MIXED_OOD_REGIME):
        regimes = [pool[idx % len(pool)] for idx in range(instances_per_size)]
        rng.shuffle(regimes)
        return regimes
    return [regime] * instances_per_size


def generate_hidden_dataset(seed, customer_sizes, instances_per_size, schedule=None):
    schedule = list(schedule or [MIXED_ID_REGIME, MIXED_OOD_REGIME])
    rng = np.random.default_rng(seed)
    rounds = []
    for round_id, regime in enumerate(schedule):
        coordinates = []
        demands = []
        capacities = []
        baselines = []
        instance_regimes = []
        for n_customers in customer_sizes:
            for instance_regime in _instance_regimes(rng, regime, instances_per_size):
                customer_coords = sample_hidden_regime(rng, n_customers, instance_regime)
                instance = make_cvrp_instance(customer_coords, rng)
                coordinates.append(instance[0])
                demands.append(instance[2])
                capacities.append(instance[3])
                baselines.append(instance[4])
                instance_regimes.append(instance_regime)
        rounds.append(
            {
                "round_id": round_id,
                "regime": regime,
                "instance_regimes": instance_regimes,
                "coordinates": coordinates,
                "demands": demands,
                "capacities": capacities,
                "baselines": baselines,
            }
        )
    return {
        "format": "eohs-open-world-cvrp-hidden-v1",
        "seed": int(seed),
        "customer_sizes": [int(size) for size in customer_sizes],
        "instances_per_size": int(instances_per_size),
        "schedule": schedule,
        "demand_values": DEMAND_VALUES.tolist(),
        "demand_probabilities": DEMAND_PROBABILITIES.tolist(),
        "capacity_policy": "ceil(1.12 * total_customer_demand / round(sqrt(n_customers)))",
        "rounds": rounds,
    }


def generate_train_dataset(seed, n_customers, instances_per_family, families=None):
    families = list(families or TRAIN_FAMILIES)
    rng = np.random.default_rng(seed)
    instances = []
    instance_families = []
    for family in families:
        for _ in range(instances_per_family):
            customer_coords = sample_hidden_regime(rng, n_customers, family)
            instances.append(make_cvrp_instance(customer_coords, rng))
            instance_families.append(family)
    return {
        "format": "eohs-open-world-cvrp-train-v1",
        "seed": int(seed),
        "n_customers": int(n_customers),
        "families": families,
        "instances_per_family": int(instances_per_family),
        "instances": instances,
        "instance_families": instance_families,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["hidden", "train"], default="hidden")
    parser.add_argument("--seed", type=int, default=22026)
    parser.add_argument("--customer-sizes", type=int, nargs="+", default=[100])
    parser.add_argument("--instances-per-size", type=int, default=128)
    parser.add_argument("--schedule", nargs="+", default=[MIXED_ID_REGIME, MIXED_OOD_REGIME])
    parser.add_argument("--n-customers", type=int, default=30)
    parser.add_argument("--instances-per-family", type=int, default=32)
    parser.add_argument("--families", nargs="+", default=TRAIN_FAMILIES)
    args = parser.parse_args()

    if args.mode == "train":
        dataset = generate_train_dataset(
            args.seed,
            args.n_customers,
            args.instances_per_family,
            families=args.families,
        )
        total_instances = len(dataset["instances"])
    else:
        dataset = generate_hidden_dataset(
            args.seed,
            args.customer_sizes,
            args.instances_per_size,
            schedule=args.schedule,
        )
        total_instances = sum(len(item["coordinates"]) for item in dataset["rounds"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {args.output} format={dataset['format']} total_instances={total_instances}")


if __name__ == "__main__":
    main()
