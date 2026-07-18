from __future__ import annotations

import copy
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from generate_hidden_cvrp_dataset import (  # noqa: E402
    clarke_wright_reference,
    pairwise_distances,
    sample_demands_and_capacity,
)
from llm4ad.task.optimization.cvrp_construct_set import CVRPSEvaluation  # noqa: E402


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_full_instance(instance):
    if not isinstance(instance, (tuple, list)) or len(instance) != 5:
        raise ValueError("CVRP instances must be 5-tuples.")
    coords = np.asarray(instance[0], dtype=float)
    distance_matrix = np.asarray(instance[1], dtype=float)
    demands = np.asarray(instance[2], dtype=int)
    capacity = int(instance[3])
    baseline = float(instance[4])
    n_nodes = len(coords)
    if coords.ndim != 2 or coords.shape[1] != 2 or n_nodes < 3:
        raise ValueError(f"Expected CVRP coordinates with shape (n>=3, 2), got {coords.shape}.")
    if distance_matrix.shape != (n_nodes, n_nodes) or demands.shape != (n_nodes,):
        raise ValueError("CVRP distance/demand shapes do not match coordinates.")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(distance_matrix)):
        raise ValueError("CVRP coordinates and distances must be finite.")
    if demands[0] != 0 or np.any(demands[1:] <= 0):
        raise ValueError("CVRP depot demand must be zero and customer demands must be positive.")
    if capacity < int(np.max(demands)) or baseline <= 0 or not np.isfinite(baseline):
        raise ValueError("CVRP capacity/baseline is invalid.")
    return coords, distance_matrix, demands, capacity, baseline


def _instance_from_coordinates(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3:
        raise ValueError(f"Expected CVRP coordinate array with shape (n>=3, 2), got {coords.shape}.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("CVRP coordinates must be finite.")
    coords = np.clip(coords, 0.0, 1.0)
    seed = int.from_bytes(hashlib.sha256(coords.tobytes()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    demands, capacity = sample_demands_and_capacity(rng, len(coords) - 1)
    distance_matrix = pairwise_distances(coords)
    baseline = clarke_wright_reference(distance_matrix, demands, capacity)
    return coords, distance_matrix, demands, capacity, baseline


def normalize_cvrp_instance(instance):
    if isinstance(instance, (tuple, list)) and len(instance) == 5:
        return _validate_full_instance(instance)
    return _instance_from_coordinates(instance)


def is_valid_cvrp_instance(instance):
    try:
        normalize_cvrp_instance(instance)
        return True
    except Exception:
        return False


def load_train_dataset(path=None, paths=None):
    if paths is None:
        if path is None:
            raise ValueError("Either path or paths must be provided.")
        paths = [path]
    instances = []
    instance_families = []
    families = []
    resolved_paths = []
    for dataset_path in paths:
        dataset_path = resolve_repo_path(dataset_path)
        resolved_paths.append(str(dataset_path))
        with dataset_path.open("rb") as handle:
            dataset = pickle.load(handle)
        current_instances = dataset.get("instances", []) if isinstance(dataset, dict) else dataset
        instances.extend(normalize_cvrp_instance(item) for item in current_instances)
        if isinstance(dataset, dict):
            instance_families.extend(dataset.get("instance_families", []))
            for family in dataset.get("families", []):
                if family not in families:
                    families.append(family)
    if instance_families and len(instance_families) != len(instances):
        raise ValueError("CVRP family metadata length does not match instances.")
    return {
        "path": resolved_paths[0] if len(resolved_paths) == 1 else resolved_paths,
        "instances": instances,
        "instance_families": instance_families,
        "families": families,
    }


def build_wake_stream(dataset=None, datasets=None, seed=2026, batch_size=None, shuffle=False):
    train = load_train_dataset(path=dataset, paths=datasets)
    instances = train["instances"]
    if not instances:
        raise ValueError("CVRP training dataset contains no instances.")
    rng = np.random.default_rng(seed)
    round_size = min(int(batch_size or len(instances)), len(instances))
    while True:
        indices = rng.choice(len(instances), size=round_size, replace=False)
        yield [copy.deepcopy(instances[int(index)]) for index in indices]


def cvrp_descriptor(instance):
    coords, distance_matrix, demands, capacity, _baseline = normalize_cvrp_instance(instance)
    customers = coords[1:]
    center = np.mean(customers, axis=0)
    spread = np.std(customers, axis=0)
    customer_distances = distance_matrix[1:, 1:]
    upper = customer_distances[np.triu_indices(len(customers), k=1)]
    customer_demands = demands[1:].astype(float)
    return np.array(
        [
            center[0],
            center[1],
            spread[0],
            spread[1],
            float(np.mean(upper)),
            float(np.std(upper)),
            float(np.percentile(upper, 10)),
            float(np.percentile(upper, 90)),
            float(np.mean(customer_demands)),
            float(np.std(customer_demands)),
            float(np.sum(customer_demands) / capacity),
            float(len(customers)),
        ],
        dtype=float,
    )


class CVRPInMemoryEvaluation(CVRPSEvaluation):
    def __init__(self, instances, timeout_seconds=120, return_list=True):
        super().__init__(timeout_seconds=timeout_seconds, datasets=None, return_list=return_list)
        self._datasets = [normalize_cvrp_instance(instance) for instance in instances]
        self.n_instance = len(self._datasets)


def load_hidden_cvrp_dataset(path):
    path = Path(path)
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    if dataset.get("format") != "eohs-open-world-cvrp-hidden-v1":
        raise ValueError(f"Unsupported hidden CVRP dataset format in {path}.")
    required = {"seed", "customer_sizes", "instances_per_size", "schedule", "rounds"}
    missing = required.difference(dataset)
    if missing:
        raise ValueError(f"Hidden CVRP dataset is missing fields: {sorted(missing)}")
    return dataset


def hidden_round_instances(hidden_round):
    fields = (
        hidden_round["coordinates"],
        hidden_round["demands"],
        hidden_round["capacities"],
        hidden_round["baselines"],
    )
    lengths = {len(field) for field in fields}
    if len(lengths) != 1:
        raise ValueError("Hidden CVRP round fields have inconsistent lengths.")
    instances = []
    for coords, demands, capacity, baseline in zip(*fields):
        coords = np.asarray(coords, dtype=float)
        distance_matrix = pairwise_distances(coords)
        instances.append(
            _validate_full_instance(
                (coords, distance_matrix, np.asarray(demands, dtype=int), int(capacity), float(baseline))
            )
        )
    return instances
