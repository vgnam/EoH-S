from __future__ import annotations

"""Task adapter for ovrp_construct: instance loading, descriptors,
in-memory evaluation factories, and hidden ID/OOD specs.

Follows the same adapter interface as bp_1d_construct/common.py so the
shared construct-run drivers (construct_run_common.py, run_ow_cahd.py,
scripts/verify_matrix.py) can drive OVRP without task-specific code.
"""

import copy
import hashlib
from pathlib import Path

import numpy as np

from llm4ad.task.optimization.ovrp_construct import OVRPEvaluation
from post_eval_common import load_instances, resolve_repo_path


CAPACITY = 40
VALID_SIZES = (20, 50, 100, 200)


def _instance_from_coordinates(coords):
    """Derive a full OVRP instance (coords, dist, demands, capacity) from
    plain coordinates deterministically (seed from the coordinate bytes).

    The OVRP heuristic consumes (current_node, depot, unvisited_nodes,
    rest_capacity, demands, distance_matrix); demands/capacity/distances
    are derived here so LLM-synthesized regimes only need to return
    coordinates, mirroring the TSP/CVRP construct pipelines.
    """
    coords = np.asarray(coords, dtype=float)
    coords = np.clip(coords, 0.0, 1.0)
    n_nodes = len(coords)
    seed = int.from_bytes(hashlib.sha256(coords.tobytes()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    demands = np.concatenate(
        [[0], rng.integers(1, 10, size=n_nodes - 1)]
    ).astype(int)
    distance_matrix = np.linalg.norm(
        coords[:, np.newaxis] - coords, axis=2
    )
    return coords, distance_matrix, demands, int(CAPACITY)


def _validate_full_instance(instance):
    if not isinstance(instance, (tuple, list)) or len(instance) != 4:
        raise ValueError("OVRP instances must be 4-tuples.")
    coords = np.asarray(instance[0], dtype=float)
    distance_matrix = np.asarray(instance[1], dtype=float)
    demands = np.asarray(instance[2], dtype=int)
    capacity = int(instance[3])
    n_nodes = len(coords)
    if coords.ndim != 2 or coords.shape[1] != 2 or n_nodes < 3:
        raise ValueError(
            f"Expected OVRP coordinates with shape (n>=3, 2), got {coords.shape}."
        )
    if distance_matrix.shape != (n_nodes, n_nodes) or demands.shape != (n_nodes,):
        raise ValueError("OVRP distance/demand shapes do not match coordinates.")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(distance_matrix)):
        raise ValueError("OVRP coordinates and distances must be finite.")
    if np.any(demands[1:] <= 0):
        raise ValueError("OVRP customer demands must be positive.")
    if capacity < int(np.max(demands)):
        raise ValueError("OVRP capacity is invalid.")
    return coords, distance_matrix, demands, capacity


def normalize_ovrp_instance(instance):
    if isinstance(instance, (tuple, list)) and len(instance) == 4:
        return _validate_full_instance(instance)
    return _instance_from_coordinates(instance)


def is_valid_ovrp_instance(instance):
    try:
        normalize_ovrp_instance(instance)
        return True
    except Exception:
        return False


def ovrp_descriptor(instance):
    """12-dim descriptor used only by the OW-CAHD open-world controller.

    Geometry statistics of the customers plus demand/capacity statistics,
    matching the CVRP descriptor layout.
    """
    coords, distance_matrix, demands, capacity = normalize_ovrp_instance(instance)
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


def load_train_instances(paths, instances_per_dataset=None):
    merged = []
    for path in paths:
        path = resolve_repo_path(path)
        instances = load_instances(path)
        if instances_per_dataset is not None:
            instances = instances[: int(instances_per_dataset)]
        merged.extend(normalize_ovrp_instance(item) for item in instances)
    return merged


def build_wake_stream(datasets=None, dataset=None, seed=2026, batch_size=None, shuffle=False):
    paths = datasets if datasets is not None else [dataset]
    instances = load_train_instances(paths)
    if not instances:
        raise ValueError("OVRP training dataset contains no instances.")
    rng = np.random.default_rng(int(seed))
    round_size = min(int(batch_size or len(instances)), len(instances))
    while True:
        indices = rng.choice(len(instances), size=round_size, replace=False)
        yield [copy.deepcopy(instances[int(index)]) for index in indices]


def make_evaluation(instances, timeout_seconds=120, return_list=True):
    return OVRPEvaluation(
        instances=[normalize_ovrp_instance(item) for item in instances],
        timeout_seconds=timeout_seconds,
        return_list=return_list,
    )


def build_task(task_cfg):
    instances = load_train_instances(
        task_cfg["train_datasets"],
        instances_per_dataset=task_cfg.get("train_instances_per_dataset"),
    )
    return make_evaluation(
        instances,
        timeout_seconds=task_cfg.get("timeout_seconds", 120),
        return_list=task_cfg.get("return_list", True),
    )


def hidden_specs(hidden_test_cfg):
    specs = []
    for path in hidden_test_cfg.get("id_datasets", []):
        stem = Path(path).stem
        specs.append((str(resolve_repo_path(path)), f"id_{stem}"))
    for path in hidden_test_cfg.get("ood_datasets", []):
        stem = Path(path).stem
        specs.append((str(resolve_repo_path(path)), f"ood_{stem}"))
    return specs


def hidden_eval_factory(hidden_test_cfg):
    eval_instances = hidden_test_cfg.get("eval_instances")
    timeout_seconds = hidden_test_cfg.get("function_timeout_seconds") or 120

    def factory(instances, stem=None):
        if eval_instances is not None:
            instances = instances[: int(eval_instances)]
        return make_evaluation(instances, timeout_seconds=timeout_seconds)

    return factory