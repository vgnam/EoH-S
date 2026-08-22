from __future__ import annotations

"""Task adapter for vrptw_construct: instance loading, descriptors,
in-memory evaluation factories, and hidden ID/OOD specs.

Follows the same adapter interface as bp_1d_construct/common.py so the
shared construct-run drivers (construct_run_common.py, run_ow_cahd.py,
scripts/verify_matrix.py) can drive VRPTW (capacitated VRP with time
windows) without task-specific code.
"""

import copy
import hashlib
from pathlib import Path

import numpy as np

from llm4ad.task.optimization.vrptw_construct import VRPTWEvaluation
from post_eval_common import load_instances, resolve_repo_path


CAPACITY = 40
MAX_TIME = 4.6


def _instance_from_coordinates(coords):
    """Derive a full VRPTW instance (coords, distances, demands, capacity,
    service_times, time_windows) from plain coordinates deterministically
    (seed from the coordinate bytes).

    Mirrors vrptw_construct.get_instance.GetData so LLM-synthesized regimes
    only need to return coordinates, like the TSP/CVRP construct pipelines.
    """
    coords = np.asarray(coords, dtype=float)
    coords = np.clip(coords, 0.0, 1.0)
    n_nodes = len(coords)
    seed = int.from_bytes(hashlib.sha256(coords.tobytes()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    demands = np.concatenate(
        [[0], rng.integers(1, 10, size=n_nodes - 1)]
    ).astype(int)
    capacity = int(CAPACITY)
    distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
    node_service_time = rng.random(n_nodes - 1) * 0.05 + 0.15
    service_time = np.concatenate([[0.0], node_service_time])
    node_length_tw = rng.random(n_nodes - 1) * 0.05 + 0.15
    d0i = distances[0][1:]
    ei = (
        rng.random(n_nodes - 1)
        * (((MAX_TIME - node_service_time - node_length_tw) / d0i - 1) - 1)
        + 1
    )
    node_early_tw = np.multiply(ei, d0i)
    node_late_tw = node_early_tw + node_length_tw
    time_windows = np.concatenate(
        [
            np.array([[0.0, MAX_TIME]]),
            np.concatenate([node_early_tw[:, None], node_late_tw[:, None]], axis=1),
        ],
        axis=0,
    )
    return coords, distances, demands, capacity, service_time, time_windows


def _validate_full_instance(instance):
    if not isinstance(instance, (tuple, list)) or len(instance) != 6:
        raise ValueError("VRPTW instances must be 6-tuples.")
    coords = np.asarray(instance[0], dtype=float)
    distance_matrix = np.asarray(instance[1], dtype=float)
    demands = np.asarray(instance[2], dtype=int)
    capacity = int(instance[3])
    service_time = np.asarray(instance[4], dtype=float)
    time_windows = np.asarray(instance[5], dtype=float)
    n_nodes = len(coords)
    if coords.ndim != 2 or coords.shape[1] != 2 or n_nodes < 3:
        raise ValueError(
            f"Expected VRPTW coordinates with shape (n>=3, 2), got {coords.shape}."
        )
    if (
        distance_matrix.shape != (n_nodes, n_nodes)
        or demands.shape != (n_nodes,)
        or service_time.shape != (n_nodes,)
        or time_windows.shape != (n_nodes, 2)
    ):
        raise ValueError("VRPTW field shapes do not match coordinates.")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(distance_matrix)):
        raise ValueError("VRPTW coordinates and distances must be finite.")
    if demands[0] != 0 or np.any(demands[1:] <= 0):
        raise ValueError("VRPTW depot demand must be zero and customer demands positive.")
    if service_time[0] != 0 or np.any(service_time[1:] < 0):
        raise ValueError("VRPTW depot service time must be zero.")
    if np.any(time_windows[:, 1] <= time_windows[:, 0]):
        raise ValueError("VRPTW time windows must have late > early.")
    if capacity < int(np.max(demands)):
        raise ValueError("VRPTW capacity is invalid.")
    return coords, distance_matrix, demands, capacity, service_time, time_windows


def normalize_vrptw_instance(instance):
    if isinstance(instance, (tuple, list)) and len(instance) == 6:
        return _validate_full_instance(instance)
    return _instance_from_coordinates(instance)


def is_valid_vrptw_instance(instance):
    try:
        normalize_vrptw_instance(instance)
        return True
    except Exception:
        return False


def vrptw_descriptor(instance):
    """Descriptor used only by the OW-CAHD open-world controller.

    Geometry statistics of the customers, demand/capacity statistics, and
    time-window statistics (width and position relative to the horizon).
    """
    (
        coords,
        distance_matrix,
        demands,
        capacity,
        service_time,
        time_windows,
    ) = normalize_vrptw_instance(instance)
    customers = coords[1:]
    center = np.mean(customers, axis=0)
    spread = np.std(customers, axis=0)
    customer_distances = distance_matrix[1:, 1:]
    upper = customer_distances[np.triu_indices(len(customers), k=1)]
    customer_demands = demands[1:].astype(float)
    customer_windows = time_windows[1:]
    widths = customer_windows[:, 1] - customer_windows[:, 0]
    midpoints = (customer_windows[:, 0] + customer_windows[:, 1]) / 2.0
    horizon = float(time_windows[0, 1])
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
            float(np.mean(widths)),
            float(np.mean(midpoints) / horizon if horizon > 0 else 0.0),
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
        merged.extend(normalize_vrptw_instance(item) for item in instances)
    return merged


def build_wake_stream(datasets=None, dataset=None, seed=2026, batch_size=None, shuffle=False):
    paths = datasets if datasets is not None else [dataset]
    instances = load_train_instances(paths)
    if not instances:
        raise ValueError("VRPTW training dataset contains no instances.")
    rng = np.random.default_rng(int(seed))
    round_size = min(int(batch_size or len(instances)), len(instances))
    while True:
        indices = rng.choice(len(instances), size=round_size, replace=False)
        yield [copy.deepcopy(instances[int(index)]) for index in indices]


def make_evaluation(instances, timeout_seconds=120, return_list=True):
    return VRPTWEvaluation(
        instances=[normalize_vrptw_instance(item) for item in instances],
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
