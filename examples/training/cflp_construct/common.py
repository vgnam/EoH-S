from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from llm4ad.task.optimization.cflp_construct import CFLPEvaluation
from post_eval_common import load_instances, resolve_repo_path


def normalize_cflp_instance(instance):
    if not isinstance(instance, dict):
        raise ValueError("CFLP instance must be a dictionary.")
    capacities = [int(value) for value in instance["facility_capacities"]]
    demands = [int(value) for value in instance["customer_demands"]]
    costs = np.asarray(instance["assignment_costs"], dtype=int)
    if not capacities or not demands or costs.shape != (len(capacities), len(demands)):
        raise ValueError("Invalid CFLP dimensions.")
    if min(capacities) <= 0 or min(demands) <= 0 or np.any(costs < 0):
        raise ValueError("CFLP capacities/demands must be positive and costs nonnegative.")
    if sum(capacities) < sum(demands):
        raise ValueError("CFLP instance has insufficient total capacity.")
    return {
        "facility_capacities": capacities,
        "customer_demands": demands,
        "assignment_costs": costs.tolist(),
    }


def is_valid_cflp_instance(instance):
    try:
        normalize_cflp_instance(instance)
        return True
    except Exception:
        return False


def cflp_descriptor(instance):
    item = normalize_cflp_instance(instance)
    capacities = np.asarray(item["facility_capacities"], dtype=float)
    demands = np.asarray(item["customer_demands"], dtype=float)
    costs = np.asarray(item["assignment_costs"], dtype=float)
    return np.asarray([
        len(capacities), len(demands),
        np.sum(demands) / np.sum(capacities),
        np.mean(capacities), np.std(capacities),
        np.mean(demands), np.std(demands),
        np.mean(costs), np.std(costs),
        np.mean(np.min(costs, axis=0)),
    ], dtype=float)


def load_train_instances(paths, instances_per_dataset=None):
    merged = []
    for path in paths:
        instances = load_instances(resolve_repo_path(path))
        if instances_per_dataset is not None:
            instances = instances[: int(instances_per_dataset)]
        merged.extend(normalize_cflp_instance(item) for item in instances)
    return merged


def build_wake_stream(datasets=None, dataset=None, seed=2026, batch_size=None, shuffle=True):
    paths = datasets if datasets is not None else [dataset]
    instances = load_train_instances(paths)
    if not instances:
        raise ValueError("CFLP training dataset contains no instances.")
    rng = np.random.default_rng(int(seed))
    size = min(int(batch_size or len(instances)), len(instances))
    while True:
        indices = rng.choice(len(instances), size=size, replace=False)
        yield [copy.deepcopy(instances[int(index)]) for index in indices]


def make_evaluation(instances, timeout_seconds=600, return_list=True):
    return CFLPEvaluation(
        instances=[normalize_cflp_instance(item) for item in instances],
        timeout_seconds=timeout_seconds,
        return_list=return_list,
    )


def build_task(task_cfg):
    instances = load_train_instances(task_cfg["train_datasets"], task_cfg.get("train_instances_per_dataset"))
    return make_evaluation(instances, task_cfg.get("timeout_seconds", 600), task_cfg.get("return_list", True))


def hidden_specs(hidden_test_cfg):
    specs = []
    for label in ("id", "ood"):
        for path in hidden_test_cfg.get(f"{label}_datasets", []):
            specs.append((str(resolve_repo_path(path)), f"{label}_{Path(path).stem}"))
    return specs


def hidden_eval_factory(hidden_test_cfg):
    limit = hidden_test_cfg.get("eval_instances")
    timeout = hidden_test_cfg.get("function_timeout_seconds") or 600
    def factory(instances, stem=None):
        selected = instances[: int(limit)] if limit is not None else instances
        return make_evaluation(selected, timeout_seconds=timeout, return_list=True)
    return factory
