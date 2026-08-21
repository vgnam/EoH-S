from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from llm4ad.task.optimization.set_cover_construct import SCPEvaluation
from post_eval_common import load_instances, resolve_repo_path


def normalize_scp_instance(instance):
    if not isinstance(instance, (list, tuple)) or len(instance) != 2:
        raise ValueError("Set-cover instance must be (universal_set, subsets).")
    universal_set, subsets = instance
    universe = [int(value) for value in universal_set]
    normalized = [[int(value) for value in subset] for subset in subsets]
    universe_values = set(universe)
    if not universe or not normalized or any(not set(subset) <= universe_values for subset in normalized):
        raise ValueError("Invalid set-cover universe/subsets.")
    if set().union(*(set(subset) for subset in normalized)) != universe_values:
        raise ValueError("Set-cover subsets do not cover the universe.")
    return universe, normalized


def is_valid_scp_instance(instance):
    try:
        normalize_scp_instance(instance)
        return True
    except Exception:
        return False


def normalize_scp_instances(instances):
    normalized = []
    for instance in instances:
        try:
            normalized.append(normalize_scp_instance(instance))
        except (KeyError, TypeError, ValueError):
            continue
    if not normalized:
        raise ValueError("Set-cover dataset contains no feasible instances.")
    return normalized


def scp_descriptor(instance):
    universe, subsets = normalize_scp_instance(instance)
    sizes = np.asarray([len(set(subset)) for subset in subsets], dtype=float)
    frequencies = np.asarray([
        sum(element in subset for subset in subsets) for element in universe
    ], dtype=float)
    density = float(np.sum(sizes) / (len(universe) * len(subsets)))
    return np.asarray([
        len(universe),
        len(subsets),
        density,
        np.mean(sizes),
        np.std(sizes),
        np.mean(frequencies),
        np.std(frequencies),
        np.min(frequencies),
        np.max(frequencies),
    ], dtype=float)


def load_train_instances(paths, instances_per_dataset=None):
    merged = []
    for path in paths:
        instances = load_instances(resolve_repo_path(path))
        if instances_per_dataset is not None:
            instances = instances[: int(instances_per_dataset)]
        merged.extend(normalize_scp_instances(instances))
    return merged


def build_wake_stream(datasets=None, dataset=None, seed=2026, batch_size=None, shuffle=True):
    paths = datasets if datasets is not None else [dataset]
    instances = load_train_instances(paths)
    if not instances:
        raise ValueError("Set-cover training dataset contains no instances.")
    rng = np.random.default_rng(int(seed))
    size = min(int(batch_size or len(instances)), len(instances))
    while True:
        indices = rng.choice(len(instances), size=size, replace=False)
        yield [copy.deepcopy(instances[int(index)]) for index in indices]


def make_evaluation(instances, timeout_seconds=120, return_list=True):
    return SCPEvaluation(
        instances=normalize_scp_instances(instances),
        timeout_seconds=timeout_seconds,
        return_list=return_list,
    )


def build_task(task_cfg):
    instances = load_train_instances(
        task_cfg["train_datasets"],
        task_cfg.get("train_instances_per_dataset"),
    )
    return make_evaluation(
        instances,
        timeout_seconds=task_cfg.get("timeout_seconds", 120),
        return_list=task_cfg.get("return_list", True),
    )


def hidden_specs(hidden_test_cfg):
    specs = []
    for label in ("id", "ood"):
        for path in hidden_test_cfg.get(f"{label}_datasets", []):
            specs.append((str(resolve_repo_path(path)), f"{label}_{Path(path).stem}"))
    return specs


def hidden_eval_factory(hidden_test_cfg):
    limit = hidden_test_cfg.get("eval_instances")
    timeout = hidden_test_cfg.get("function_timeout_seconds") or 120

    def factory(instances, stem=None):
        feasible = normalize_scp_instances(instances)
        selected = feasible[: int(limit)] if limit is not None else feasible
        return make_evaluation(selected, timeout_seconds=timeout, return_list=True)

    return factory
