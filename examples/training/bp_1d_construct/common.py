from __future__ import annotations

"""Task adapter for bp_1d_construct: instance loading, descriptors,
in-memory evaluation factories, and hidden ID/OOD specs."""

import copy
from pathlib import Path

import numpy as np

from llm4ad.task.optimization.bp_1d_construct import BP1DEvaluation
from post_eval_common import load_instances, resolve_repo_path


CAPACITY = 100
VALID_SIZES = (100, 500, 1000, 2000)


def normalize_bp1d_instance(instance):
    """Accept (items, capacity) or a plain item list and normalize."""
    values = list(instance)
    if values and all(isinstance(item, (int, float, np.integer)) for item in values):
        items, capacity = values, CAPACITY
    elif len(values) == 2 and isinstance(values[1], (int, float, np.integer)):
        items, capacity = values[0], int(values[1])
    else:
        raise ValueError(f"Invalid BP1D instance: {instance!r}")
    items = [int(item) for item in items]
    return items, int(capacity)


def is_valid_bp1d_instance(instance):
    try:
        items, capacity = normalize_bp1d_instance(instance)
    except Exception:
        return False
    return (
        capacity == CAPACITY
        and 0 < len(items)
        and all(isinstance(item, int) and 1 <= item <= capacity for item in items)
    )


def bp1d_descriptor(instance):
    items, capacity = normalize_bp1d_instance(instance)
    values = np.asarray(items, dtype=float)
    return np.array(
        [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
            float(np.percentile(values, 10)),
            float(np.percentile(values, 50)),
            float(np.percentile(values, 90)),
            float(np.sum(values) / capacity),
            float(len(values)),
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
        merged.extend(normalize_bp1d_instance(item) for item in instances)
    return merged


def build_wake_stream(datasets=None, dataset=None, seed=2026, batch_size=None, shuffle=False):
    paths = datasets if datasets is not None else [dataset]
    instances = load_train_instances(paths)
    if not instances:
        raise ValueError("BP1D training dataset contains no instances.")
    rng = np.random.default_rng(int(seed))
    round_size = min(int(batch_size or len(instances)), len(instances))
    while True:
        indices = rng.choice(len(instances), size=round_size, replace=False)
        yield [copy.deepcopy(instances[int(index)]) for index in indices]


def make_evaluation(instances, timeout_seconds=600, return_list=True):
    return BP1DEvaluation(
        instances=[normalize_bp1d_instance(item) for item in instances],
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
        timeout_seconds=task_cfg.get("timeout_seconds", 600),
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
    timeout_seconds = hidden_test_cfg.get("function_timeout_seconds") or 600

    def factory(instances, stem=None):
        if eval_instances is not None:
            instances = instances[: int(eval_instances)]
        return make_evaluation(instances, timeout_seconds=timeout_seconds)

    return factory
