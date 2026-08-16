from __future__ import annotations

"""Task adapter for bp_2d_construct: instance loading, descriptors,
in-memory evaluation factories, and hidden ID/OOD specs."""

import copy
from pathlib import Path

import numpy as np

from llm4ad.task.optimization.bp_2d_construct import BP2DEvaluation
from post_eval_common import load_instances, resolve_repo_path


BIN_DIMENSIONS = (100, 100)
VALID_SIZES = (50, 100, 200, 500)


def _is_pair(value):
    try:
        return len(value) == 2 and all(
            isinstance(item, (int, float, np.integer)) for item in value
        )
    except Exception:
        return False


def normalize_bp2d_instance(instance):
    """Accept (items, bin_dims) or a plain (w, h) item list and normalize."""
    values = list(instance)
    if values and all(_is_pair(item) for item in values):
        items, bin_dims = values, BIN_DIMENSIONS
    elif len(values) == 2 and _is_pair(values[1]):
        items, bin_dims = values[0], (int(values[1][0]), int(values[1][1]))
    else:
        raise ValueError(f"Invalid BP2D instance: {instance!r}")
    items = [(int(width), int(height)) for width, height in items]
    return items, tuple(bin_dims)


def is_valid_bp2d_instance(instance):
    try:
        items, bin_dims = normalize_bp2d_instance(instance)
    except Exception:
        return False
    if tuple(bin_dims) != BIN_DIMENSIONS or not items:
        return False
    return all(
        isinstance(width, int)
        and isinstance(height, int)
        and 1 <= width <= bin_dims[0]
        and 1 <= height <= bin_dims[1]
        for width, height in items
    )


def bp2d_descriptor(instance):
    items, bin_dims = normalize_bp2d_instance(instance)
    values = np.asarray(items, dtype=float)
    widths = values[:, 0]
    heights = values[:, 1]
    areas = widths * heights
    return np.array(
        [
            float(np.mean(widths)),
            float(np.std(widths)),
            float(np.mean(heights)),
            float(np.std(heights)),
            float(np.mean(areas)),
            float(np.std(areas)),
            float(np.percentile(areas, 10)),
            float(np.percentile(areas, 90)),
            float(np.sum(areas) / (bin_dims[0] * bin_dims[1])),
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
        merged.extend(normalize_bp2d_instance(item) for item in instances)
    return merged


def build_wake_stream(datasets=None, dataset=None, seed=2026, batch_size=None, shuffle=False):
    paths = datasets if datasets is not None else [dataset]
    instances = load_train_instances(paths)
    if not instances:
        raise ValueError("BP2D training dataset contains no instances.")
    rng = np.random.default_rng(int(seed))
    round_size = min(int(batch_size or len(instances)), len(instances))
    while True:
        indices = rng.choice(len(instances), size=round_size, replace=False)
        yield [copy.deepcopy(instances[int(index)]) for index in indices]


def make_evaluation(instances, timeout_seconds=600, return_list=True):
    return BP2DEvaluation(
        instances=[normalize_bp2d_instance(item) for item in instances],
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
