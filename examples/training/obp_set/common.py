from __future__ import annotations

"""Task adapter for online_bin_packing_set: instance loading, descriptors,
in-memory evaluation factories, and hidden ID/OOD specs."""

import copy
from pathlib import Path

import numpy as np

from llm4ad.task.optimization.online_bin_packing_set import OBPSEvaluation
from post_eval_common import load_instances, resolve_repo_path


CAPACITY = 100
VALID_SIZES = (200, 500, 1000)


def normalize_obp_instance(instance):
    if not isinstance(instance, dict):
        raise ValueError(f"OBP instances must be dicts, got {type(instance).__name__}.")
    capacity = int(instance.get("capacity", CAPACITY))
    if capacity != CAPACITY:
        raise ValueError(f"OBP capacity must be {CAPACITY}, got {capacity}.")
    items = [float(item) for item in instance["items"]]
    num_items = int(instance.get("num_items", len(items)))
    if num_items != len(items):
        raise ValueError(
            f"OBP num_items {num_items} does not match len(items) {len(items)}."
        )
    return {"capacity": capacity, "num_items": num_items, "items": items}


def is_valid_obp_instance(instance):
    try:
        normalized = normalize_obp_instance(instance)
    except Exception:
        return False
    return (
        0 < normalized["num_items"]
        and all(0 < item <= CAPACITY for item in normalized["items"])
    )


def obp_descriptor(instance):
    normalized = normalize_obp_instance(instance)
    values = np.asarray(normalized["items"], dtype=float)
    return np.array(
        [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
            float(np.percentile(values, 10)),
            float(np.percentile(values, 50)),
            float(np.percentile(values, 90)),
            float(np.sum(values) / CAPACITY),
            float(len(values)),
        ],
        dtype=float,
    )


def load_train_instances(paths, n_instances=None):
    merged = []
    for path in paths:
        path = resolve_repo_path(path)
        merged.extend(
            normalize_obp_instance(item) for item in load_instances(path)
        )
    if n_instances is not None:
        merged = merged[: int(n_instances)]
    return merged


def build_wake_stream(datasets=None, dataset=None, seed=2026, batch_size=None, shuffle=False):
    paths = datasets if datasets is not None else [dataset]
    instances = load_train_instances(paths)
    if not instances:
        raise ValueError("OBP training dataset contains no instances.")
    rng = np.random.default_rng(int(seed))
    round_size = min(int(batch_size or len(instances)), len(instances))
    while True:
        indices = rng.choice(len(instances), size=round_size, replace=False)
        yield [copy.deepcopy(instances[int(index)]) for index in indices]


def make_evaluation(instances, timeout_seconds=120, return_list=True):
    return OBPSEvaluation(
        instances=[normalize_obp_instance(item) for item in instances],
        timeout_seconds=timeout_seconds,
        return_list=return_list,
    )


def build_task(task_cfg):
    instances = load_train_instances(
        task_cfg["train_datasets"],
        n_instances=task_cfg.get("train_n_instances"),
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
        if stem.startswith("dataset_obp_hidden_"):
            stem = stem[len("dataset_obp_hidden_"):]
        specs.append((str(resolve_repo_path(path)), stem))
    for path in hidden_test_cfg.get("ood_datasets", []):
        stem = Path(path).stem
        if stem.startswith("dataset_obp_hidden_"):
            stem = stem[len("dataset_obp_hidden_"):]
        specs.append((str(resolve_repo_path(path)), stem))
    return specs


def hidden_eval_factory(hidden_test_cfg):
    eval_instances = hidden_test_cfg.get("eval_instances")
    timeout_seconds = hidden_test_cfg.get("function_timeout_seconds") or 120

    def factory(instances, stem=None):
        if eval_instances is not None:
            instances = instances[: int(eval_instances)]
        return make_evaluation(instances, timeout_seconds=timeout_seconds)

    return factory
