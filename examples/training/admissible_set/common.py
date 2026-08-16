from __future__ import annotations

"""Task adapter for admissible_set: instance loading, descriptors,
in-memory evaluation factories, and hidden ID/OOD specs."""

import copy
from pathlib import Path

import numpy as np

from llm4ad.task.optimization.admissible_set import ASPEvaluation
from post_eval_common import load_instances, resolve_repo_path


OPTIMAL_SET_LENGTH = {
    "n12w7": 792,
    "n15w10": 3003,
    "n21w15": 43596,
    "n24w17": 237984,
}


def normalize_asp_instance(instance):
    if not isinstance(instance, dict):
        raise ValueError(
            f"Admissible instances must be dicts, got {type(instance).__name__}."
        )
    dimension = int(instance["dimension"])
    weight = int(instance["weight"])
    seed = instance.get("seed")
    if f"n{dimension}w{weight}" not in OPTIMAL_SET_LENGTH:
        raise ValueError(
            f"Unknown admissible (dimension, weight) pair: ({dimension}, {weight})."
        )
    if dimension % 3 != 0:
        raise ValueError(f"Admissible dimension must be divisible by 3: {dimension}.")
    if seed is not None:
        seed = int(seed) % 4294967295
    return {"dimension": dimension, "weight": weight, "seed": seed}


def is_valid_asp_instance(instance):
    try:
        normalize_asp_instance(instance)
        return True
    except Exception:
        return False


def asp_descriptor(instance):
    normalized = normalize_asp_instance(instance)
    return np.array(
        [
            float(normalized["dimension"]),
            float(normalized["weight"]),
            float(normalized["seed"] or 0),
        ],
        dtype=float,
    )


def load_train_instances(paths, n_instances=None):
    merged = []
    for path in paths:
        path = resolve_repo_path(path)
        merged.extend(
            normalize_asp_instance(item) for item in load_instances(path)
        )
    if n_instances is not None:
        merged = merged[: int(n_instances)]
    return merged


def build_wake_stream(datasets=None, dataset=None, seed=2026, batch_size=None, shuffle=False):
    paths = datasets if datasets is not None else [dataset]
    instances = load_train_instances(paths)
    if not instances:
        raise ValueError("Admissible training dataset contains no instances.")
    rng = np.random.default_rng(int(seed))
    round_size = min(int(batch_size or len(instances)), len(instances))
    while True:
        indices = rng.choice(len(instances), size=round_size, replace=False)
        yield [copy.deepcopy(instances[int(index)]) for index in indices]


def make_evaluation(instances, timeout_seconds=600, return_list=True):
    return ASPEvaluation(
        datasets=[normalize_asp_instance(item) for item in instances],
        timeout_seconds=timeout_seconds,
        return_list=return_list,
    )


def build_task(task_cfg):
    instances = load_train_instances(
        task_cfg["train_datasets"],
        n_instances=task_cfg.get("train_instances"),
    )
    return make_evaluation(
        instances,
        timeout_seconds=task_cfg.get("timeout_seconds", 600),
        return_list=task_cfg.get("return_list", True),
    )


def hidden_specs(hidden_test_cfg):
    specs = []
    for path in hidden_test_cfg.get("id_datasets", []):
        specs.append((str(resolve_repo_path(path)), "id"))
    for path in hidden_test_cfg.get("ood_datasets", []):
        specs.append((str(resolve_repo_path(path)), "ood"))
    return specs


def _balanced_slice(instances, cap):
    """Take up to cap instances, balanced across (dimension, weight) families."""
    groups = {}
    order = []
    for instance in instances:
        key = (int(instance["dimension"]), int(instance["weight"]))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(instance)
    if not order:
        return []
    per_family = max(1, int(cap) // len(order))
    selected = []
    for key in order:
        selected.extend(groups[key][:per_family])
    family_index = 0
    while len(selected) < int(cap):
        key = order[family_index % len(order)]
        extra = groups[key][per_family:]
        if not extra:
            family_index += 1
            if family_index > len(order) * 2:
                break
            continue
        selected.append(extra[0])
        groups[key] = groups[key][:per_family] + extra[1:]
        family_index += 1
    return selected[: int(cap)]


def hidden_eval_factory(hidden_test_cfg):
    id_eval_instances = hidden_test_cfg.get("id_eval_instances")
    ood_eval_instances = hidden_test_cfg.get("ood_eval_instances")
    timeout_seconds = hidden_test_cfg.get("function_timeout_seconds") or 600

    def factory(instances, stem=None):
        cap = ood_eval_instances if stem == "ood" else id_eval_instances
        if cap is not None:
            instances = _balanced_slice(instances, cap)
        return make_evaluation(instances, timeout_seconds=timeout_seconds)

    return factory
