from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from llm4ad.task.optimization.jssp_construct import JSSPEvaluation
from post_eval_common import load_instances, resolve_repo_path


def normalize_jssp_instance(instance):
    if not isinstance(instance, (list, tuple)) or len(instance) != 3:
        raise ValueError("JSSP instance must be (processing_times, n_jobs, n_machines).")
    processing_times, n_jobs, n_machines = instance
    matrix = np.asarray(processing_times, dtype=int)
    n_jobs, n_machines = int(n_jobs), int(n_machines)
    if matrix.shape != (n_jobs, n_machines) or np.any(matrix <= 0):
        raise ValueError("Invalid JSSP processing-time matrix.")
    return matrix.tolist(), n_jobs, n_machines


def is_valid_jssp_instance(instance):
    try:
        normalize_jssp_instance(instance)
        return True
    except Exception:
        return False


def jssp_descriptor(instance):
    processing_times, n_jobs, n_machines = normalize_jssp_instance(instance)
    values = np.asarray(processing_times, dtype=float)
    machine_loads = np.sum(values, axis=0)
    job_loads = np.sum(values, axis=1)
    return np.asarray([
        n_jobs,
        n_machines,
        np.mean(values),
        np.std(values),
        np.percentile(values, 10),
        np.percentile(values, 90),
        np.std(machine_loads) / max(np.mean(machine_loads), 1.0),
        np.std(job_loads) / max(np.mean(job_loads), 1.0),
    ], dtype=float)


def load_train_instances(paths, instances_per_dataset=None):
    merged = []
    for path in paths:
        instances = load_instances(resolve_repo_path(path))
        if instances_per_dataset is not None:
            instances = instances[: int(instances_per_dataset)]
        merged.extend(normalize_jssp_instance(item) for item in instances)
    return merged


def build_wake_stream(datasets=None, dataset=None, seed=2026, batch_size=None, shuffle=True):
    paths = datasets if datasets is not None else [dataset]
    instances = load_train_instances(paths)
    if not instances:
        raise ValueError("JSSP training dataset contains no instances.")
    rng = np.random.default_rng(int(seed))
    size = min(int(batch_size or len(instances)), len(instances))
    while True:
        indices = rng.choice(len(instances), size=size, replace=False)
        yield [copy.deepcopy(instances[int(index)]) for index in indices]


def make_evaluation(instances, timeout_seconds=600, return_list=True):
    return JSSPEvaluation(
        instances=[normalize_jssp_instance(item) for item in instances],
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
        timeout_seconds=task_cfg.get("timeout_seconds", 600),
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
    timeout = hidden_test_cfg.get("function_timeout_seconds") or 600

    def factory(instances, stem=None):
        selected = instances[: int(limit)] if limit is not None else instances
        return make_evaluation(selected, timeout_seconds=timeout, return_list=True)

    return factory
