from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from llm4ad.task.optimization.fssp_construct import FSSPEvaluation
from post_eval_common import load_instances, resolve_repo_path


def normalize_fssp_instance(instance):
    if not isinstance(instance, (list, tuple)) or len(instance) != 3:
        raise ValueError("FSSP instance must be (processing_times, n_jobs, n_machines).")
    processing_times, n_jobs, n_machines = instance
    matrix = np.asarray(processing_times, dtype=int)
    n_jobs, n_machines = int(n_jobs), int(n_machines)
    if matrix.shape != (n_jobs, n_machines) or np.any(matrix <= 0):
        raise ValueError("Invalid FSSP processing-time matrix.")
    return matrix.tolist(), n_jobs, n_machines


def is_valid_fssp_instance(instance):
    try:
        normalize_fssp_instance(instance)
        return True
    except Exception:
        return False


def fssp_descriptor(instance):
    processing_times, n_jobs, n_machines = normalize_fssp_instance(instance)
    values = np.asarray(processing_times, dtype=float)
    return np.asarray([n_jobs, n_machines, np.mean(values), np.std(values),
        np.percentile(values, 10), np.percentile(values, 90),
        np.std(np.sum(values, axis=0)) / max(np.mean(np.sum(values, axis=0)), 1.0),
        np.std(np.sum(values, axis=1)) / max(np.mean(np.sum(values, axis=1)), 1.0)], dtype=float)


def load_train_instances(paths, instances_per_dataset=None):
    merged=[]
    for path in paths:
        values=load_instances(resolve_repo_path(path))
        if instances_per_dataset is not None: values=values[:int(instances_per_dataset)]
        merged.extend(normalize_fssp_instance(value) for value in values)
    return merged


def build_wake_stream(datasets=None,dataset=None,seed=2026,batch_size=None,shuffle=True):
    values=load_train_instances(datasets if datasets is not None else [dataset])
    rng=np.random.default_rng(int(seed)); size=min(int(batch_size or len(values)),len(values))
    while True:
        yield [copy.deepcopy(values[int(i)]) for i in rng.choice(len(values),size=size,replace=False)]


def make_evaluation(instances,timeout_seconds=120,return_list=True):
    return FSSPEvaluation(instances=[normalize_fssp_instance(x) for x in instances],timeout_seconds=timeout_seconds,return_list=return_list)


def build_task(cfg):
    return make_evaluation(load_train_instances(cfg["train_datasets"],cfg.get("train_instances_per_dataset")),cfg.get("timeout_seconds",120),cfg.get("return_list",True))


def hidden_specs(cfg):
    return [(str(resolve_repo_path(path)),f"{label}_{Path(path).stem}") for label in ("id","ood") for path in cfg.get(f"{label}_datasets",[])]


def hidden_eval_factory(cfg):
    limit=cfg.get("eval_instances"); timeout=cfg.get("function_timeout_seconds") or 120
    def factory(instances,stem=None):
        return make_evaluation(instances[:int(limit)] if limit is not None else instances,timeout,True)
    return factory
