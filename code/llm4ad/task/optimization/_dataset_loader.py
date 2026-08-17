from __future__ import annotations

import pickle
from pathlib import Path


_SPLIT_DIRS = {
    "train": "train_datasets",
    "test": "test_datasets",
    "id": "test_datasets",
    "ood": "ood_test_datasets/mixture",
}


def _load_instances(path: Path):
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        instances = data.get("instances")
        if instances is None:
            raise ValueError(f"Dataset {path} is a dict without an 'instances' field.")
    else:
        instances = data
    if not isinstance(instances, (list, tuple)) or not instances:
        raise ValueError(f"Dataset {path} must contain a non-empty list of instances.")
    return list(instances)


def _validate_instances(instances, path):
    for index, instance in enumerate(instances):
        if not isinstance(instance, (list, tuple)) or len(instance) < 2:
            raise ValueError(
                f"Instance {index} in {path} must be a tuple/list with at least "
                "two fields (task-specific layout)."
            )
    return instances


def load_dataset_file(
    task_dir,
    filename=None,
    split="train",
    size=None,
    **kwargs,
):
    """Load task datasets from the task's train, ID, or OOD directories.

    Datasets are plain pickled lists of instances. Instance layouts are
    task-specific — BP1D/BP2D use (items, container) tuples while
    OVRP/VRPTW use (coordinates, distance_matrix, demands, ...) tuples —
    so only coarse structure (a non-empty list of tuple/list instances)
    is validated here; tasks interpret their own layouts.
    """
    task_dir = Path(task_dir)
    if filename is None:
        if split not in _SPLIT_DIRS:
            raise ValueError(f"Unsupported dataset split: {split!r}")
        if size is None:
            raise ValueError("size must be provided when filename is omitted.")
        path = task_dir / _SPLIT_DIRS[split] / f"size_{size}.pkl"
    else:
        path = Path(filename)
        if not path.is_absolute():
            path = task_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    instances = _load_instances(path)
    return _validate_instances(instances, path)
