from __future__ import annotations

import pickle
from numbers import Integral
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
        if not isinstance(instance, (list, tuple)) or len(instance) != 2:
            raise ValueError(
                f"Instance {index} in {path} must be a 2-tuple "
                "(items, container-size)."
            )
        items, container = instance
        if not isinstance(items, (list, tuple)) or not items:
            raise ValueError(f"Instance {index} in {path} has invalid items.")
        if not isinstance(container, Integral) and not (
            isinstance(container, (list, tuple)) and len(container) == 2
        ):
            raise ValueError(f"Instance {index} in {path} has an invalid container size.")
    return instances


def load_dataset_file(
    task_dir,
    filename=None,
    split="train",
    size=None,
    **kwargs,
):
    """Load BP1D/BP2D datasets from the task's train, ID, or OOD directories."""
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
