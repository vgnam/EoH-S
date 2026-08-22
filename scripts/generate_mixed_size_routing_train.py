from __future__ import annotations

"""Generate mixed-size training sets for TSP, CVRP, OVRP, and VRPTW.

The default protocol keeps the original training budget (four families times
32 instances) while replacing the fixed size of 30 with a near-balanced mix
of 20, 50, and 100 cities/customers.
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TSP_SCRIPT_DIR = REPO_ROOT / "examples" / "training" / "tsp_set"
CVRP_SCRIPT_DIR = REPO_ROOT / "examples" / "training" / "cvrp_set"
for module_dir in (TSP_SCRIPT_DIR, CVRP_SCRIPT_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from generate_hidden_tsp_dataset import (  # noqa: E402
    TRAIN_FAMILIES,
    make_tsp_instance,
    sample_hidden_regime,
)
from generate_hidden_cvrp_dataset import make_cvrp_instance  # noqa: E402


DEFAULT_TASKS = ("tsp", "cvrp", "ovrp", "vrptw")
DEFAULT_SIZES = (20, 50, 100)
DEFAULT_INSTANCES_PER_FAMILY = 32
TASK_SEEDS = {
    "tsp": 12026,
    "cvrp": 22026,
    "ovrp": 32026,
    "vrptw": 42026,
}


def _size_schedule(sizes, count, family_index, rng):
    """Return a shuffled, near-balanced schedule with rotated remainders."""
    schedule = [int(sizes[(index + family_index) % len(sizes)]) for index in range(count)]
    rng.shuffle(schedule)
    return schedule


def _output_path(task, family, instances_per_family):
    if task in ("tsp", "cvrp"):
        return (
            REPO_ROOT
            / "datasets"
            / task
            / f"dataset_{task}_train_mixed_sizes_{family}_{instances_per_family}.pkl"
        )
    return (
        REPO_ROOT
        / "code"
        / "llm4ad"
        / "task"
        / "optimization"
        / f"{task}_construct"
        / "train_datasets"
        / f"family_{family}_mixed_sizes.pkl"
    )


def _coordinates_for(task, instance):
    if task == "tsp":
        return np.asarray(instance[0], dtype=float)
    if task == "cvrp":
        return np.asarray(instance[0], dtype=float)[1:]
    return np.asarray(instance, dtype=float)[1:]


def _build_family_dataset(task, family, family_index, sizes, count, seed):
    rng = np.random.default_rng(seed)
    schedule = _size_schedule(sizes, count, family_index, rng)
    instances = []

    for size in schedule:
        if task == "tsp":
            coords = sample_hidden_regime(rng, size, family)
            instance = make_tsp_instance(coords)
        elif task == "cvrp":
            customer_coords = sample_hidden_regime(rng, size, family)
            instance = make_cvrp_instance(customer_coords, rng)
        else:
            # OVRP/VRPTW loaders derive all non-coordinate fields
            # deterministically. Row 0 is the depot.
            instance = sample_hidden_regime(rng, size + 1, family)
        instances.append(instance)

    size_counts = dict(sorted(Counter(schedule).items()))
    unit = "city" if task == "tsp" else "customer"
    return {
        "format": f"eohs-open-world-{task}-train-mixed-size-v1",
        "seed": int(seed),
        f"{unit}_sizes": [int(size) for size in sizes],
        "family": family,
        "families": [family],
        "instances_per_family": int(count),
        "size_counts": size_counts,
        "instances": instances,
        "instance_families": [family] * count,
        "instance_sizes": schedule,
    }


def _validate_dataset(task, dataset, expected_family, expected_sizes, expected_count):
    instances = dataset.get("instances", [])
    stored_sizes = [int(size) for size in dataset.get("instance_sizes", [])]
    actual_sizes = [len(_coordinates_for(task, instance)) for instance in instances]

    if len(instances) != expected_count:
        raise ValueError(f"{task}/{expected_family}: expected {expected_count} instances")
    if stored_sizes != actual_sizes:
        raise ValueError(f"{task}/{expected_family}: instance_sizes do not match arrays")
    if set(actual_sizes) != set(expected_sizes):
        raise ValueError(
            f"{task}/{expected_family}: expected sizes {expected_sizes}, got {sorted(set(actual_sizes))}"
        )
    if dataset.get("instance_families") != [expected_family] * expected_count:
        raise ValueError(f"{task}/{expected_family}: invalid family metadata")
    if max(Counter(actual_sizes).values()) - min(Counter(actual_sizes).values()) > 1:
        raise ValueError(f"{task}/{expected_family}: size distribution is not balanced")


def generate_task(task, sizes, instances_per_family):
    written = []
    for family_index, family in enumerate(TRAIN_FAMILIES):
        # Family-specific seeds make each file reproducible independently.
        seed = TASK_SEEDS[task] + family_index * 1009
        dataset = _build_family_dataset(
            task,
            family,
            family_index,
            sizes,
            instances_per_family,
            seed,
        )
        _validate_dataset(task, dataset, family, sizes, instances_per_family)
        path = _output_path(task, family, instances_per_family)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        written.append(path)
        counts = ", ".join(f"n={size}:{count}" for size, count in dataset["size_counts"].items())
        print(f"saved {path.relative_to(REPO_ROOT)} ({counts})")
    return written


def verify_file(path, task, family, sizes, instances_per_family):
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    _validate_dataset(task, dataset, family, sizes, instances_per_family)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--instances-per-family", type=int, default=DEFAULT_INSTANCES_PER_FAMILY)
    args = parser.parse_args()

    sizes = tuple(dict.fromkeys(int(size) for size in args.sizes))
    if not sizes or min(sizes) < 2:
        parser.error("--sizes must contain positive routing sizes >= 2")
    if args.instances_per_family < len(sizes):
        parser.error("--instances-per-family must be at least the number of sizes")

    all_paths = []
    for task in args.tasks:
        all_paths.extend(generate_task(task, sizes, args.instances_per_family))

    for path in all_paths:
        task = next(task for task in args.tasks if f"{task}_" in path.name or f"{task}_construct" in str(path))
        family = next(family for family in TRAIN_FAMILIES if family in path.stem)
        verify_file(path, task, family, sizes, args.instances_per_family)
    print(f"verified {len(all_paths)} files; tasks={args.tasks}; sizes={list(sizes)}")


if __name__ == "__main__":
    main()
