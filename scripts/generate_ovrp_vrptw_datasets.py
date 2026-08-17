from __future__ import annotations

"""Generate OVRP/VRPTW construct datasets mirroring the TSP/CVRP layout.

Sizes / counts match the TSP/CVRP construct protocol:
  * train: 4 families x 32 = 128 instances, fixed 30 customers
    (one pkl per family: family_<name>.pkl),
  * hidden ID / OOD: 3 city sizes (20/50/100) x 128 instances each
    (one pkl per size per split).
  * ID instances are balanced across the 4 train families (32/family/size);
  * OOD instances are gaussian mixtures with random Dirichlet weights
    (TSP mixed_structures sampler).

Each stored instance is only the node coordinates (n_nodes, 2) with the
depot at row 0; demands / capacity / (service times / time windows) are
derived deterministically at load time by the task's common.py
(_instance_from_coordinates), exactly like LLM-synthesized OW-CAHD regimes.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "examples" / "training"))
sys.path.insert(0, str(REPO_ROOT / "examples" / "training" / "tsp_set"))

from generate_hidden_tsp_dataset import (  # noqa: E402
    TRAIN_FAMILIES,
    sample_bezier,
    sample_cluster,
    sample_grid_holes,
    sample_mixed_structures,
    sample_uniform,
    _clip,
)

TRAIN_N_CUSTOMERS = 30
HIDDEN_SIZES = (20, 50, 100)
INSTANCES_PER_FAMILY = 32
INSTANCES_PER_SIZE = 128
SEED = {"ovrp": 32026, "vrptw": 42026}


def _sample_family(rng, family, n_nodes):
    if family == "uniform":
        return sample_uniform(rng, n_nodes)
    if family == "cluster":
        return sample_cluster(rng, n_nodes)
    if family == "bezier":
        return sample_bezier(rng, n_nodes)
    if family == "grid_holes":
        return sample_grid_holes(rng, n_nodes)
    raise ValueError(f"Unknown family {family}")


def _directory(task, split):
    task_dir = REPO_ROOT / "code" / "llm4ad" / "task" / "optimization" / f"{task}_construct"
    if split == "train":
        return task_dir / "train_datasets"
    if split == "id":
        return task_dir / "test_datasets"
    return task_dir / "ood_test_datasets" / "mixture"


def generate_train(task):
    """One pkl per family; every instance = coordinates (31, 2), depot at row 0."""
    rng = np.random.default_rng(SEED[task])
    n_nodes = TRAIN_N_CUSTOMERS + 1
    out_dir = _directory(task, "train")
    out_dir.mkdir(parents=True, exist_ok=True)
    for family in TRAIN_FAMILIES:
        coords = []
        for _ in range(INSTANCES_PER_FAMILY):
            coords.append(_sample_family(rng, family, n_nodes))
        dataset = {
            "format": f"eohs-open-world-{task}-train-v2",
            "seed": int(SEED[task]),
            "n_customers": TRAIN_N_CUSTOMERS,
            "family": family,
            "instances_per_family": INSTANCES_PER_FAMILY,
            "instances": coords,
            "instance_families": [family] * INSTANCES_PER_FAMILY,
        }
        path = out_dir / f"family_{family}.pkl"
        with path.open("wb") as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved {path} ({len(coords)} instances, family={family})")


def generate_hidden(task, split):
    """One pkl per size; ID balanced 4 families, OOD gaussian Dirichlet mixture."""
    rng = np.random.default_rng(SEED[task] + (11 if split == "id" else 22))
    out_dir = _directory(task, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    for size in HIDDEN_SIZES:
        n_nodes = size + 1
        coords = []
        instance_families = []
        if split == "id":
            # balanced assignment: 32 instances per family per size, shuffled
            families = list(TRAIN_FAMILIES) * (INSTANCES_PER_SIZE // len(TRAIN_FAMILIES))
            rng.shuffle(families)
            for family in families:
                coords.append(_sample_family(rng, family, n_nodes))
                instance_families.append(family)
        else:
            for _ in range(INSTANCES_PER_SIZE):
                coords.append(sample_mixed_structures(rng, n_nodes))
                instance_families.append("mixed_structures")
        dataset = {
            "format": f"eohs-open-world-{task}-hidden-v2",
            "seed": int(SEED[task] + (11 if split == "id" else 22)),
            "split": split,
            "customer_sizes": [int(size)],
            "instances_per_size": INSTANCES_PER_SIZE,
            "instances": coords,
            "instance_families": instance_families,
        }
        path = out_dir / f"size_{size}.pkl"
        with path.open("wb") as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved {path} ({len(coords)} instances, split={split}, size={size})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["ovrp", "vrptw"])
    args = parser.parse_args()
    generate_train(args.task)
    generate_hidden(args.task, "id")
    generate_hidden(args.task, "ood")


if __name__ == "__main__":
    main()