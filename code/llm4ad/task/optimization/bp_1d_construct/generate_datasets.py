from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


CAPACITY = 100
SIZES = (100, 500, 1000, 2000)
INSTANCES_PER_SIZE = 64
ID_FAMILY = "beta"
OOD_FAMILIES = ("uniform_small", "bimodal", "near_capacity", "lognormal")
SEED = 20260702


def _balanced_families(rng, families, count):
    selected = [families[index % len(families)] for index in range(count)]
    rng.shuffle(selected)
    return selected


def _sample_beta_items(rng, num_items, capacity=CAPACITY):
    weights = 50.0 - rng.beta(2.0, 5.0, num_items) * 40.0
    items = np.rint(np.clip(weights, 1, capacity)).astype(np.int64)
    return items.tolist(), {"alpha": 2.0, "beta": 5.0}


def _sample_uniform_small(rng, num_items, capacity):
    low = int(rng.integers(3, 8))
    high = int(rng.integers(18, 31))
    items = rng.integers(low, high + 1, size=num_items)
    return np.clip(items, 1, capacity).astype(np.int64).tolist(), {
        "low": low,
        "high": high,
    }


def _sample_bimodal(rng, num_items, capacity):
    small_mean = float(rng.uniform(8.0, 20.0))
    large_mean = float(rng.uniform(70.0, 90.0))
    std = float(rng.uniform(2.0, 6.0))
    large_weight = float(rng.uniform(0.35, 0.65))
    use_large = rng.random(num_items) < large_weight
    samples = rng.normal(small_mean, std, num_items)
    n_large = int(np.sum(use_large))
    if n_large:
        samples[use_large] = rng.normal(large_mean, std, n_large)
    items = np.rint(np.clip(samples, 1, capacity)).astype(np.int64)
    return items.tolist(), {
        "small_mean": small_mean,
        "large_mean": large_mean,
        "std": std,
        "large_weight": large_weight,
    }


def _sample_near_capacity(rng, num_items, capacity):
    center = float(rng.uniform(82.0, 95.0))
    std = float(rng.uniform(3.0, 8.0))
    items = np.rint(np.clip(rng.normal(center, std, num_items), 1, capacity)).astype(np.int64)
    return items.tolist(), {"center": center, "std": std}


def _sample_lognormal(rng, num_items, capacity):
    mean = float(rng.uniform(3.0, 4.2))
    sigma = float(rng.uniform(0.3, 0.6))
    items = np.rint(
        np.clip(np.exp(rng.normal(mean, sigma, num_items)), 1, capacity)
    ).astype(np.int64)
    return items.tolist(), {"mean": mean, "sigma": sigma}


def _sample_ood_family(rng, num_items, capacity, family):
    if family == "uniform_small":
        return _sample_uniform_small(rng, num_items, capacity)
    if family == "bimodal":
        return _sample_bimodal(rng, num_items, capacity)
    if family == "near_capacity":
        return _sample_near_capacity(rng, num_items, capacity)
    if family == "lognormal":
        return _sample_lognormal(rng, num_items, capacity)
    raise ValueError(f"Unknown OOD family: {family}")


def generate_split(task_dir, split, size, rng):
    if split not in {"train", "id", "ood"}:
        raise ValueError("split must be one of: train, id, ood")
    if split == "ood":
        instance_families = _balanced_families(rng, OOD_FAMILIES, INSTANCES_PER_SIZE)
    else:
        instance_families = [ID_FAMILY] * INSTANCES_PER_SIZE

    instances = []
    records = []
    for index, family in enumerate(instance_families):
        instance_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        instance_rng = np.random.default_rng(instance_seed)
        if split == "ood":
            items, parameters = _sample_ood_family(
                instance_rng, size, CAPACITY, family
            )
        else:
            items, parameters = _sample_beta_items(instance_rng, size, CAPACITY)
        instances.append((items, CAPACITY))
        records.append(
            {
                "index": index,
                "family": family,
                "seed": instance_seed,
                "parameters": parameters,
                "num_items": int(size),
            }
        )
    return instances, records


def _validate_split(instances, records, split, size):
    if len(instances) != INSTANCES_PER_SIZE or len(records) != INSTANCES_PER_SIZE:
        raise ValueError(f"Unexpected instance count in split {split} size {size}.")
    for instance, record in zip(instances, records):
        items, capacity = instance
        if len(items) != size or capacity != CAPACITY:
            raise ValueError(f"Invalid shape in split {split} size {size}.")
        if any(not isinstance(item, (int, np.integer)) or item < 1 or item > capacity for item in items):
            raise ValueError(f"Invalid item weight in split {split} size {size}.")
    counts = Counter(record["family"] for record in records)
    if split == "ood":
        expected = INSTANCES_PER_SIZE // len(OOD_FAMILIES)
        if any(count != expected for count in counts.values()) or len(counts) != len(OOD_FAMILIES):
            raise ValueError(f"Unbalanced OOD families for size {size}: {counts}")
    return dict(sorted(counts.items()))


def _write_pickle(path, instances):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(instances, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _file_entry(task_dir, path, split, size, counts):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "split": split,
        "size": int(size),
        "path": path.relative_to(task_dir).as_posix(),
        "n_instances": int(INSTANCES_PER_SIZE),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "distribution_counts": counts,
    }


def generate_default_splits(
    output_dir,
    *,
    seed=SEED,
    sizes=SIZES,
    instances=INSTANCES_PER_SIZE,
    capacity=CAPACITY,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    all_entries = []
    split_offsets = {"train": 0, "id": 10_000, "ood": 20_000}
    for split, directory in (("train", "train_datasets"), ("id", "test_datasets"), ("ood", "ood_test_datasets/mixture")):
        split_entries = []
        for size_index, size in enumerate(sizes):
            split_rng = np.random.default_rng(int(seed) + split_offsets[split] + size_index)
            instances_data, records = generate_split(output_dir, split, int(size), split_rng)
            counts = _validate_split(instances_data, records, split, int(size))
            path = output_dir / directory / f"size_{size}.pkl"
            _write_pickle(path, instances_data)
            entry = _file_entry(output_dir, path, split, int(size), counts)
            split_entries.append(entry)
            all_entries.append(entry)
            print(f"saved {path} instances={len(instances_data)} families={counts}")
        if split == "ood":
            metadata = {
                "task": "bp_1d_construct",
                "suite": "mixture",
                "split": "ood_test",
                "reference_split": "train (ID)",
                "seed": int(seed),
                "families": list(OOD_FAMILIES),
                "datasets": split_entries,
            }
            metadata_path = output_dir / directory / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    root_metadata = {
        "task": "bp_1d_construct",
        "seed": int(seed),
        "instances_per_size": int(instances),
        "sizes": [int(size) for size in sizes],
        "capacity": int(capacity),
        "id_families": [ID_FAMILY],
        "ood_families": list(OOD_FAMILIES),
        "files": all_entries,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(root_metadata, indent=2), encoding="utf-8"
    )
    return all_entries


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic BP1D train/ID/OOD splits.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--instances", type=int, default=INSTANCES_PER_SIZE)
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    args = parser.parse_args()
    generate_default_splits(
        args.output_dir,
        seed=args.seed,
        sizes=args.sizes,
        instances=args.instances,
        capacity=args.capacity,
    )


if __name__ == "__main__":
    main()
