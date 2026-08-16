from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


BIN_DIMENSIONS = (100, 100)
SIZES = (50, 100, 200, 500)
INSTANCES_PER_SIZE = 64
ID_FAMILY = "uniform"
OOD_FAMILIES = ("slender", "large_squares", "bimodal", "small_dense")
SEED = 20260702


def _balanced_families(rng, families, count):
    selected = [families[index % len(families)] for index in range(count)]
    rng.shuffle(selected)
    return selected


def _sample_uniform_dimensions(rng, num_items, bin_dimensions):
    widths = rng.integers(10, bin_dimensions[0] - 9, size=num_items)
    heights = rng.integers(10, bin_dimensions[1] - 9, size=num_items)
    return list(zip(widths.tolist(), heights.tolist())), {}


def _sample_slender(rng, num_items, bin_dimensions):
    widths = rng.integers(5, 26, size=num_items)
    heights = rng.integers(45, 96, size=num_items)
    return list(zip(widths.tolist(), heights.tolist())), {}


def _sample_large_squares(rng, num_items, bin_dimensions):
    sides = rng.integers(45, 91, size=num_items)
    return [(int(side), int(side)) for side in sides], {}


def _sample_bimodal(rng, num_items, bin_dimensions):
    large_weight = float(rng.uniform(0.35, 0.65))
    use_large = rng.random(num_items) < large_weight
    sides = np.zeros(num_items, dtype=np.int64)
    n_large = int(np.sum(use_large))
    sides[~use_large] = rng.integers(10, 31, size=num_items - n_large)
    if n_large:
        sides[use_large] = rng.integers(60, 91, size=n_large)
    return [(int(side), int(side)) for side in sides], {
        "large_weight": large_weight
    }


def _sample_small_dense(rng, num_items, bin_dimensions):
    widths = rng.integers(5, 26, size=num_items)
    heights = rng.integers(5, 26, size=num_items)
    return list(zip(widths.tolist(), heights.tolist())), {}


def _sample_ood_family(rng, num_items, bin_dimensions, family):
    if family == "slender":
        return _sample_slender(rng, num_items, bin_dimensions)
    if family == "large_squares":
        return _sample_large_squares(rng, num_items, bin_dimensions)
    if family == "bimodal":
        return _sample_bimodal(rng, num_items, bin_dimensions)
    if family == "small_dense":
        return _sample_small_dense(rng, num_items, bin_dimensions)
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
            item_dimensions, parameters = _sample_ood_family(
                instance_rng, size, BIN_DIMENSIONS, family
            )
        else:
            item_dimensions, parameters = _sample_uniform_dimensions(
                instance_rng, size, BIN_DIMENSIONS
            )
        instances.append((item_dimensions, BIN_DIMENSIONS))
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
        item_dimensions, bin_dimensions = instance
        if len(item_dimensions) != size or tuple(bin_dimensions) != BIN_DIMENSIONS:
            raise ValueError(f"Invalid shape in split {split} size {size}.")
        for width, height in item_dimensions:
            if (
                not isinstance(width, (int, np.integer))
                or not isinstance(height, (int, np.integer))
                or width < 1
                or height < 1
                or width > BIN_DIMENSIONS[0]
                or height > BIN_DIMENSIONS[1]
            ):
                raise ValueError(f"Invalid item dimensions in split {split} size {size}.")
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
    bin_dimensions=BIN_DIMENSIONS,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
                "task": "bp_2d_construct",
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
        "task": "bp_2d_construct",
        "seed": int(seed),
        "instances_per_size": int(instances),
        "sizes": [int(size) for size in sizes],
        "bin_dimensions": [int(value) for value in bin_dimensions],
        "id_families": [ID_FAMILY],
        "ood_families": list(OOD_FAMILIES),
        "files": all_entries,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(root_metadata, indent=2), encoding="utf-8"
    )
    return all_entries


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic BP2D train/ID/OOD splits.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--instances", type=int, default=INSTANCES_PER_SIZE)
    args = parser.parse_args()
    generate_default_splits(
        args.output_dir,
        seed=args.seed,
        sizes=args.sizes,
        instances=args.instances,
        bin_dimensions=BIN_DIMENSIONS,
    )


if __name__ == "__main__":
    main()
