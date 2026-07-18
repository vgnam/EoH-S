from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


CAPACITY = 100
TRAIN_FAMILIES = ("uniform", "normal", "weibull", "exponential")
OOD_FAMILIES = ("bimodal", "u_shaped", "complementary_pairs", "discrete_spikes")
DEFAULT_SIZES = (200, 500, 1000)


def _integer_items(samples, capacity):
    return np.rint(np.clip(samples, 1, capacity)).astype(np.int64)


def _sample_id_family(rng, num_items, capacity, family):
    """Sample an ID item stream, with parameters varying between instances."""
    if family == "uniform":
        low = int(rng.integers(5, 21))
        high = int(rng.integers(70, 96))
        return rng.integers(low, high + 1, size=num_items), {"low": low, "high": high}
    if family == "normal":
        mean = float(rng.uniform(35.0, 65.0))
        std = float(rng.uniform(8.0, 18.0))
        return _integer_items(rng.normal(mean, std, num_items), capacity), {
            "mean": mean,
            "std": std,
        }
    if family == "weibull":
        shape = float(rng.uniform(1.5, 3.5))
        scale = float(rng.uniform(25.0, 55.0))
        return _integer_items(rng.weibull(shape, num_items) * scale, capacity), {
            "shape": shape,
            "scale": scale,
        }
    if family == "exponential":
        scale = float(rng.uniform(20.0, 45.0))
        return _integer_items(rng.exponential(scale, num_items), capacity), {"scale": scale}
    raise ValueError(f"Unknown ID family: {family}")


def _sample_ood_family(rng, num_items, capacity, family):
    """Sample held-out stream structures that are absent from the ID families."""
    if family == "bimodal":
        small_mean = float(rng.uniform(12.0, 25.0))
        large_mean = float(rng.uniform(70.0, 88.0))
        std = float(rng.uniform(2.0, 6.0))
        large_weight = float(rng.uniform(0.35, 0.65))
        use_large = rng.random(num_items) < large_weight
        samples = rng.normal(small_mean, std, num_items)
        samples[use_large] = rng.normal(large_mean, std, int(np.sum(use_large)))
        return _integer_items(samples, capacity), {
            "small_mean": small_mean,
            "large_mean": large_mean,
            "std": std,
            "large_weight": large_weight,
        }
    if family == "u_shaped":
        alpha = float(rng.uniform(0.30, 0.65))
        beta = float(rng.uniform(0.30, 0.65))
        samples = 1.0 + (capacity - 1.0) * rng.beta(alpha, beta, num_items)
        return _integer_items(samples, capacity), {"alpha": alpha, "beta": beta}
    if family == "complementary_pairs":
        first = rng.integers(8, capacity - 7, size=(num_items + 1) // 2)
        noise = rng.integers(-2, 3, size=len(first))
        second = capacity - first + noise
        items = np.column_stack([first, second]).reshape(-1)[:num_items]
        # Preserve pair locality while changing the online arrival pattern per instance.
        blocks = items.reshape(-1, 2) if num_items % 2 == 0 else None
        if blocks is not None:
            rng.shuffle(blocks)
            items = blocks.reshape(-1)
        return np.clip(items, 1, capacity).astype(np.int64), {"pair_noise": 2}
    if family == "discrete_spikes":
        offset = int(rng.integers(-3, 4))
        spikes = np.clip(np.array([12, 27, 43, 58, 73, 88]) + offset, 1, capacity)
        probabilities = rng.dirichlet(np.full(len(spikes), 1.5))
        items = rng.choice(spikes, size=num_items, p=probabilities)
        return items.astype(np.int64), {
            "spikes": spikes.tolist(),
            "probabilities": probabilities.tolist(),
        }
    raise ValueError(f"Unknown OOD family: {family}")


def _balanced_families(rng, families, count):
    selected = [families[index % len(families)] for index in range(count)]
    rng.shuffle(selected)
    return selected


def generate_dataset(*, seed, num_items, instances, split, capacity=CAPACITY):
    if split not in {"train", "id", "ood"}:
        raise ValueError("split must be one of: train, id, ood")
    if num_items < 2 or instances < 1 or capacity < 2:
        raise ValueError("num_items, instances, and capacity must be positive")

    rng = np.random.default_rng(seed)
    families = OOD_FAMILIES if split == "ood" else TRAIN_FAMILIES
    instance_families = _balanced_families(rng, families, instances)
    dataset = {}
    for index, family in enumerate(instance_families):
        instance_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        instance_rng = np.random.default_rng(instance_seed)
        if split == "ood":
            items, parameters = _sample_ood_family(instance_rng, num_items, capacity, family)
        else:
            items, parameters = _sample_id_family(instance_rng, num_items, capacity, family)
        dataset[f"instance_{split}_{num_items}_{index:03d}"] = {
            "capacity": int(capacity),
            "num_items": int(num_items),
            "items": items,
            "split": split,
            "family": family,
            "parameters": parameters,
            "seed": instance_seed,
        }
    return dataset


def validate_dataset(dataset, *, split, num_items, instances, capacity=CAPACITY):
    if len(dataset) != instances:
        raise ValueError(f"Expected {instances} instances, got {len(dataset)}")
    allowed_families = set(OOD_FAMILIES if split == "ood" else TRAIN_FAMILIES)
    counts = {family: 0 for family in allowed_families}
    for name, instance in dataset.items():
        items = np.asarray(instance["items"])
        if instance["split"] != split or instance["num_items"] != num_items:
            raise ValueError(f"Invalid split/size metadata in {name}")
        if instance["capacity"] != capacity or items.shape != (num_items,):
            raise ValueError(f"Invalid capacity/item shape in {name}")
        if not np.issubdtype(items.dtype, np.integer):
            raise ValueError(f"Items must be integers in {name}")
        if np.any(items < 1) or np.any(items > capacity):
            raise ValueError(f"Items outside [1, capacity] in {name}")
        family = instance["family"]
        if family not in allowed_families:
            raise ValueError(f"Unexpected family {family!r} in {name}")
        counts[family] += 1
    if max(counts.values()) - min(counts.values()) > 1:
        raise ValueError(f"Families are not balanced: {counts}")
    return counts


def _write_dataset(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_default_splits(
    output_dir,
    *,
    seed=32026,
    train_sizes=DEFAULT_SIZES,
    test_sizes=DEFAULT_SIZES,
    train_instances=128,
    test_instances=128,
    capacity=CAPACITY,
):
    output_dir = Path(output_dir)
    written = []
    split_offsets = {"train": 0, "id": 10_000, "ood": 20_000}
    split_specs = (
        ("train", train_sizes, train_instances),
        ("id", test_sizes, test_instances),
        ("ood", test_sizes, test_instances),
    )
    for split, sizes, instance_count in split_specs:
        for size_index, num_items in enumerate(sizes):
            dataset_seed = int(seed + split_offsets[split] + size_index)
            dataset = generate_dataset(
                seed=dataset_seed,
                num_items=int(num_items),
                instances=int(instance_count),
                split=split,
                capacity=int(capacity),
            )
            counts = validate_dataset(
                dataset,
                split=split,
                num_items=int(num_items),
                instances=int(instance_count),
                capacity=int(capacity),
            )
            if split == "train":
                filename = f"dataset_obp_train_size{num_items}.pkl"
            else:
                filename = f"dataset_obp_hidden_{split}_size{num_items}.pkl"
            path = output_dir / filename
            _write_dataset(path, dataset)
            written.append(path)
            print(f"saved {path} instances={len(dataset)} families={counts}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic OBP train/ID/OOD splits.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=32026)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--test-sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--train-instances", type=int, default=128)
    parser.add_argument("--test-instances", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    args = parser.parse_args()
    generate_default_splits(
        args.output_dir,
        seed=args.seed,
        train_sizes=args.train_sizes,
        test_sizes=args.test_sizes,
        train_instances=args.train_instances,
        test_instances=args.test_instances,
        capacity=args.capacity,
    )


if __name__ == "__main__":
    main()
