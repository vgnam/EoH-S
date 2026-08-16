from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


TRAIN_INSTANCES = 32
ID_INSTANCES = 32
OOD_INSTANCES = 24
SEED = 20260702
OOD_SPECS = ((12, 7), (21, 15), (24, 17))


def _family_name(dimension, weight):
    return f"n{dimension}w{weight}"


def _make_instances(rng, specs, used_seeds):
    instances = []
    for dimension, weight in specs:
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        while seed in used_seeds:
            seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        used_seeds.add(seed)
        instances.append(
            {
                "dimension": int(dimension),
                "weight": int(weight),
                "seed": seed,
            }
        )
    return instances


def _manifest(split, instances):
    family_counts = Counter(_family_name(item["dimension"], item["weight"]) for item in instances)
    return {
        "split": split,
        "families": sorted(family_counts),
        "instances": instances,
        "family_counts": dict(sorted(family_counts.items())),
    }


def _write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(manifest, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _file_entry(root, path, manifest):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "split": manifest["split"],
        "path": path.relative_to(root).as_posix(),
        "n_instances": len(manifest["instances"]),
        "family_counts": manifest["family_counts"],
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def generate_default_manifests(
    output_dir,
    *,
    seed=SEED,
    train_instances=TRAIN_INSTANCES,
    id_instances=ID_INSTANCES,
    ood_instances=OOD_INSTANCES,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    used_seeds = set()

    train_specs = [(15, 10)] * train_instances
    id_specs = [(15, 10)] * id_instances
    ood_specs = []
    per_family = ood_instances // len(OOD_SPECS)
    for _ in range(per_family):
        ood_specs.extend(OOD_SPECS)
    if len(ood_specs) < ood_instances:
        ood_specs.extend(OOD_SPECS[: ood_instances - len(ood_specs)])
    rng.shuffle(ood_specs)

    specs_by_split = {
        "train": train_specs,
        "id": id_specs,
        "ood": ood_specs,
    }
    entries = []
    for split, specs in specs_by_split.items():
        instances = _make_instances(rng, specs, used_seeds)
        manifest = _manifest(split, instances)
        path = output_dir / split / f"asp_{split}.pkl"
        _write_manifest(path, manifest)
        entries.append(_file_entry(output_dir, path, manifest))
        print(f"saved {path} instances={len(instances)} families={manifest['family_counts']}")

    metadata = {
        "task": "admissible_set",
        "seed": int(seed),
        "train_instances": int(train_instances),
        "id_instances": int(id_instances),
        "ood_instances": int(ood_instances),
        "files": entries,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return entries


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic admissible-set manifests.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[5] / "datasets" / "admissible")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-instances", type=int, default=TRAIN_INSTANCES)
    parser.add_argument("--id-instances", type=int, default=ID_INSTANCES)
    parser.add_argument("--ood-instances", type=int, default=OOD_INSTANCES)
    args = parser.parse_args()
    generate_default_manifests(
        args.output_dir,
        seed=args.seed,
        train_instances=args.train_instances,
        id_instances=args.id_instances,
        ood_instances=args.ood_instances,
    )


if __name__ == "__main__":
    main()
