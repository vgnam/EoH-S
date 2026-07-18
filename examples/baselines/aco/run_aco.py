"""Run ACO baselines on this repository's TSP, CVRP, and BPP datasets."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from llm4ad.baselines.aco import (  # noqa: E402
    ACOParameters,
    solve_bpp_aco,
    solve_cvrp_aco,
    solve_tsp_aco,
)


ROUTING_SIZES = (20, 50, 100)
BPP_SIZES = (200, 500, 1000)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _routing_dataset_paths(problem: str, split: str, sizes: list[int] | None) -> list[Path]:
    dataset_dir = REPO_ROOT / "datasets" / problem
    if split == "train":
        if sizes is not None and 30 not in sizes:
            return []
        return sorted(dataset_dir.glob(f"dataset_{problem}_train_fixed30_*_32.pkl"))
    selected_sizes = sizes or list(ROUTING_SIZES)
    return [
        dataset_dir / f"dataset_{problem}_hidden_{split}_size{size}.pkl"
        for size in selected_sizes
    ]


def _bpp_dataset_paths(split: str, sizes: list[int] | None) -> list[Path]:
    selected_sizes = sizes or list(BPP_SIZES)
    dataset_dir = REPO_ROOT / "datasets" / "obp"
    if split == "train":
        return [dataset_dir / f"dataset_obp_train_size{size}.pkl" for size in selected_sizes]
    return [
        dataset_dir / f"dataset_obp_hidden_{split}_size{size}.pkl"
        for size in selected_sizes
    ]


def _load_tsp(path: Path) -> list[tuple[np.ndarray, np.ndarray, float]]:
    dataset = _load_pickle(path)
    if isinstance(dataset, dict) and "rounds" in dataset:
        instances = []
        for hidden_round in dataset["rounds"]:
            for coordinates, baseline in zip(
                hidden_round["coordinates"], hidden_round["baselines"]
            ):
                coordinates = np.asarray(coordinates, dtype=float)
                distances = np.linalg.norm(
                    coordinates[:, None, :] - coordinates[None, :, :], axis=2
                )
                instances.append((coordinates, distances, float(baseline)))
        return instances
    raw_instances = dataset.get("instances", []) if isinstance(dataset, dict) else dataset
    return [
        (np.asarray(coords, dtype=float), np.asarray(distances, dtype=float), float(baseline))
        for coords, distances, baseline in raw_instances
    ]


def _load_cvrp(
    path: Path,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]:
    dataset = _load_pickle(path)
    if isinstance(dataset, dict) and "rounds" in dataset:
        instances = []
        for hidden_round in dataset["rounds"]:
            fields = zip(
                hidden_round["coordinates"],
                hidden_round["demands"],
                hidden_round["capacities"],
                hidden_round["baselines"],
            )
            for coordinates, demands, capacity, baseline in fields:
                coordinates = np.asarray(coordinates, dtype=float)
                distances = np.linalg.norm(
                    coordinates[:, None, :] - coordinates[None, :, :], axis=2
                )
                instances.append(
                    (
                        coordinates,
                        distances,
                        np.asarray(demands, dtype=int),
                        int(capacity),
                        float(baseline),
                    )
                )
        return instances
    raw_instances = dataset.get("instances", []) if isinstance(dataset, dict) else dataset
    return [
        (
            np.asarray(coords, dtype=float),
            np.asarray(distances, dtype=float),
            np.asarray(demands, dtype=int),
            int(capacity),
            float(baseline),
        )
        for coords, distances, demands, capacity, baseline in raw_instances
    ]


def _load_bpp(path: Path) -> list[tuple[str, dict[str, Any]]]:
    dataset = _load_pickle(path)
    if isinstance(dataset, dict) and "instances" in dataset:
        dataset = dataset["instances"]
    if isinstance(dataset, dict):
        return [(str(name), instance) for name, instance in dataset.items()]
    return [(f"instance_{index}", instance) for index, instance in enumerate(dataset)]


def _limit(instances: list[Any], max_instances: int) -> list[Any]:
    return instances if max_instances == 0 else instances[:max_instances]


def _run_tsp(
    path: Path,
    split: str,
    parameters: ACOParameters,
    runs: int,
    seed: int,
    max_instances: int,
) -> list[dict[str, Any]]:
    records = []
    for instance_index, (_coords, distances, baseline) in enumerate(
        _limit(_load_tsp(path), max_instances)
    ):
        for run in range(runs):
            started = time.perf_counter()
            solution = solve_tsp_aco(
                distances,
                parameters=parameters,
                seed=seed + instance_index * 1009 + run,
            )
            records.append(
                {
                    "problem": "tsp",
                    "split": split,
                    "dataset": path.name,
                    "size": len(distances),
                    "instance": instance_index,
                    "run": run,
                    "objective": solution.cost,
                    "reference": baseline,
                    "utility": (baseline - solution.cost) / baseline,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    return records


def _run_cvrp(
    path: Path,
    split: str,
    parameters: ACOParameters,
    runs: int,
    seed: int,
    max_instances: int,
) -> list[dict[str, Any]]:
    records = []
    instances = _limit(_load_cvrp(path), max_instances)
    for instance_index, (_coords, distances, demands, capacity, baseline) in enumerate(instances):
        for run in range(runs):
            started = time.perf_counter()
            solution = solve_cvrp_aco(
                distances,
                demands,
                capacity,
                parameters=parameters,
                seed=seed + instance_index * 1009 + run,
            )
            records.append(
                {
                    "problem": "cvrp",
                    "split": split,
                    "dataset": path.name,
                    "size": len(demands) - 1,
                    "instance": instance_index,
                    "run": run,
                    "objective": solution.cost,
                    "reference": baseline,
                    "utility": (baseline - solution.cost) / baseline,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    return records


def _run_bpp(
    path: Path,
    split: str,
    parameters: ACOParameters,
    runs: int,
    seed: int,
    max_instances: int,
    position_buckets: int,
) -> list[dict[str, Any]]:
    records = []
    for instance_index, (name, instance) in enumerate(_limit(_load_bpp(path), max_instances)):
        items = np.asarray(instance["items"], dtype=int)
        capacity = int(instance["capacity"])
        for run in range(runs):
            started = time.perf_counter()
            solution = solve_bpp_aco(
                items,
                capacity,
                parameters=parameters,
                seed=seed + instance_index * 1009 + run,
                position_buckets=position_buckets,
            )
            gap = (solution.bin_count - solution.lower_bound) / solution.lower_bound
            records.append(
                {
                    "problem": "bpp_offline",
                    "split": split,
                    "dataset": path.name,
                    "size": len(items),
                    "instance": name,
                    "run": run,
                    "objective": solution.bin_count,
                    "reference": solution.lower_bound,
                    "utility": -gap,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    return records


def _summarise(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["problem"], record["split"], record["size"])].append(record)
    summaries = []
    for (problem, split, size), values in sorted(groups.items()):
        objectives = np.asarray([value["objective"] for value in values], dtype=float)
        utilities = np.asarray([value["utility"] for value in values], dtype=float)
        summaries.append(
            {
                "problem": problem,
                "split": split,
                "size": size,
                "observations": len(values),
                "mean_objective": float(np.mean(objectives)),
                "std_objective": float(np.std(objectives)),
                "mean_utility": float(np.mean(utilities)),
                "std_utility": float(np.std(utilities)),
                "mean_elapsed_seconds": float(
                    np.mean([value["elapsed_seconds"] for value in values])
                ),
            }
        )
    return summaries


def _write_results(
    output_path: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "format": "eohs-aco-baselines-v1",
        "source": "https://github.com/FeiLiu36/EoH/tree/main/examples/aco_pheromone",
        "bpp_protocol": "offline item-order ACO with Best Fit decoding",
        "config": serializable_config,
        "summary": summaries,
        "records": records,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("tsp", "cvrp", "bpp", "all"), default="all")
    parser.add_argument("--split", choices=("train", "id", "ood", "all"), default="id")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="Dataset sizes; defaults depend on problem",
    )
    parser.add_argument("--max-instances", type=int, default=16, help="0 means every instance")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ants", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--evaporation", type=float, default=0.1)
    parser.add_argument("--elitist-weight", type=float, default=0.0)
    parser.add_argument("--position-buckets", type=int, default=16, help="BPP only")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_instances < 0 or args.runs < 1:
        raise ValueError("max-instances must be non-negative and runs must be positive")
    parameters = ACOParameters(
        ants=args.ants,
        iterations=args.iterations,
        alpha=args.alpha,
        beta=args.beta,
        evaporation=args.evaporation,
        elitist_weight=args.elitist_weight,
    )
    parameters.validate()
    problems = ("tsp", "cvrp", "bpp") if args.problem == "all" else (args.problem,)
    splits = ("train", "id", "ood") if args.split == "all" else (args.split,)
    records: list[dict[str, Any]] = []

    for problem in problems:
        for split in splits:
            paths = (
                _bpp_dataset_paths(split, args.sizes)
                if problem == "bpp"
                else _routing_dataset_paths(problem, split, args.sizes)
            )
            if not paths:
                print(f"skip {problem}/{split}: no matching dataset size")
                continue
            for path in paths:
                if not path.exists():
                    raise FileNotFoundError(path)
                print(f"run {problem}/{split}: {path.relative_to(REPO_ROOT)}")
                if problem == "tsp":
                    records.extend(
                        _run_tsp(path, split, parameters, args.runs, args.seed, args.max_instances)
                    )
                elif problem == "cvrp":
                    records.extend(
                        _run_cvrp(path, split, parameters, args.runs, args.seed, args.max_instances)
                    )
                else:
                    records.extend(
                        _run_bpp(
                            path,
                            split,
                            parameters,
                            args.runs,
                            args.seed,
                            args.max_instances,
                            args.position_buckets,
                        )
                    )

    summaries = _summarise(records)
    for summary in summaries:
        print(
            "{problem:11s} {split:5s} n={size:4d} objective={mean_objective:.6f} "
            "utility={mean_utility:+.6f}".format(**summary)
        )
    output_path = args.output or (
        REPO_ROOT
        / "results"
        / "aco_baselines"
        / f"aco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    _write_results(output_path, args, records, summaries)
    print(f"saved JSON: {output_path}")
    print(f"saved CSV:  {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
