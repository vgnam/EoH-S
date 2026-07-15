from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHODS = ("eohs", "eoh", "funsearch", "reevo")
DEFAULT_SCHEDULE = (
    "uniform",
    "cluster",
    "diagonal",
    "bezier_surprise",
    "bezier_surprise",
    "grid",
    "cluster",
    "uniform",
)


@dataclass(frozen=True)
class TSPInstance:
    coords: np.ndarray
    distance_matrix: np.ndarray
    reference_length: float
    regime: str
    round_id: int
    instance_id: int


@dataclass(frozen=True)
class HeuristicSet:
    method: str
    run: str
    source: Path
    functions: tuple[Callable, ...]


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


def tour_length(distance_matrix: np.ndarray, route: list[int]) -> float:
    if not route:
        return math.inf
    total = 0.0
    for i in range(len(route) - 1):
        total += float(distance_matrix[route[i], route[i + 1]])
    total += float(distance_matrix[route[-1], route[0]])
    return total


def nearest_neighbor_route(distance_matrix: np.ndarray, start: int) -> list[int]:
    n = distance_matrix.shape[0]
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        next_node = min(unvisited, key=lambda node: distance_matrix[current, node])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return route


def two_opt(distance_matrix: np.ndarray, route: list[int], max_passes: int = 30) -> list[int]:
    best = route[:]
    best_length = tour_length(distance_matrix, best)
    n = len(best)

    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n):
                if k - i == 1:
                    continue
                candidate = best[:i] + best[i:k][::-1] + best[k:]
                candidate_length = tour_length(distance_matrix, candidate)
                if candidate_length + 1e-12 < best_length:
                    best = candidate
                    best_length = candidate_length
                    improved = True
        if not improved:
            break
    return best


def reference_tour_length(distance_matrix: np.ndarray, starts: int = 8) -> float:
    n = distance_matrix.shape[0]
    start_nodes = np.linspace(0, n - 1, min(starts, n), dtype=int)
    best_length = math.inf
    for start in start_nodes:
        route = nearest_neighbor_route(distance_matrix, int(start))
        route = two_opt(distance_matrix, route)
        best_length = min(best_length, tour_length(distance_matrix, route))
    return best_length


def sample_uniform(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.random((n, 2))


def sample_cluster(rng: np.random.Generator, n: int, centers: int = 4, std: float = 0.055) -> np.ndarray:
    center_xy = rng.uniform(0.15, 0.85, size=(centers, 2))
    assignment = rng.integers(0, centers, size=n)
    coords = center_xy[assignment] + rng.normal(0.0, std, size=(n, 2))
    return np.clip(coords, 0.0, 1.0)


def sample_diagonal(rng: np.random.Generator, n: int) -> np.ndarray:
    t = np.sort(rng.random(n))
    coords = np.column_stack((t, t))
    coords += rng.normal(0.0, 0.035, size=(n, 2))
    return np.clip(coords, 0.0, 1.0)


def sample_grid(rng: np.random.Generator, n: int) -> np.ndarray:
    side = int(math.ceil(math.sqrt(n)))
    values = np.linspace(0.08, 0.92, side)
    grid = np.array([(x, y) for x in values for y in values], dtype=float)
    rng.shuffle(grid)
    coords = grid[:n] + rng.normal(0.0, 0.018, size=(n, 2))
    return np.clip(coords, 0.0, 1.0)


def sample_bezier_surprise(rng: np.random.Generator, n: int) -> np.ndarray:
    control = rng.uniform(0.08, 0.92, size=(4, 2))
    t = np.sort(rng.random(n))[:, None]
    coords = (
        (1 - t) ** 3 * control[0]
        + 3 * (1 - t) ** 2 * t * control[1]
        + 3 * (1 - t) * t**2 * control[2]
        + t**3 * control[3]
    )
    coords += rng.normal(0.0, 0.025, size=(n, 2))
    return np.clip(coords, 0.0, 1.0)


def sample_regime(rng: np.random.Generator, regime: str, n: int) -> np.ndarray:
    samplers = {
        "uniform": sample_uniform,
        "cluster": sample_cluster,
        "diagonal": sample_diagonal,
        "grid": sample_grid,
        "bezier_surprise": sample_bezier_surprise,
    }
    if regime not in samplers:
        raise ValueError(f"Unknown regime: {regime}")
    return samplers[regime](rng, n)


def build_stream(
    seed: int,
    n_cities: int,
    instances_per_round: int,
    schedule: tuple[str, ...],
) -> list[TSPInstance]:
    stream: list[TSPInstance] = []
    rng = np.random.default_rng(seed)
    for round_id, regime in enumerate(schedule):
        for instance_id in range(instances_per_round):
            coords = sample_regime(rng, regime, n_cities)
            distance_matrix = pairwise_distances(coords)
            baseline = reference_tour_length(distance_matrix)
            stream.append(
                TSPInstance(
                    coords=coords,
                    distance_matrix=distance_matrix,
                    reference_length=baseline,
                    regime=regime,
                    round_id=round_id,
                    instance_id=instance_id,
                )
            )
    return stream


def extract_function_blocks(source: str) -> list[str]:
    blocks: list[str] = []
    for chunk in re.split(r"\n# Function\s+\d+", source):
        start = chunk.find("def select_next_node")
        if start < 0:
            continue
        block = chunk[start:].strip()
        block = re.sub(r"\n\s*\{[^{}]*\}\s*$", "", block)
        blocks.append(block)
    return blocks


def load_heuristic_functions(path: Path, max_functions: int) -> tuple[Callable, ...]:
    source = path.read_text(encoding="utf-8", errors="replace")
    functions: list[Callable] = []
    for idx, block in enumerate(extract_function_blocks(source), start=1):
        namespace = {"np": np, "math": math}
        try:
            exec(compile(block, f"{path}#function_{idx}", "exec"), namespace)
        except Exception as exc:
            print(f"skip {path.name} function {idx}: {exc}")
            continue
        func = namespace.get("select_next_node")
        if callable(func):
            functions.append(func)
        if len(functions) >= max_functions:
            break
    return tuple(functions)


def discover_heuristic_sets(methods: tuple[str, ...], max_functions: int) -> list[HeuristicSet]:
    heuristic_dir = REPO_ROOT / "heuristics" / "heuristics"
    sets: list[HeuristicSet] = []
    for method in methods:
        pattern = f"heuristics_tsp_{method}_run*_top10.py"
        for path in sorted(heuristic_dir.glob(pattern)):
            match = re.search(r"_run(\d+)_", path.name)
            run = f"run{match.group(1)}" if match else path.stem
            functions = load_heuristic_functions(path, max_functions=max_functions)
            if not functions:
                print(f"skip {path}: no callable select_next_node functions found")
                continue
            sets.append(HeuristicSet(method=method, run=run, source=path, functions=functions))
    if not sets:
        raise FileNotFoundError("No heuristic files found for the selected methods.")
    return sets


def heuristic_route_length(func: Callable, distance_matrix: np.ndarray, seed: int) -> float:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    current_node = 0
    destination_node = 0
    route = [current_node]
    unvisited = np.arange(1, n, dtype=int)

    while len(unvisited) > 0:
        try:
            next_node = func(current_node, destination_node, unvisited.copy(), distance_matrix)
        except Exception:
            return math.inf

        try:
            next_node = int(next_node)
        except Exception:
            return math.inf

        if next_node not in set(unvisited.tolist()):
            return math.inf

        route.append(next_node)
        unvisited = unvisited[unvisited != next_node]
        current_node = next_node

    return tour_length(distance_matrix, route)


def score_function(func: Callable, instance: TSPInstance, seed: int) -> float:
    length = heuristic_route_length(func, instance.distance_matrix, seed)
    if not math.isfinite(length):
        return -1.0
    return -((length - instance.reference_length) / instance.reference_length)


def score_set(
    heuristic_set: HeuristicSet,
    instance: TSPInstance,
    mode: str,
    seed: int,
) -> float:
    functions = heuristic_set.functions[:1] if mode == "single" else heuristic_set.functions
    scores = [
        score_function(func, instance, seed + 1009 * idx)
        for idx, func in enumerate(functions)
    ]
    return max(scores)


def summarize_run(round_rows: list[dict[str, str]], surprise_round: int, tolerance: float) -> dict[str, str]:
    scores_by_round = {
        int(row["round"]): float(row["score"])
        for row in round_rows
    }
    all_scores = list(scores_by_round.values())
    pre_scores = [score for round_id, score in scores_by_round.items() if round_id < surprise_round]
    pre_score = float(np.mean(pre_scores)) if pre_scores else math.nan
    surprise_score = scores_by_round.get(surprise_round, math.nan)
    dip_depth = max(0.0, pre_score - surprise_score) if math.isfinite(pre_score) else math.nan

    recovery_time = "NA"
    if math.isfinite(pre_score):
        target = pre_score - tolerance
        for round_id in sorted(scores_by_round):
            if round_id >= surprise_round and scores_by_round[round_id] >= target:
                recovery_time = str(round_id - surprise_round)
                break

    first = round_rows[0]
    return {
        "method": first["method"],
        "run": first["run"],
        "mode": first["mode"],
        "n_functions": first["n_functions"],
        "mean_score": f"{float(np.mean(all_scores)):.8f}",
        "pre_score": f"{pre_score:.8f}",
        "surprise_score": f"{surprise_score:.8f}",
        "dip_depth": f"{dip_depth:.8f}",
        "recovery_time": recovery_time,
        "round_score_std": f"{float(np.std(all_scores)):.8f}",
    }


def aggregate_methods(run_summaries: list[dict[str, str]]) -> list[dict[str, str]]:
    methods = sorted({row["method"] for row in run_summaries})
    metrics = ("mean_score", "pre_score", "surprise_score", "dip_depth", "round_score_std")
    output = []
    for method in methods:
        rows = [row for row in run_summaries if row["method"] == method]
        summary = {"method": method, "runs": str(len(rows)), "mode": rows[0]["mode"]}
        for metric in metrics:
            values = np.array([float(row[metric]) for row in rows], dtype=float)
            summary[f"{metric}_mean"] = f"{float(np.mean(values)):.8f}"
            summary[f"{metric}_std"] = f"{float(np.std(values)):.8f}"
        recovered = [row["recovery_time"] for row in rows if row["recovery_time"] != "NA"]
        summary["recovered_runs"] = str(len(recovered))
        summary["recovery_time_mean"] = (
            f"{float(np.mean([int(value) for value in recovered])):.8f}" if recovered else "NA"
        )
        output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the open-world TSP surprise benchmark.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(DEFAULT_METHODS))
    parser.add_argument("--mode", choices=("portfolio", "single"), default="portfolio")
    parser.add_argument("--max-functions", type=int, default=10)
    parser.add_argument("--n-cities", type=int, default=40)
    parser.add_argument("--instances-per-round", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--recovery-tolerance", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "open_world_tsp_surprise")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = tuple(args.methods)
    schedule = DEFAULT_SCHEDULE
    surprise_round = schedule.index("bezier_surprise")

    print("building stream...")
    stream = build_stream(
        seed=args.seed,
        n_cities=args.n_cities,
        instances_per_round=args.instances_per_round,
        schedule=schedule,
    )

    print("loading heuristics...")
    heuristic_sets = discover_heuristic_sets(methods, max_functions=args.max_functions)

    per_round_rows: list[dict[str, str]] = []
    run_summaries: list[dict[str, str]] = []

    for heuristic_set in heuristic_sets:
        print(
            f"evaluating {heuristic_set.method}/{heuristic_set.run} "
            f"({len(heuristic_set.functions)} functions)"
        )
        rows_for_run: list[dict[str, str]] = []
        for round_id, regime in enumerate(schedule):
            instances = [item for item in stream if item.round_id == round_id]
            instance_scores = [
                score_set(
                    heuristic_set,
                    instance,
                    mode=args.mode,
                    seed=args.seed + 100000 * round_id + instance.instance_id,
                )
                for instance in instances
            ]
            row = {
                "method": heuristic_set.method,
                "run": heuristic_set.run,
                "mode": args.mode,
                "n_functions": str(1 if args.mode == "single" else len(heuristic_set.functions)),
                "round": str(round_id),
                "regime": regime,
                "phase": "surprise" if regime == "bezier_surprise" else "known",
                "score": f"{float(np.mean(instance_scores)):.8f}",
            }
            rows_for_run.append(row)
            per_round_rows.append(row)
        run_summaries.append(
            summarize_run(
                rows_for_run,
                surprise_round=surprise_round,
                tolerance=args.recovery_tolerance,
            )
        )

    method_summary = aggregate_methods(run_summaries)
    write_csv(args.output_dir / "per_round.csv", per_round_rows)
    write_csv(args.output_dir / "run_summary.csv", run_summaries)
    write_csv(args.output_dir / "method_summary.csv", method_summary)

    print(f"wrote {args.output_dir / 'per_round.csv'}")
    print(f"wrote {args.output_dir / 'run_summary.csv'}")
    print(f"wrote {args.output_dir / 'method_summary.csv'}")


if __name__ == "__main__":
    main()
