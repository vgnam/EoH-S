from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_STREAM_CONFIG = {"seed": 2026, "rounds": 6, "batch_size": 8, "n_cities": 40}


def pairwise_distances(coords):
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


def tour_cost(distance_matrix, route):
    total = 0.0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i], route[i + 1]]
    total += distance_matrix[route[-1], route[0]]
    return float(total)


def nearest_neighbor_reference(distance_matrix):
    n = len(distance_matrix)
    route = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda node: distance_matrix[current, node])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour_cost(distance_matrix, route)


def make_tsp_instance(coords):
    distance_matrix = pairwise_distances(coords)
    baseline = nearest_neighbor_reference(distance_matrix)
    return coords, distance_matrix, baseline


def sample_uniform(rng, n_cities):
    return rng.random((n_cities, 2))


def sample_cluster(rng, n_cities, n_clusters=4, std=0.06):
    centers = rng.uniform(0.15, 0.85, size=(n_clusters, 2))
    assignment = rng.integers(0, n_clusters, size=n_cities)
    coords = centers[assignment] + rng.normal(0.0, std, size=(n_cities, 2))
    return np.clip(coords, 0.0, 1.0)


def sample_bezier_surprise(rng, n_cities):
    control = rng.uniform(0.08, 0.92, size=(4, 2))
    t = np.sort(rng.random(n_cities))[:, None]
    coords = (
        (1 - t) ** 3 * control[0]
        + 3 * (1 - t) ** 2 * t * control[1]
        + 3 * (1 - t) * t**2 * control[2]
        + t**3 * control[3]
    )
    coords += rng.normal(0.0, 0.025, size=(n_cities, 2))
    return np.clip(coords, 0.0, 1.0)


def build_wake_stream(seed=2026, rounds=6, batch_size=8, n_cities=40):
    rng = np.random.default_rng(seed)
    schedule = ["uniform", "uniform", "cluster", "bezier", "bezier", "cluster"]
    stream = []
    for round_id in range(rounds):
        regime = schedule[round_id % len(schedule)]
        batch = []
        for _ in range(batch_size):
            if regime == "uniform":
                coords = sample_uniform(rng, n_cities)
            elif regime == "cluster":
                coords = sample_cluster(rng, n_cities)
            else:
                coords = sample_bezier_surprise(rng, n_cities)
            batch.append(make_tsp_instance(coords))
        stream.append({"round_id": round_id, "regime": regime, "instances": batch})
    return stream


def score_summary(scores):
    values = np.asarray(scores, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"avg": None, "min": None, "max": None, "n": 0}
    return {
        "avg": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(len(values)),
    }


def function_to_callable(function):
    namespace = {"np": np, "math": math}
    exec(str(function), namespace)
    func = namespace.get(function.name)
    if not callable(func):
        raise ValueError(f"Could not compile callable function {function.name}.")
    return func


def generate_neighborhood_matrix(coords):
    coords = np.asarray(coords)
    n = len(coords)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        matrix[i] = np.argsort(np.linalg.norm(coords[i] - coords, axis=1))
    return matrix


def score_callable_on_instance(heuristic, instance):
    coords, distance_matrix, baseline = instance
    n = len(coords)
    neighbor_matrix = generate_neighborhood_matrix(coords)
    destination_node = 0
    current_node = 0
    route = np.zeros(n, dtype=int)
    for i in range(1, n - 1):
        near_nodes = neighbor_matrix[current_node][1:]
        mask = ~np.isin(near_nodes, route[:i])
        unvisited = near_nodes[mask]
        next_node = heuristic(current_node, destination_node, unvisited, distance_matrix)
        try:
            next_node = int(next_node)
        except Exception:
            return None
        if next_node in route[:i] or next_node not in set(unvisited.tolist()):
            return None
        current_node = next_node
        route[i] = current_node

    mask = ~np.isin(np.arange(n), route[: n - 1])
    route[n - 1] = np.arange(n)[mask][0]
    length = tour_cost(distance_matrix, route.tolist())
    return -length


def baseline_score_on_instance(instance):
    _coords, _distance_matrix, baseline = instance
    return -float(baseline)


def _function_key(function) -> str:
    payload = f"{function.name}\n{str(function)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hidden_function_scores(function, instances, *, seed: int, round_id: int):
    """Evaluate one function reproducibly without exposing hidden instances to training."""
    heuristic = function_to_callable(function)
    function_key = _function_key(function)
    scores = []
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        for instance_id, instance in enumerate(instances):
            seed_material = f"{seed}:{round_id}:{instance_id}:{function_key}".encode("utf-8")
            eval_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
            np.random.seed(eval_seed)
            random.seed(eval_seed)
            score = score_callable_on_instance(heuristic, instance)
            if score is None or not math.isfinite(score):
                return None
            scores.append(float(score))
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    return np.asarray(scores, dtype=float)


def _hidden_function_scores_worker(queue, name, source, instances, seed, round_id):
    try:
        function = _HiddenEvalFunction(name, source)
        scores = _hidden_function_scores(
            function,
            instances,
            seed=seed,
            round_id=round_id,
        )
        if scores is None:
            queue.put(("invalid", None))
        else:
            queue.put(("ok", scores.tolist()))
    except Exception as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _hidden_function_scores_with_timeout(
    function,
    instances,
    *,
    seed: int,
    round_id: int,
    timeout_seconds: float | None,
):
    if timeout_seconds is None or timeout_seconds <= 0:
        return _hidden_function_scores(
            function,
            instances,
            seed=seed,
            round_id=round_id,
        )

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_hidden_function_scores_worker,
        args=(
            queue,
            function.name,
            str(function),
            instances,
            seed,
            round_id,
        ),
    )
    process.start()
    process.join(float(timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join()
        queue.cancel_join_thread()
        queue.close()
        return None

    if queue.empty():
        queue.cancel_join_thread()
        queue.close()
        return None
    status, payload = queue.get()
    queue.cancel_join_thread()
    queue.close()
    if status != "ok":
        return None
    return np.asarray(payload, dtype=float)


def _deduplicate_functions(functions):
    unique = []
    seen = set()
    for original_index, function in enumerate(functions, start=1):
        key = _function_key(function)
        if key in seen:
            continue
        seen.add(key)
        unique.append((original_index, function, key))
    return unique


def _exact_hindsight_portfolio(score_matrix, original_indices, portfolio_size):
    """Return the exact cardinality-constrained hidden-test oracle."""
    n_functions = score_matrix.shape[0]
    if n_functions == 0:
        raise ValueError("Cannot build a hindsight portfolio without valid candidate functions.")
    size = min(int(portfolio_size), n_functions)
    if size <= 0:
        raise ValueError("portfolio_size must be positive.")

    best_value = -math.inf
    best_combination = None
    for combination in itertools.combinations(range(n_functions), size):
        value = float(np.mean(np.max(score_matrix[list(combination)], axis=0)))
        if value > best_value:
            best_value = value
            best_combination = combination

    selected_original_indices = [original_indices[index] for index in best_combination]
    return best_value, selected_original_indices, math.comb(n_functions, size)


def evaluate_empirical_regret(history, stream_config, hidden_test_config, portfolio_size):
    """Compute per-round empirical regret on held-out instances H_t.

    H_t is generated only here, after evolution, with a seed distinct from the
    wake stream. G_t is the candidate pool available to the online portfolio
    selector at round t, and P_t is the selected portfolio stored in history.
    """
    stream_config = dict(stream_config)
    hidden_test_config = dict(hidden_test_config)
    hidden_seed = int(hidden_test_config["seed"])
    if hidden_seed == int(stream_config["seed"]):
        raise ValueError("hidden_test.seed must differ from stream.seed.")

    hidden_stream_config = dict(stream_config)
    hidden_stream_config["seed"] = hidden_seed
    hidden_stream_config["batch_size"] = int(
        hidden_test_config.get("batch_size", stream_config["batch_size"])
    )
    hidden_stream = build_wake_stream(**hidden_stream_config)
    if len(history) != len(hidden_stream):
        raise ValueError(
            f"History has {len(history)} rounds but hidden stream has {len(hidden_stream)} rounds."
        )

    per_round = []
    for result, hidden_round in zip(history, hidden_stream):
        if result.round_id != hidden_round["round_id"]:
            raise ValueError("History and hidden stream round IDs are not aligned.")

        candidates = _deduplicate_functions(result.candidate_pool)
        portfolio = _deduplicate_functions(result.portfolio)
        score_cache = {}

        def hidden_scores(function, key):
            if key not in score_cache:
                try:
                    score_cache[key] = _hidden_function_scores(
                        function,
                        hidden_round["instances"],
                        seed=hidden_seed,
                        round_id=result.round_id,
                    )
                except Exception:
                    score_cache[key] = None
            return score_cache[key]

        valid_candidate_indices = []
        candidate_rows = []
        for original_index, function, key in candidates:
            scores = hidden_scores(function, key)
            if scores is not None:
                valid_candidate_indices.append(original_index)
                candidate_rows.append(scores)
        if not candidate_rows:
            raise RuntimeError(f"Round {result.round_id} has no valid candidates on hidden test H_t.")
        candidate_matrix = np.vstack(candidate_rows)

        valid_portfolio_rows = []
        for _, function, key in portfolio:
            scores = hidden_scores(function, key)
            if scores is not None:
                valid_portfolio_rows.append(scores)
        if not valid_portfolio_rows:
            raise RuntimeError(f"Round {result.round_id} portfolio is invalid on hidden test H_t.")
        portfolio_matrix = np.vstack(valid_portfolio_rows)

        hindsight_utility, oracle_indices, oracle_portfolios_evaluated = _exact_hindsight_portfolio(
            candidate_matrix,
            valid_candidate_indices,
            portfolio_size,
        )
        online_utility = float(np.mean(np.max(portfolio_matrix, axis=0)))
        regret = hindsight_utility - online_utility
        if regret < -1e-9:
            raise RuntimeError(
                f"Round {result.round_id} produced negative regret ({regret}); "
                "P_t is not contained in the evaluated candidate pool G_t."
            )
        if -1e-12 < regret < 0.0:
            regret = 0.0

        per_round.append(
            {
                "round_id": result.round_id,
                "regime": hidden_round["regime"],
                "hidden_instances": len(hidden_round["instances"]),
                "candidate_functions": len(candidates),
                "valid_candidate_functions": len(candidate_rows),
                "portfolio_functions": len(portfolio),
                "valid_portfolio_functions": len(valid_portfolio_rows),
                "portfolio_limit": int(portfolio_size),
                "hindsight_utility": hindsight_utility,
                "online_utility": online_utility,
                "empirical_regret_proxy": regret,
                "oracle_search": "exact",
                "oracle_portfolios_evaluated": oracle_portfolios_evaluated,
                "oracle_candidate_indices": oracle_indices,
            }
        )

    regret_values = np.asarray([row["empirical_regret_proxy"] for row in per_round], dtype=float)
    hindsight_values = np.asarray([row["hindsight_utility"] for row in per_round], dtype=float)
    online_values = np.asarray([row["online_utility"] for row in per_round], dtype=float)
    overall = {
        "rounds": len(per_round),
        "portfolio_limit": int(portfolio_size),
        "avg_empirical_regret_proxy": float(np.mean(regret_values)),
        "cumulative_empirical_regret_proxy": float(np.sum(regret_values)),
        "min_empirical_regret_proxy": float(np.min(regret_values)),
        "max_empirical_regret_proxy": float(np.max(regret_values)),
        "avg_hindsight_utility": float(np.mean(hindsight_values)),
        "avg_online_utility": float(np.mean(online_values)),
        "wake_stream_config": stream_config,
        "hidden_stream_config": hidden_stream_config,
    }
    return per_round, overall


def save_regret_post_eval(
    log_dir: str | Path,
    method_name: str,
    history,
    *,
    stream_config,
    hidden_test_config,
    portfolio_size,
):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    per_round, overall = evaluate_empirical_regret(
        history,
        stream_config,
        hidden_test_config,
        portfolio_size,
    )
    overall = {"method": method_name, **overall}

    per_round_path = log_dir / "post_eval_empirical_regret_per_round.csv"
    csv_rows = []
    for row in per_round:
        csv_row = dict(row)
        csv_row["oracle_candidate_indices"] = json.dumps(row["oracle_candidate_indices"])
        csv_rows.append(csv_row)
    with per_round_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    summary_path = log_dir / "post_eval_empirical_regret_summary.json"
    summary_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    return per_round, overall, {"per_round": per_round_path, "summary": summary_path}


def print_regret_post_eval(method_name: str, per_round, overall, paths):
    print(f"\nPost-train hidden-test empirical regret: {method_name}")
    for row in per_round:
        print(
            f"round={row['round_id']} regime={row['regime']} "
            f"hidden={row['hidden_instances']} "
            f"hindsight={row['hindsight_utility']:.6f} "
            f"online={row['online_utility']:.6f} "
            f"regret={row['empirical_regret_proxy']:.6f}"
        )
    print(
        f"overall avg_empirical_regret={overall['avg_empirical_regret_proxy']:.6f} "
        f"cumulative_regret={overall['cumulative_empirical_regret_proxy']:.6f}"
    )
    print(f"hidden-test regret summary saved to {paths['summary']}")


def load_hidden_tsp_dataset(path: str | Path):
    path = Path(path)
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    if dataset.get("format") != "eohs-open-world-tsp-hidden-coords-v1":
        raise ValueError(f"Unsupported hidden TSP dataset format in {path}.")
    required = {"seed", "city_sizes", "instances_per_size", "schedule", "rounds"}
    missing = required.difference(dataset)
    if missing:
        raise ValueError(f"Hidden TSP dataset is missing fields: {sorted(missing)}")
    return dataset


def _utility_stats(scores, baseline_scores=None):
    values = np.asarray(scores, dtype=float)
    stats = {
        "hidden_utility_mean": float(np.mean(values)),
        "hidden_utility_min": float(np.min(values)),
        "hidden_utility_max": float(np.max(values)),
        "hidden_utility_std": float(np.std(values)),
    }
    if baseline_scores is not None:
        baselines = np.asarray(baseline_scores, dtype=float)
        stats["baseline_improvement_rate"] = float(np.mean(values > baselines))
        stats["baseline_non_worse_rate"] = float(np.mean(values >= baselines))
    return stats


class _HiddenEvalFunction:
    def __init__(self, name, source):
        self.name = name
        self.source = source

    def __str__(self):
        return self.source


def _evaluate_hidden_round(task):
    if len(task) == 5:
        (
            hidden_round,
            portfolio_specs,
            seed,
            city_sizes,
            function_timeout_seconds,
        ) = task
        speed_probe_timeout_seconds = None
    else:
        (
            hidden_round,
            portfolio_specs,
            seed,
            city_sizes,
            function_timeout_seconds,
            speed_probe_timeout_seconds,
        ) = task
    round_id = int(hidden_round["round_id"])
    portfolio = [
        _HiddenEvalFunction(name, source)
        for name, source in portfolio_specs
    ]
    coordinates = hidden_round["coordinates"]
    instances = [make_tsp_instance(np.asarray(coords, dtype=float)) for coords in coordinates]
    instance_sizes = np.asarray([len(instance[0]) for instance in instances], dtype=int)
    largest_size = int(np.max(instance_sizes))
    probe_indices = np.flatnonzero(instance_sizes == largest_size)[:1]
    probe_instances = [instances[int(index)] for index in probe_indices]
    valid_rows = []
    for function in portfolio:
        try:
            if speed_probe_timeout_seconds is not None and speed_probe_timeout_seconds > 0:
                probe_scores = _hidden_function_scores_with_timeout(
                    function,
                    probe_instances,
                    seed=seed,
                    round_id=round_id,
                    timeout_seconds=speed_probe_timeout_seconds,
                )
                if probe_scores is None:
                    continue
            scores = _hidden_function_scores(
                function,
                instances,
                seed=seed,
                round_id=round_id,
            ) if function_timeout_seconds is None else _hidden_function_scores_with_timeout(
                function,
                instances,
                seed=seed,
                round_id=round_id,
                timeout_seconds=function_timeout_seconds,
            )
        except Exception:
            scores = None
        if scores is not None:
            valid_rows.append(scores)
    if not valid_rows:
        raise RuntimeError(f"No valid portfolio functions on hidden round {round_id}.")

    best_scores = np.max(np.vstack(valid_rows), axis=0)
    baseline_scores = np.asarray(
        [baseline_score_on_instance(instance) for instance in instances],
        dtype=float,
    )
    regime = hidden_round["regime"]
    round_row = {
        "round_id": round_id,
        "regime": regime,
        "hidden_instances": len(instances),
        "city_sizes": city_sizes,
        "portfolio_functions": len(portfolio),
        "valid_portfolio_functions": len(valid_rows),
        **_utility_stats(best_scores, baseline_scores),
    }
    size_rows = []
    for size in city_sizes:
        mask = instance_sizes == size
        size_rows.append(
            {
                "round_id": round_id,
                "regime": regime,
                "n_cities": size,
                "hidden_instances": int(np.sum(mask)),
                **_utility_stats(best_scores[mask], baseline_scores[mask]),
            }
        )
    return round_row, size_rows, best_scores.tolist(), baseline_scores.tolist(), instance_sizes.tolist()


def evaluate_hidden_portfolio_utility(
    portfolios_by_round,
    hidden_dataset,
    *,
    round_workers=1,
    function_timeout_seconds: float | None = None,
    speed_probe_timeout_seconds: float | None = None,
):
    """Evaluate final portfolios on held-out H_t without exposing H_t during training."""
    seed = int(hidden_dataset["seed"])
    city_sizes = [int(size) for size in hidden_dataset["city_sizes"]]
    tasks = []

    for hidden_round in hidden_dataset["rounds"]:
        round_id = int(hidden_round["round_id"])
        if round_id not in portfolios_by_round:
            raise ValueError(f"No portfolio supplied for hidden round {round_id}.")
        portfolio = _deduplicate_functions(portfolios_by_round[round_id])
        if not portfolio:
            raise ValueError(f"Portfolio for hidden round {round_id} is empty.")
        portfolio_specs = [
            (function.name, str(function))
            for _, function, _ in portfolio
        ]
        tasks.append(
            (
                hidden_round,
                portfolio_specs,
                seed,
                city_sizes,
                function_timeout_seconds,
                speed_probe_timeout_seconds,
            )
        )

    if round_workers > 1:
        with ProcessPoolExecutor(max_workers=min(round_workers, len(tasks))) as executor:
            results = list(executor.map(_evaluate_hidden_round, tasks))
    else:
        results = [_evaluate_hidden_round(task) for task in tasks]

    per_round = []
    per_round_size = []
    all_scores = []
    all_sizes = []
    all_regimes = []
    all_baseline_scores = []
    for round_row, size_rows, best_scores, baseline_scores, instance_sizes in results:
        per_round.append(round_row)
        per_round_size.extend(size_rows)
        all_scores.extend(best_scores)
        all_baseline_scores.extend(baseline_scores)
        all_sizes.extend(instance_sizes)
        all_regimes.extend([round_row["regime"]] * len(best_scores))

    all_scores_array = np.asarray(all_scores, dtype=float)
    all_baseline_scores_array = np.asarray(all_baseline_scores, dtype=float)
    all_sizes_array = np.asarray(all_sizes, dtype=int)
    all_regimes_array = np.asarray(all_regimes, dtype=object)
    per_size = []
    for size in city_sizes:
        mask = all_sizes_array == size
        per_size.append(
            {
                "n_cities": size,
                "hidden_instances": int(np.sum(mask)),
                **_utility_stats(all_scores_array[mask], all_baseline_scores_array[mask]),
            }
        )

    train_regimes = {"uniform", "cluster", "bezier"}
    known_mask = np.asarray([regime in train_regimes for regime in all_regimes_array], dtype=bool)
    surprise_mask = ~known_mask
    known_utility = (
        float(np.mean(all_scores_array[known_mask]))
        if np.any(known_mask)
        else None
    )
    surprise_utility = (
        float(np.mean(all_scores_array[surprise_mask]))
        if np.any(surprise_mask)
        else None
    )
    summary = {
        "metric": "hidden-test best-of-portfolio raw objective",
        "utility_formula": "mean_x max_h -tour_length_x(h)",
        "rounds": len(per_round),
        "hidden_instances": len(all_scores),
        "city_sizes": city_sizes,
        "instances_per_size_per_round": int(hidden_dataset["instances_per_size"]),
        **_utility_stats(all_scores_array, all_baseline_scores_array),
        "worst_round_utility": float(
            min(row["hidden_utility_mean"] for row in per_round)
        ),
        "known_regime_utility": known_utility,
        "surprise_regime_utility": surprise_utility,
        "hidden_dataset": {
            "format": hidden_dataset["format"],
            "seed": seed,
            "schedule": hidden_dataset["schedule"],
        },
    }
    return per_round, per_round_size, per_size, summary


def save_hidden_utility_post_eval(
    log_dir: str | Path,
    method_name: str,
    portfolios_by_round,
    hidden_dataset_path: str | Path,
    *,
    portfolio_protocol: str,
    round_workers: int = 1,
    function_timeout_seconds: float | None = None,
    speed_probe_timeout_seconds: float | None = None,
):
    log_dir = Path(log_dir)
    hidden_dataset_path = Path(hidden_dataset_path)
    dataset = load_hidden_tsp_dataset(hidden_dataset_path)
    per_round, per_round_size, per_size, summary = evaluate_hidden_portfolio_utility(
        portfolios_by_round,
        dataset,
        round_workers=round_workers,
        function_timeout_seconds=function_timeout_seconds,
        speed_probe_timeout_seconds=speed_probe_timeout_seconds,
    )
    summary = {
        "method": method_name,
        "portfolio_protocol": portfolio_protocol,
        "hidden_dataset_path": str(hidden_dataset_path.resolve()),
        **summary,
    }

    paths = {
        "per_round": log_dir / "post_eval_hidden_utility_per_round.csv",
        "per_round_size": log_dir / "post_eval_hidden_utility_per_round_size.csv",
        "per_size": log_dir / "post_eval_hidden_utility_per_size.csv",
        "summary": log_dir / "post_eval_hidden_utility_summary.json",
    }

    def write_rows(path, rows):
        csv_rows = []
        for row in rows:
            csv_row = dict(row)
            if isinstance(csv_row.get("city_sizes"), list):
                csv_row["city_sizes"] = json.dumps(csv_row["city_sizes"])
            csv_rows.append(csv_row)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    write_rows(paths["per_round"], per_round)
    write_rows(paths["per_round_size"], per_round_size)
    write_rows(paths["per_size"], per_size)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return per_round, per_round_size, per_size, summary, paths


def print_hidden_utility_post_eval(method_name, per_round, summary, paths):
    print(f"\nPost-train hidden portfolio raw objective: {method_name}")
    for row in per_round:
        print(
            f"round={row['round_id']} regime={row['regime']} "
            f"hidden={row['hidden_instances']} utility={row['hidden_utility_mean']:.6f} "
            f"baseline_improve={row['baseline_improvement_rate']:.3f}"
        )
    print(
        f"overall hidden_utility={summary['hidden_utility_mean']:.6f} "
        f"worst_round={summary['worst_round_utility']:.6f} "
        f"baseline_improve={summary['baseline_improvement_rate']:.3f}"
    )
    print(f"hidden utility summary saved to {paths['summary']}")


def evaluate_function_set(functions, stream_config=None):
    stream_config = dict(DEFAULT_STREAM_CONFIG if stream_config is None else stream_config)
    compiled = []
    for idx, function in enumerate(functions, start=1):
        try:
            compiled.append((idx, function, function_to_callable(function)))
        except Exception:
            continue

    per_round = []
    all_instance_scores = []
    for round_item in build_wake_stream(**stream_config):
        instance_best_scores = []
        for instance in round_item["instances"]:
            candidate_scores = []
            for _, _, callable_func in compiled:
                score = score_callable_on_instance(callable_func, instance)
                if score is not None and math.isfinite(score):
                    candidate_scores.append(float(score))
            if candidate_scores:
                instance_best_scores.append(max(candidate_scores))

        summary = score_summary(instance_best_scores)
        all_instance_scores.extend(instance_best_scores)
        per_round.append(
            {
                "round_id": round_item["round_id"],
                "regime": round_item["regime"],
                "instances": len(round_item["instances"]),
                "valid_instances": len(instance_best_scores),
                "best_of_set_avg_score": summary["avg"],
                "best_of_set_min_score": summary["min"],
                "best_of_set_max_score": summary["max"],
            }
        )

    overall = score_summary(all_instance_scores)
    overall.update(
        {
            "rounds": len(per_round),
            "functions": len(functions),
            "compiled_functions": len(compiled),
            "stream_config": stream_config,
        }
    )
    return per_round, overall


def save_post_eval(log_dir: str | Path, method_name: str, functions, stream_config=None):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    per_round, overall = evaluate_function_set(functions, stream_config=stream_config)
    overall = {"method": method_name, **overall}

    per_round_path = log_dir / "post_eval_open_world_per_round.csv"
    with per_round_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_round[0].keys()))
        writer.writeheader()
        writer.writerows(per_round)

    summary_path = log_dir / "post_eval_open_world_summary.json"
    summary_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")

    portfolio_path = log_dir / "post_eval_open_world_functions.json"
    function_rows = []
    for idx, function in enumerate(functions, start=1):
        function_rows.append(
            {
                "rank": idx,
                "name": function.name,
                "original_score_summary": score_summary(function.score)
                if isinstance(function.score, (list, tuple, np.ndarray))
                else function.score,
                "algorithm": getattr(function, "algorithm", ""),
                "function": str(function),
            }
        )
    portfolio_path.write_text(json.dumps(function_rows, indent=2), encoding="utf-8")
    return per_round, overall, {"per_round": per_round_path, "summary": summary_path, "functions": portfolio_path}


def print_post_eval(method_name: str, per_round, overall, paths):
    print(f"\nPost-train open-world evaluation: {method_name}")
    for row in per_round:
        avg = row["best_of_set_avg_score"]
        avg_text = "NA" if avg is None else f"{avg:.6f}"
        print(
            f"round={row['round_id']} regime={row['regime']} "
            f"valid={row['valid_instances']}/{row['instances']} best_of_set_avg={avg_text}"
        )
    avg = overall["avg"]
    avg_text = "NA" if avg is None else f"{avg:.6f}"
    print(
        f"overall best_of_set_avg={avg_text} "
        f"compiled_functions={overall['compiled_functions']}/{overall['functions']}"
    )
    print(f"post-eval summary saved to {paths['summary']}")
