from __future__ import annotations

import csv
import hashlib
import math
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cvrp_common import CVRPInMemoryEvaluation, hidden_round_instances, load_hidden_cvrp_dataset


ID_REGIMES = {"uniform", "cluster", "bezier", "grid_holes", "mixed_id"}


class HiddenEvalFunction:
    def __init__(self, name, source):
        self.name = name
        self.source = source

    def __str__(self):
        return self.source


def function_to_callable(function):
    namespace = {"np": np, "math": math}
    exec(str(function), namespace)
    result = namespace.get(function.name)
    if not callable(result):
        raise ValueError(f"Could not compile callable function {function.name}.")
    return result


def function_key(function):
    payload = f"{function.name}\n{str(function)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deduplicate_functions(functions):
    unique = []
    seen = set()
    for function in functions:
        key = function_key(function)
        if key not in seen:
            seen.add(key)
            unique.append(function)
    return unique


def hidden_function_scores(function, instances, *, seed, round_id):
    heuristic = function_to_callable(function)
    key = function_key(function)
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        material = f"{seed}:{round_id}:{key}".encode("utf-8")
        eval_seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
        np.random.seed(eval_seed)
        random.seed(eval_seed)
        evaluation = CVRPInMemoryEvaluation(instances, return_list=True)
        result = evaluation.evaluate(heuristic)
        if result is None or len(result) != len(instances):
            return None
        scores = np.asarray(result, dtype=float)
        if not np.all(np.isfinite(scores)):
            return None
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    return scores


def _score_worker(queue, name, source, instances, seed, round_id):
    try:
        scores = hidden_function_scores(
            HiddenEvalFunction(name, source),
            instances,
            seed=seed,
            round_id=round_id,
        )
        queue.put(("ok", scores.tolist()) if scores is not None else ("invalid", None))
    except Exception as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def hidden_function_scores_with_timeout(function, instances, *, seed, round_id, timeout_seconds):
    if timeout_seconds is None or timeout_seconds <= 0:
        return hidden_function_scores(function, instances, seed=seed, round_id=round_id)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_score_worker,
        args=(queue, function.name, str(function), instances, seed, round_id),
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
    return np.asarray(payload, dtype=float) if status == "ok" else None


def utility_stats(scores):
    scores = np.asarray(scores, dtype=float)
    return {
        "hidden_utility_mean": float(np.mean(scores)),
        "hidden_utility_min": float(np.min(scores)),
        "hidden_utility_max": float(np.max(scores)),
        "hidden_utility_std": float(np.std(scores)),
    }


def _evaluate_hidden_round(task):
    hidden_round, portfolio_specs, seed, customer_sizes, timeout_seconds, probe_timeout = task
    round_id = int(hidden_round["round_id"])
    instances = hidden_round_instances(hidden_round)
    sizes = np.asarray([len(instance[0]) - 1 for instance in instances], dtype=int)
    largest_size = int(np.max(sizes))
    probe_index = int(np.flatnonzero(sizes == largest_size)[0])
    valid_rows = []
    for name, source in portfolio_specs:
        function = HiddenEvalFunction(name, source)
        try:
            if probe_timeout is not None and probe_timeout > 0:
                probe = hidden_function_scores_with_timeout(
                    function,
                    [instances[probe_index]],
                    seed=seed,
                    round_id=round_id,
                    timeout_seconds=probe_timeout,
                )
                if probe is None:
                    continue
            scores = hidden_function_scores_with_timeout(
                function,
                instances,
                seed=seed,
                round_id=round_id,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            scores = None
        if scores is not None:
            valid_rows.append(scores)
    if not valid_rows:
        raise RuntimeError(f"No valid CVRP portfolio functions on hidden round {round_id}.")
    best_scores = np.max(np.vstack(valid_rows), axis=0)
    size_rows = []
    for size in customer_sizes:
        mask = sizes == size
        if np.any(mask):
            size_rows.append(
                {
                    "round_id": round_id,
                    "regime": hidden_round["regime"],
                    "n_customers": int(size),
                    "hidden_instances": int(np.sum(mask)),
                    "portfolio_functions": len(portfolio_specs),
                    "valid_portfolio_functions": len(valid_rows),
                    **utility_stats(best_scores[mask]),
                }
            )
    return size_rows


def evaluate_hidden_portfolio_utility(
    portfolios_by_round,
    hidden_dataset,
    *,
    round_workers=1,
    function_timeout_seconds=None,
    speed_probe_timeout_seconds=None,
):
    customer_sizes = [int(size) for size in hidden_dataset["customer_sizes"]]
    tasks = []
    for hidden_round in hidden_dataset["rounds"]:
        round_id = int(hidden_round["round_id"])
        if round_id not in portfolios_by_round:
            raise ValueError(f"No CVRP portfolio supplied for hidden round {round_id}.")
        portfolio = deduplicate_functions(portfolios_by_round[round_id])
        if not portfolio:
            raise ValueError(f"CVRP portfolio for hidden round {round_id} is empty.")
        tasks.append(
            (
                hidden_round,
                [(function.name, str(function)) for function in portfolio],
                int(hidden_dataset["seed"]),
                customer_sizes,
                function_timeout_seconds,
                speed_probe_timeout_seconds,
            )
        )
    if round_workers > 1:
        with ProcessPoolExecutor(max_workers=min(round_workers, len(tasks))) as executor:
            nested_rows = list(executor.map(_evaluate_hidden_round, tasks))
    else:
        nested_rows = [_evaluate_hidden_round(task) for task in tasks]
    return [row for rows in nested_rows for row in rows]


def save_hidden_utility_post_eval(
    log_dir,
    method_name,
    portfolios_by_round,
    hidden_dataset_path,
    *,
    portfolio_protocol,
    round_workers=1,
    function_timeout_seconds=None,
    speed_probe_timeout_seconds=None,
    output_prefix="post_eval_hidden_utility",
):
    del method_name, portfolio_protocol
    hidden_dataset = load_hidden_cvrp_dataset(hidden_dataset_path)
    size_rows = evaluate_hidden_portfolio_utility(
        portfolios_by_round,
        hidden_dataset,
        round_workers=round_workers,
        function_timeout_seconds=function_timeout_seconds,
        speed_probe_timeout_seconds=speed_probe_timeout_seconds,
    )
    rows = []
    for size in sorted({row["n_customers"] for row in size_rows}):
        output_row = {"n_customers": size}
        for label, is_id in (("id", True), ("ood", False)):
            selected = [
                row
                for row in size_rows
                if row["n_customers"] == size and ((row["regime"] in ID_REGIMES) == is_id)
            ]
            count = sum(row["hidden_instances"] for row in selected)
            output_row[f"{label}_utility_mean"] = (
                sum(row["hidden_utility_mean"] * row["hidden_instances"] for row in selected) / count
                if count
                else None
            )
        rows.append(output_row)

    output_path = Path(log_dir) / f"{output_prefix}.csv"
    merged = {row["n_customers"]: row for row in rows}
    if output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as handle:
            for old_row in csv.DictReader(handle):
                size = int(old_row["n_customers"])
                target = merged.setdefault(
                    size,
                    {"n_customers": size, "id_utility_mean": None, "ood_utility_mean": None},
                )
                for field in ("id_utility_mean", "ood_utility_mean"):
                    if target.get(field) is None and old_row.get(field):
                        target[field] = float(old_row[field])
    rows = [merged[size] for size in sorted(merged)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["n_customers", "id_utility_mean", "ood_utility_mean"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows, output_path


def print_hidden_utility_post_eval(method_name, rows, output_path):
    print(f"\nPost-train CVRP hidden utility by size: {method_name}")
    for row in rows:
        id_mean = row["id_utility_mean"]
        ood_mean = row["ood_utility_mean"]
        id_text = f"{id_mean:.6f}" if id_mean is not None else "n/a"
        ood_text = f"{ood_mean:.6f}" if ood_mean is not None else "n/a"
        print(f"n={row['n_customers']} id={id_text} ood={ood_text}")
    print(f"CVRP hidden utility saved to {output_path}")
