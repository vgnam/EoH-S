from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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


def select_top_train(functions, top_k=10):
    """Rank functions by scalar/mean training gap before hidden evaluation."""
    scored = []
    for function in functions:
        score = getattr(function, "score", None)
        if score is None:
            continue
        values = np.asarray(score, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if len(values):
            scored.append((float(np.mean(values)), function))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [function for _score, function in scored[: int(top_k)]]


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


def raw_objectives(gap_scores, reference_objectives):
    gaps = np.asarray(gap_scores, dtype=float)
    references = np.asarray(reference_objectives, dtype=float)
    return references * (1.0 - gaps)


def raw_stats(gap_scores, reference_objectives):
    objectives = raw_objectives(gap_scores, reference_objectives)
    references = np.asarray(reference_objectives, dtype=float)
    return {
        "raw_objective_mean": float(np.mean(objectives)),
        "raw_objective_min": float(np.min(objectives)),
        "raw_objective_max": float(np.max(objectives)),
        "raw_objective_std": float(np.std(objectives)),
        "reference_objective_mean": float(np.mean(references)),
    }


def _evaluate_hidden_round(task):
    hidden_round, portfolio_specs, seed, customer_sizes, timeout_seconds, probe_timeout = task
    round_id = int(hidden_round["round_id"])
    instances = hidden_round_instances(hidden_round)
    sizes = np.asarray([len(instance[0]) - 1 for instance in instances], dtype=int)
    largest_size = int(np.max(sizes))
    probe_index = int(np.flatnonzero(sizes == largest_size)[0])
    def evaluate_function(spec):
        name, source = spec
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
                    return None
            scores = hidden_function_scores_with_timeout(
                function,
                instances,
                seed=seed,
                round_id=round_id,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            scores = None
        return scores

    if timeout_seconds is not None and len(portfolio_specs) > 1:
        with ThreadPoolExecutor(max_workers=len(portfolio_specs)) as executor:
            evaluated_rows = list(executor.map(evaluate_function, portfolio_specs))
    else:
        evaluated_rows = [evaluate_function(spec) for spec in portfolio_specs]
    valid_top10_rows = [scores for scores in evaluated_rows[:10] if scores is not None]
    if not valid_top10_rows:
        raise RuntimeError(f"No valid CVRP portfolio functions on hidden round {round_id}.")
    top1_scores = evaluated_rows[0]
    top10_scores = np.max(np.vstack(valid_top10_rows), axis=0)
    best_scores = top10_scores
    reference_objectives = np.asarray(
        [float(instance[4]) for instance in instances], dtype=float
    )
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
                    "valid_portfolio_functions": len(valid_top10_rows),
                    **utility_stats(best_scores[mask]),
                    **raw_stats(best_scores[mask], reference_objectives[mask]),
                    "top1_gap_mean": (
                        float(np.mean(top1_scores[mask]))
                        if top1_scores is not None else None
                    ),
                    "top1_raw_mean": (
                        float(np.mean(raw_objectives(
                            top1_scores[mask], reference_objectives[mask]
                        )))
                        if top1_scores is not None else None
                    ),
                    "top10_gap_mean": float(np.mean(top10_scores[mask])),
                    "top10_raw_mean": float(np.mean(raw_objectives(
                        top10_scores[mask], reference_objectives[mask]
                    ))),
                }
            )
    raw_round = {
        "round_id": round_id,
        "regime": hidden_round["regime"],
        "instance_sizes": sizes.tolist(),
        "functions": [
            {
                "index": index,
                "name": name,
                "function_sha256": hashlib.sha256(
                    f"{name}\n{source}".encode("utf-8")
                ).hexdigest(),
                "valid": scores is not None,
                "scores": scores.tolist() if scores is not None else None,
                "gap_scores": scores.tolist() if scores is not None else None,
                "raw_objectives": (
                    raw_objectives(scores, reference_objectives).tolist()
                    if scores is not None else None
                ),
            }
            for index, ((name, source), scores) in enumerate(
                zip(portfolio_specs, evaluated_rows)
            )
        ],
        "top1_gap_scores": top1_scores.tolist() if top1_scores is not None else None,
        "top1_raw_objectives": (
            raw_objectives(top1_scores, reference_objectives).tolist()
            if top1_scores is not None else None
        ),
        "top10_gap_scores": top10_scores.tolist(),
        "top10_raw_objectives": raw_objectives(
            top10_scores, reference_objectives
        ).tolist(),
        "oracle_scores": top10_scores.tolist(),
        "reference_objectives": reference_objectives.tolist(),
    }
    return size_rows, raw_round


def evaluate_hidden_portfolio_utility(
    portfolios_by_round,
    hidden_dataset,
    *,
    round_workers=1,
    function_timeout_seconds=None,
    speed_probe_timeout_seconds=None,
    return_raw=False,
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
            results = list(executor.map(_evaluate_hidden_round, tasks))
    else:
        results = [_evaluate_hidden_round(task) for task in tasks]
    size_rows = [row for rows, _raw_round in results for row in rows]
    raw_rounds = [raw_round for _rows, raw_round in results]
    return (size_rows, raw_rounds) if return_raw else size_rows


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
    hidden_dataset = load_hidden_cvrp_dataset(hidden_dataset_path)
    size_rows, raw_rounds = evaluate_hidden_portfolio_utility(
        portfolios_by_round,
        hidden_dataset,
        round_workers=round_workers,
        function_timeout_seconds=function_timeout_seconds,
        speed_probe_timeout_seconds=speed_probe_timeout_seconds,
        return_raw=True,
    )
    report_metrics = (
        "top1_gap_mean",
        "top1_raw_mean",
        "top10_gap_mean",
        "top10_raw_mean",
        "reference_objective_mean",
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
            for metric in report_metrics:
                metric_rows = [row for row in selected if row.get(metric) is not None]
                metric_count = sum(row["hidden_instances"] for row in metric_rows)
                output_row[f"{label}_{metric}"] = (
                    sum(row[metric] * row["hidden_instances"] for row in metric_rows)
                    / metric_count
                    if metric_count else None
                )
            # Backward-compatible alias: utility is the best-of-top-10 gap.
            output_row[f"{label}_utility_mean"] = output_row[f"{label}_top10_gap_mean"]
        rows.append(output_row)

    output_path = Path(log_dir) / f"{output_prefix}.csv"
    merged = {row["n_customers"]: row for row in rows}
    if output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as handle:
            for old_row in csv.DictReader(handle):
                size = int(old_row["n_customers"])
                target = merged.setdefault(
                    size,
                    {"n_customers": size},
                )
                fields = [
                    f"{label}_{metric}"
                    for label in ("id", "ood")
                    for metric in report_metrics
                ] + ["id_utility_mean", "ood_utility_mean"]
                for field in fields:
                    if target.get(field) is None and old_row.get(field):
                        target[field] = float(old_row[field])
    rows = [merged[size] for size in sorted(merged)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["n_customers"] + [
                f"{label}_{metric}"
                for label in ("id", "ood")
                for metric in report_metrics
            ] + ["id_utility_mean", "ood_utility_mean"],
        )
        writer.writeheader()
        writer.writerows(rows)

    raw_path = Path(log_dir) / (
        f"{output_prefix}_raw_{Path(hidden_dataset_path).stem}.json"
    )
    raw_payload = {
        "format": "hidden-portfolio-objectives-v2",
        "task": "cvrp",
        "method": method_name,
        "portfolio_protocol": portfolio_protocol,
        "hidden_dataset": str(Path(hidden_dataset_path)),
        "hidden_dataset_seed": int(hidden_dataset["seed"]),
        "rounds": raw_rounds,
    }
    raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    return rows, output_path


def print_hidden_utility_post_eval(method_name, rows, output_path):
    print(f"\nPost-train CVRP hidden raw objective and gap by size: {method_name}")
    for row in rows:
        def metric_text(label):
            values = [row.get(f"{label}_{metric}") for metric in (
                "top1_raw_mean", "top1_gap_mean", "top10_raw_mean", "top10_gap_mean"
            )]
            if any(value is None for value in values):
                return "n/a"
            return (
                f"top1(raw={values[0]:.6f}, gap={values[1]:+.6f}) "
                f"top10(raw={values[2]:.6f}, gap={values[3]:+.6f})"
            )
        print(
            f"n={row['n_customers']} id=[{metric_text('id')}] "
            f"ood=[{metric_text('ood')}]"
        )
    print(f"CVRP hidden report saved to {output_path}")
