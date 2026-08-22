from __future__ import annotations

"""Shared post-training IID/OOD evaluation helpers.

Every construct task in examples/training follows the TSP/CVRP construct
protocol: train on merged IID train datasets, then evaluate the final
population/portfolio on held-out ID (test) and OOD datasets and write
post_eval_hidden_<stem>.csv plus a JSON summary per dataset.

This module only depends on llm4ad.base; the task-specific pieces
(instance loading/normalization and the evaluation factory) are provided by
each task common.py.
"""

import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy 2.x pickles with a NumPy 1.x runtime.

    NumPy 2 moved private modules from ``numpy.core`` to ``numpy._core``.
    Pickles created by NumPy 2 therefore cannot normally be opened by NumPy
    1.x even though the serialized array representation is compatible.
    """

    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_env_file(path=None):
    """Load KEY=VALUE pairs from a .env file into os.environ (no dependency).

    Values already present in the environment are NOT overwritten, so
    explicit shell variables keep precedence. Defaults to <repo root>/.env.
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / ".env"
    path = Path(path)
    if not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


# Every construct run script imports this module, so loading the local .env
# here makes API credentials available without extra setup.
load_env_file()


def resolve_repo_path(path):
    """Resolve a config path against the repository root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def load_instances(path):
    """Load instances from a dataset pkl into a plain list.

    Handles the four dataset layouts in this repo:
      * BP1D/BP2D pkls: list of (items, capacity/bin_dims) tuples
      * OBP train/id/ood pkls: dict name -> instance dict
      * admissible manifests: dict with an "instances" list of
        {dimension, weight, seed} records
    """
    path = Path(path)
    with path.open("rb") as handle:
        data = _NumpyCompatUnpickler(handle).load()
    if isinstance(data, dict):
        if "instances" in data and not isinstance(data.get("split"), str):
            instances = data["instances"]
            if isinstance(instances, dict):
                return list(instances.values())
            return list(instances)
        if "split" in data and "instances" in data:
            return list(data["instances"])
        return list(data.values())
    return list(data)


def _raw_objectives(gap_scores, reference_objectives):
    gaps = np.asarray(gap_scores, dtype=float)
    references = np.asarray(reference_objectives, dtype=float)
    return references * (1.0 - gaps)


def score_functions(functions, evaluation, reference_objectives=None):
    """Score every function on an in-memory evaluation (return_list=True).

    Returns a list of rows with keys index, name, scores, mean, min, max,
    std; functions whose evaluation fails are reported with scores=None.
    """
    secure_evaluator = SecureEvaluator(evaluation, debug_mode=False)
    rows = []
    for index, function in enumerate(functions):
        program = TextFunctionProgramConverter.function_to_program(
            function, evaluation.template_program
        )
        score = None
        if program is not None:
            try:
                score = secure_evaluator.evaluate_program(program)
            except Exception:
                score = None
        if score is None:
            rows.append({
                "index": index,
                "name": function.name,
                "scores": None,
                "mean": None,
                "min": None,
                "max": None,
                "std": None,
            })
            continue
        values = np.asarray(score, dtype=float).ravel()
        if len(values) == 0 or not np.all(np.isfinite(values)):
            rows.append({
                "index": index,
                "name": function.name,
                "scores": None,
                "mean": None,
                "min": None,
                "max": None,
                "std": None,
            })
            continue
        rows.append({
            "index": index,
            "name": function.name,
            "scores": [float(value) for value in values],
            "gap_scores": [float(value) for value in values],
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        })
        if reference_objectives is not None:
            references = np.asarray(reference_objectives, dtype=float)
            if len(values) != len(references):
                raise ValueError(
                    f"Function returned {len(values)} scores for {len(references)} instances."
                )
            raw = _raw_objectives(values, references)
            rows[-1].update({
                "raw_objectives": raw.tolist(),
                "raw_mean": float(np.mean(raw)),
                "raw_min": float(np.min(raw)),
                "raw_max": float(np.max(raw)),
                "raw_std": float(np.std(raw)),
            })
    return rows


def post_eval_dataset(
    log_dir,
    method_label,
    dataset_path,
    stem,
    evaluation_factory,
    functions,
):
    """Evaluate the final population on one hidden dataset and persist.

    Returns a summary dict. Writes post_eval_hidden_<stem>.csv and
    post_eval_hidden_<stem>.json into log_dir.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset_path)
    instances = load_instances(dataset_path)
    if not instances:
        raise ValueError(
            f"Hidden dataset {dataset_path} contains no instances."
        )

    evaluation = evaluation_factory(instances, stem=stem)
    evaluated_instances = int(getattr(evaluation, "n_instance", len(instances)))
    reference_objectives = getattr(evaluation, "_baselines", None)
    if reference_objectives is not None:
        reference_objectives = np.asarray(
            reference_objectives[:evaluated_instances], dtype=float
        )
        if (
            len(reference_objectives) != evaluated_instances
            or not np.all(np.isfinite(reference_objectives))
            or np.any(reference_objectives <= 0)
        ):
            reference_objectives = None
    rows = score_functions(functions, evaluation, reference_objectives)
    valid_rows = [row for row in rows if row["scores"] is not None]
    if not valid_rows:
        raise RuntimeError(
            f"No valid {method_label} functions on hidden dataset {dataset_path}."
        )

    best_row = max(valid_rows, key=lambda row: row["mean"])
    top1_row = rows[0] if rows and rows[0]["scores"] is not None else None
    top10_rows = [row for row in rows[:10] if row["scores"] is not None]
    if not top10_rows:
        raise RuntimeError(
            f"No valid top-10 {method_label} functions on hidden dataset {dataset_path}."
        )
    top10_gaps = np.max(
        np.vstack([row["scores"] for row in top10_rows]), axis=0
    )
    summary = {
        "method": method_label,
        "dataset": str(dataset_path),
        "stem": stem,
        "n_instances": evaluated_instances,
        "n_functions": len(functions),
        "n_valid_functions": len(valid_rows),
        "best_function_index": best_row["index"],
        "best_function_name": best_row["name"],
        "best_mean": best_row["mean"],
        "best_min": best_row["min"],
        "best_max": best_row["max"],
        "top1_function_index": top1_row["index"] if top1_row else None,
        "top1_function_name": top1_row["name"] if top1_row else None,
        "top1_gap_mean": top1_row["mean"] if top1_row else None,
        "top10_functions": len(top10_rows),
        "top10_gap_mean": float(np.mean(top10_gaps)),
    }
    if reference_objectives is not None:
        top10_raw = _raw_objectives(top10_gaps, reference_objectives)
        summary.update({
            "reference_raw_mean": float(np.mean(reference_objectives)),
            "top1_raw_mean": top1_row["raw_mean"] if top1_row else None,
            "top10_raw_mean": float(np.mean(top10_raw)),
            "top1_gap_scores": top1_row["gap_scores"] if top1_row else None,
            "top1_raw_objectives": top1_row["raw_objectives"] if top1_row else None,
            "top10_gap_scores": top10_gaps.tolist(),
            "top10_raw_objectives": top10_raw.tolist(),
            "reference_objectives": reference_objectives.tolist(),
        })

    csv_path = log_dir / f"post_eval_hidden_{stem}.csv"
    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        for field in ("scores", "gap_scores", "raw_objectives"):
            if field in csv_row:
                csv_row[field] = (
                    json.dumps(row[field]) if row[field] is not None else ""
                )
        csv_rows.append(csv_row)
    fieldnames = []
    for row in csv_rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path = log_dir / f"post_eval_hidden_{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = log_dir / f"post_eval_hidden_{stem}_top1_top10.csv"
    report_fields = [
        "method", "stem", "n_instances", "top10_functions",
        "reference_raw_mean", "top1_raw_mean", "top1_gap_mean",
        "top10_raw_mean", "top10_gap_mean",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_fields)
        writer.writeheader()
        writer.writerow({field: summary.get(field) for field in report_fields})
    return summary, csv_path


def run_post_eval(
    log_dir,
    method_label,
    specs,
    evaluation_factory,
    functions,
):
    """Run post-eval over a list of (dataset_path, stem) specs.

    Returns a list of summaries; writes one CSV/JSON pair per dataset.
    """
    summaries = []
    for dataset_path, stem in specs:
        try:
            summary, csv_path = post_eval_dataset(
                log_dir,
                method_label,
                dataset_path,
                stem,
                evaluation_factory,
                functions,
            )
            summaries.append(summary)
            print(
                f"[{method_label}] {stem}: instances={summary['n_instances']} "
                f"valid={summary['n_valid_functions']}/{summary['n_functions']} "
                f"top1_gap={summary['top1_gap_mean']} "
                f"top10_gap={summary['top10_gap_mean']:.6f} "
                f"top1_raw={summary.get('top1_raw_mean')} "
                f"top10_raw={summary.get('top10_raw_mean')} -> {csv_path}"
            )
        except Exception as exc:
            error_path = Path(log_dir) / f"post_eval_hidden_{stem}_error.json"
            error_path.write_text(
                json.dumps(
                    {
                        "dataset": str(dataset_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"[{method_label}] post-eval failed for {dataset_path}: "
                f"{type(exc).__name__}: {exc}"
            )
    return summaries
