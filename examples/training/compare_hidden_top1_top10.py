from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


RUNS = {
    "tsp": {
        "eoh": [
            "20260819_205435",
            "20260820_030339",
            "20260820_041811",
            "20260820_045812",
            "20260821_101857",
        ],
        "eohs": [
            "20260818_204742",
            "20260819_022314",
            "20260819_032723",
            "20260819_234550",
            "20260820_011745",
        ],
        "ow_cahd": [
            "20260818_210558",
            "20260818_211527",
            "20260819_012403",
            "20260819_055108",
            "20260819_094433",
        ],
    },
    "cvrp": {
        "eoh": [
            "20260821_110717",
            "20260821_122903",
            "20260821_132339",
            "20260821_141426",
            "20260821_152002",
        ],
        "eohs": [
            "20260821_114653",
            "20260821_125411",
            "20260821_140130",
            "20260822_002901",
            "20260822_003333",
        ],
        "ow_cahd": [
            "20260821_115446",
            "20260821_161631",
            "20260821_185304",
            "20260821_192509",
            "20260821_212829",
        ],
    },
}


def load_size_rows(path: Path):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key = "n_cities" if "n_cities" in rows[0] else "n_customers"
    return {int(row[key]): row for row in rows}


def optional_float(row, key):
    value = None if row is None else row.get(key)
    return float(value) if value not in (None, "") else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or repo_root / "logs" / "comparisons").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    for task, methods in RUNS.items():
        for method, run_ids in methods.items():
            for run_id in run_ids:
                run_dir = repo_root / "logs" / task / method / run_id
                top1_rows = load_size_rows(run_dir / "post_eval_hidden_top1.csv")
                combined_rows = load_size_rows(run_dir / "post_eval_hidden_utility.csv")
                for size in (20, 50, 100):
                    for split in ("id", "ood"):
                        combined = combined_rows.get(size)
                        top1_gap = optional_float(combined, f"{split}_top1_gap_mean")
                        top10_gap = optional_float(combined, f"{split}_top10_gap_mean")
                        top1_raw = optional_float(combined, f"{split}_top1_raw_mean")
                        top10_raw = optional_float(combined, f"{split}_top10_raw_mean")
                        reference_raw = optional_float(
                            combined, f"{split}_reference_objective_mean"
                        )
                        if top1_gap is None:
                            top1_gap = optional_float(
                                top1_rows.get(size), f"{split}_utility_mean"
                            )
                        if top10_gap is None:
                            top10_gap = optional_float(
                                combined, f"{split}_utility_mean"
                            )
                        raw_rows.append(
                            {
                                "task": task,
                                "method": method,
                                "run": run_id,
                                "size": size,
                                "split": split,
                                "reference_raw": reference_raw,
                                "top1_raw": top1_raw,
                                "top1_gap": top1_gap,
                                "top10_raw": top10_raw,
                                "top10_gap": top10_gap,
                                "gap_gain_top10_minus_top1": (
                                    top10_gap - top1_gap
                                    if top1_gap is not None and top10_gap is not None
                                    else None
                                ),
                                "raw_reduction_top1_minus_top10": (
                                    top1_raw - top10_raw
                                    if top1_raw is not None and top10_raw is not None
                                    else None
                                ),
                                "top1_status": (
                                    "ok" if top1_gap is not None else "timeout_or_missing"
                                ),
                            }
                        )

    raw_path = output_dir / "tsp_cvrp_top1_top10_5runs_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows = []
    for task in RUNS:
        for method in RUNS[task]:
            for size in (20, 50, 100):
                for split in ("id", "ood"):
                    selected = [
                        row
                        for row in raw_rows
                        if row["task"] == task
                        and row["method"] == method
                        and row["size"] == size
                        and row["split"] == split
                    ]
                    output = {
                        "task": task,
                        "method": method,
                        "size": size,
                        "split": split,
                        "runs_requested": len(selected),
                    }
                    for metric in (
                        "reference_raw",
                        "top1_raw",
                        "top1_gap",
                        "top10_raw",
                        "top10_gap",
                        "gap_gain_top10_minus_top1",
                        "raw_reduction_top1_minus_top10",
                    ):
                        values = [float(row[metric]) for row in selected if row[metric] is not None]
                        output[f"{metric}_n"] = len(values)
                        output[f"{metric}_mean"] = statistics.mean(values) if values else None
                        output[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else None
                    summary_rows.append(output)

    summary_path = output_dir / "tsp_cvrp_top1_top10_5runs_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()
