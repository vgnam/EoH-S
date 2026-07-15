from __future__ import annotations

import argparse
import json
from pathlib import Path

from post_train_open_world_eval import (
    load_hidden_tsp_dataset,
    print_hidden_utility_post_eval,
    save_hidden_utility_post_eval,
)


class LoggedFunction:
    def __init__(self, row):
        self.name = row.get("name", "select_next_node")
        self.algorithm = row.get("algorithm", "")
        self.score = row.get("score")
        self.source = row["function"]

    def __str__(self):
        return self.source


def load_rows(path):
    return [LoggedFunction(row) for row in json.loads(path.read_text(encoding="utf-8"))]


def count_eohs_samples(log_dir):
    return sum(
        len(json.loads(path.read_text(encoding="utf-8")))
        for path in (log_dir / "samples").glob("samples_*.json")
    )


def load_completed_portfolios(log_dir, method, hidden_dataset):
    round_ids = [int(item["round_id"]) for item in hidden_dataset["rounds"]]
    if method == "eohs":
        samples = count_eohs_samples(log_dir)
        if samples < 500:
            raise ValueError(f"EOHS run is incomplete: found only {samples}/500 samples.")
        final_path = log_dir / "post_eval_open_world_functions.json"
        if not final_path.exists():
            population_paths = sorted(
                (log_dir / "population").glob("pop_*.json"),
                key=lambda path: int(path.stem.split("_")[-1]),
            )
            if not population_paths:
                raise FileNotFoundError("No final EOHS population found.")
            final_path = population_paths[-1]
        final_population = load_rows(final_path)
        return (
            {round_id: final_population for round_id in round_ids},
            "fixed final EOHS population",
        )

    history_path = log_dir / "history.jsonl"
    history_rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    if len(history_rows) < len(round_ids):
        raise ValueError(
            f"OW-CAHD run is incomplete: found {len(history_rows)}/{len(round_ids)} rounds."
        )
    config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
    if int(config["config"].get("max_sample_nums", 0)) < 500:
        raise ValueError("OW-CAHD run was not configured for max_sample_nums >= 500.")
    final_path = log_dir / "post_eval_open_world_functions.json"
    if not final_path.exists():
        final_round = max(int(row["round_id"]) for row in history_rows)
        final_path = log_dir / f"portfolio_round_{final_round}.json"
    if not final_path.exists():
        raise FileNotFoundError(f"Missing final OW-CAHD portfolio: {final_path}")
    final_portfolio = load_rows(final_path)
    return (
        {round_id: final_portfolio for round_id in round_ids},
        "fixed final OW-CAHD portfolio",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["eohs", "ow_cahd"])
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("hidden_dataset", type=Path)
    parser.add_argument("--function-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--speed-probe-timeout-seconds", type=float, default=3.0)
    args = parser.parse_args()

    hidden_dataset = load_hidden_tsp_dataset(args.hidden_dataset)
    portfolios, protocol = load_completed_portfolios(
        args.log_dir,
        args.method,
        hidden_dataset,
    )
    per_round, _, _, summary, paths = save_hidden_utility_post_eval(
        args.log_dir,
        args.method,
        portfolios,
        args.hidden_dataset,
        portfolio_protocol=protocol,
        round_workers=1 if args.function_timeout_seconds > 0 else 6,
        function_timeout_seconds=args.function_timeout_seconds,
        speed_probe_timeout_seconds=args.speed_probe_timeout_seconds,
    )
    print_hidden_utility_post_eval(args.method, per_round, summary, paths)


if __name__ == "__main__":
    main()
