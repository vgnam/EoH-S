from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvrp_common import load_hidden_cvrp_dataset
from post_train_hidden_eval import print_hidden_utility_post_eval, save_hidden_utility_post_eval


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


def load_completed_portfolio(log_dir, method):
    final_path = log_dir / "post_eval_open_world_functions.json"
    if method == "eohs":
        sample_count = sum(
            len(json.loads(path.read_text(encoding="utf-8")))
            for path in (log_dir / "samples").glob("samples_*.json")
        )
        if sample_count < 500:
            raise ValueError(f"CVRP EOHS run is incomplete: found {sample_count}/500 samples.")
        if not final_path.exists():
            populations = sorted(
                (log_dir / "population").glob("pop_*.json"),
                key=lambda path: int(path.stem.split("_")[-1]),
            )
            if not populations:
                raise FileNotFoundError("No final CVRP EOHS population found.")
            final_path = populations[-1]
        return load_rows(final_path), "fixed final EOHS population"

    history_path = log_dir / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
    if not history or int(config["config"].get("max_sample_nums", 0)) < 500:
        raise ValueError("CVRP OW-CAHD run is incomplete.")
    if not final_path.exists():
        final_round = max(int(row["round_id"]) for row in history)
        final_path = log_dir / f"portfolio_round_{final_round}.json"
    return load_rows(final_path), "fixed final OW-CAHD portfolio"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["eohs", "ow_cahd"])
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("hidden_datasets", type=Path, nargs="+")
    parser.add_argument("--function-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--speed-probe-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--output-prefix", default="post_eval_hidden_utility")
    args = parser.parse_args()

    portfolio, protocol = load_completed_portfolio(args.log_dir, args.method)
    for hidden_dataset_path in args.hidden_datasets:
        hidden_dataset = load_hidden_cvrp_dataset(hidden_dataset_path)
        portfolios_by_round = {
            int(item["round_id"]): portfolio for item in hidden_dataset["rounds"]
        }
        rows, output_path = save_hidden_utility_post_eval(
            args.log_dir,
            args.method,
            portfolios_by_round,
            hidden_dataset_path,
            portfolio_protocol=protocol,
            round_workers=1,
            function_timeout_seconds=args.function_timeout_seconds,
            speed_probe_timeout_seconds=args.speed_probe_timeout_seconds,
            output_prefix=args.output_prefix,
        )
        print_hidden_utility_post_eval(args.method, rows, output_path)


if __name__ == "__main__":
    main()
