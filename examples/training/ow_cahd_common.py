from __future__ import annotations

"""Shared OW-CAHD run logger used by every task run_ow_cahd.py."""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def score_summary(score):
    if score is None:
        return None
    values = np.asarray(
        score if isinstance(score, (list, tuple, np.ndarray)) else [score],
        dtype=float,
    ).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return {
        "avg": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(len(values)),
    }


def function_record(function, rank):
    return {
        "rank": rank,
        "name": function.name,
        "score_summary": score_summary(function.score),
        "score": function.score,
        "algorithm": getattr(function, "algorithm", ""),
        "function": str(function),
    }


class OWCAHDLogger:
    def __init__(self, root, task_tag, method="ow_cahd"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(root) / task_tag / method / stamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_jsonl = self.log_dir / "history.jsonl"
        self.history_csv = self.log_dir / "history.csv"
        self.summary_json = self.log_dir / "summary.json"
        self._csv_initialized = False
        self._last_token_usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def write_config(self, config, stream_config, hidden_test_config):
        payload = {
            "stream": stream_config,
            "hidden_test": hidden_test_config,
            "config": dict(config.__dict__),
        }
        (self.log_dir / "config.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _save_regime(self, result):
        if result.accepted_regime is None:
            return [], None
        regimes_dir = self.log_dir / "regimes"
        regimes_dir.mkdir(parents=True, exist_ok=True)
        programs = list(result.accepted_regime_generator_programs)
        if not programs and result.accepted_regime_generator_program:
            programs = [result.accepted_regime_generator_program]
        paths = []
        for component_id, program in enumerate(programs):
            suffix = "" if len(programs) == 1 else f"_{component_id:02d}"
            path = regimes_dir / f"{result.accepted_regime}_generator{suffix}.py"
            path.write_text(program, encoding="utf-8")
            paths.append(str(path.relative_to(self.log_dir)))
        metadata = {
            "round_id": result.round_id,
            "name": result.accepted_regime,
            "description": result.accepted_regime_description,
            "generator_paths": paths,
            "mixture_weights": result.accepted_regime_mixture_weights,
            "mixture_n_fit": result.accepted_regime_mixture_n_fit,
            "mixture_temperatures": result.accepted_regime_mixture_temperatures,
            "mixture_mus": result.accepted_regime_mixture_mus,
            "mixture_covs": result.accepted_regime_mixture_covs,
        }
        (regimes_dir / f"{result.accepted_regime}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return paths, paths[0] if paths else None

    def record_round(self, result, llm):
        portfolio = [
            function_record(function, index + 1)
            for index, function in enumerate(result.portfolio)
        ]
        candidates = [
            function_record(function, index + 1)
            for index, function in enumerate(result.candidate_pool)
        ]
        token_usage = llm.token_usage() if hasattr(llm, "token_usage") else {}
        token_delta = {
            key: int(token_usage.get(key, 0)) - int(self._last_token_usage.get(key, 0))
            for key in self._last_token_usage
        }
        self._last_token_usage = dict(token_usage)
        generator_paths, generator_path = self._save_regime(result)
        summaries = [
            item["score_summary"] for item in portfolio
            if item["score_summary"] is not None
        ]
        row = {
            "round_id": result.round_id,
            "novelty_score": float(result.novelty_score),
            "novelty_threshold": float(result.novelty_threshold),
            "novelty_triggered": bool(result.novelty_triggered),
            "accepted_regime": result.accepted_regime,
            "accepted_regime_description": result.accepted_regime_description,
            "regime_generator_path": generator_path,
            "regime_generator_paths": generator_paths,
            "belief": result.belief,
            "sleep_instances": int(result.sleep_instances),
            "eohs_sample_budget": int(result.eohs_sample_budget),
            "eohs_samples_used": int(result.eohs_samples_used),
            "eohs_total_samples_used": int(result.eohs_total_samples_used),
            "candidate_pool_size": len(candidates),
            "portfolio_size": len(portfolio),
            "portfolio_best_avg_score": max(
                (item["avg"] for item in summaries), default=None
            ),
            "token_usage": token_usage,
            "token_delta": token_delta,
        }
        with self.history_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        csv_row = dict(row)
        for field in ("regime_generator_paths", "belief", "token_usage", "token_delta"):
            csv_row[field] = json.dumps(csv_row[field])
        with self.history_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_row))
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(csv_row)
        (self.log_dir / f"portfolio_round_{result.round_id}.json").write_text(
            json.dumps(portfolio, indent=2), encoding="utf-8"
        )
        (self.log_dir / f"candidate_pool_round_{result.round_id}.json").write_text(
            json.dumps(candidates, indent=2), encoding="utf-8"
        )
        self.summary_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
        (self.log_dir / "token_usage.json").write_text(
            json.dumps(token_usage, indent=2), encoding="utf-8"
        )
