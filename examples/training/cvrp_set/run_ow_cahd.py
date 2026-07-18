import copy
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

import numpy as np
import yaml

from llm4ad.method.eohs import EoHSProfiler
from llm4ad.method.ow_cahd import OWCAHD, OWCAHDConfig
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from cvrp_common import (
    CVRPInMemoryEvaluation,
    build_wake_stream,
    cvrp_descriptor,
    is_valid_cvrp_instance,
    load_hidden_cvrp_dataset,
    resolve_repo_path,
)
from post_train_hidden_eval import print_hidden_utility_post_eval, save_hidden_utility_post_eval


def load_config():
    return yaml.safe_load((REPO_ROOT / "cfg" / "cvrp_ow_cahd.yaml").read_text(encoding="utf-8"))


def score_summary(score):
    if score is None:
        return None
    values = np.asarray(score if isinstance(score, (list, tuple, np.ndarray)) else [score], dtype=float).ravel()
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


class CVRPOWCAHDLogger:
    def __init__(self, root):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(root) / f"{stamp}_cvrp_ow_cahd"
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
        (self.log_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
        portfolio = [function_record(function, index + 1) for index, function in enumerate(result.portfolio)]
        candidates = [
            function_record(function, index + 1) for index, function in enumerate(result.candidate_pool)
        ]
        token_usage = llm.token_usage() if hasattr(llm, "token_usage") else {}
        token_delta = {
            key: int(token_usage.get(key, 0)) - int(self._last_token_usage.get(key, 0))
            for key in self._last_token_usage
        }
        self._last_token_usage = dict(token_usage)
        generator_paths, generator_path = self._save_regime(result)
        summaries = [item["score_summary"] for item in portfolio if item["score_summary"] is not None]
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
            "portfolio_best_avg_score": max((item["avg"] for item in summaries), default=None),
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
        (self.log_dir / "token_usage.json").write_text(json.dumps(token_usage, indent=2), encoding="utf-8")


def hidden_dataset_paths(hidden_test_cfg):
    paths = hidden_test_cfg.get("datasets")
    if paths is None:
        paths = [hidden_test_cfg["dataset"]]
    return [resolve_repo_path(path) for path in paths]


def main():
    cfg = load_config()
    llm_cfg = cfg["llm"]
    method_cfg = dict(cfg["method"])
    stream_config = dict(cfg["stream"])
    hidden_test_config = cfg["hidden_test"]
    wake_stream = build_wake_stream(**stream_config)
    llm = OpenAIAPI(
        base_url=os.environ.get(llm_cfg["base_url_env"], llm_cfg["base_url_default"]),
        api_key=os.environ[llm_cfg["api_key_env"]],
        model=os.environ.get(llm_cfg["model_env"], llm_cfg["model_default"]),
        timeout=llm_cfg["timeout"],
    )
    config = OWCAHDConfig(**method_cfg)
    logger = CVRPOWCAHDLogger(resolve_repo_path(cfg["logger"]["root"]))
    logger.write_config(config, stream_config, hidden_test_config)

    def round_profiler(round_id):
        if not config.print_eohs_samples:
            return None
        print(f"\nCVRP OW-CAHD round={round_id} EOHS samples:")
        return EoHSProfiler(log_dir=None, log_style="simple", create_random_path=False)

    method = OWCAHD(
        llm=llm,
        descriptor=cvrp_descriptor,
        evaluation_factory=lambda instances: CVRPInMemoryEvaluation(
            [copy.deepcopy(instance) for instance in instances], return_list=True
        ),
        validity_fn=is_valid_cvrp_instance,
        config=config,
        profiler_factory=round_profiler,
    )
    for round_id, wake_batch in enumerate(wake_stream):
        result = method.step(wake_batch, round_id=round_id)
        logger.record_round(result, llm)
        print(
            f"round={result.round_id} novelty={result.novelty_score:.3f} "
            f"accepted={result.accepted_regime} samples={result.eohs_total_samples_used}/"
            f"{config.max_sample_nums} tokens={llm.token_usage()} log_dir={logger.log_dir}"
        )
        if result.eohs_total_samples_used >= config.max_sample_nums:
            break
        if result.eohs_samples_used <= 0:
            raise RuntimeError("CVRP OW-CAHD made no EOHS sample progress.")

    final_portfolio = method.portfolio
    for hidden_dataset_path in hidden_dataset_paths(hidden_test_config):
        try:
            hidden_dataset = load_hidden_cvrp_dataset(hidden_dataset_path)
            portfolios_by_round = {
                int(item["round_id"]): final_portfolio for item in hidden_dataset["rounds"]
            }
            rows, output_path = save_hidden_utility_post_eval(
                logger.log_dir,
                "ow_cahd",
                portfolios_by_round,
                hidden_dataset_path,
                portfolio_protocol="fixed final OW-CAHD portfolio",
                round_workers=1,
                function_timeout_seconds=hidden_test_config.get("function_timeout_seconds"),
                speed_probe_timeout_seconds=hidden_test_config.get("speed_probe_timeout_seconds"),
                output_prefix="post_eval_hidden_utility",
            )
            print_hidden_utility_post_eval("ow_cahd", rows, output_path)
        except Exception as exc:
            error_path = logger.log_dir / f"post_eval_{hidden_dataset_path.stem}_error.json"
            error_path.write_text(
                json.dumps(
                    {
                        "hidden_dataset_path": str(hidden_dataset_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Post-eval failed for {hidden_dataset_path}: {type(exc).__name__}: {exc}")
    print(f"CVRP OW-CAHD logs written to {logger.log_dir}")


if __name__ == "__main__":
    main()
