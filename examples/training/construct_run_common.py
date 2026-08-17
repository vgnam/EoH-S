from __future__ import annotations

"""Shared training + post-eval runner for the non-OW methods.

Every run_eoh.py / run_eohs.py / run_mcts_ahd.py in the construct tasks
follows the TSP/CVRP construct protocol:

  1. build the task evaluation over the merged IID train datasets
     (return_list=True),
  2. run the method,
  3. write run_config.json and token_usage.json,
  4. post-evaluate the final population on held-out ID and OOD datasets
     and write post_eval_hidden_<stem>.csv per dataset.
"""

import json
import os
import sys
from pathlib import Path

import yaml

from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from post_eval_common import run_post_eval


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(name):
    return yaml.safe_load((REPO_ROOT / "cfg" / name).read_text(encoding="utf-8"))


def select_top_k(population, top_k):
    """Select the best top_k functions by their training score."""
    scored = [
        func for func in population if getattr(func, "score", None) is not None
    ]
    return sorted(scored, key=lambda func: func.score, reverse=True)[: int(top_k)]


def write_run_artifacts(log_dir, cfg, token_usage):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    (log_dir / "token_usage.json").write_text(
        json.dumps(token_usage, indent=2), encoding="utf-8"
    )


def run_construct_training(
    cfg_name,
    method_label,
    task_builder,
    profiler_factory,
    method_factory,
    hidden_specs_fn,
    hidden_eval_factory_fn,
    post_eval_top_k=None,
):
    """Run one non-OW method end to end on a construct task."""
    cfg = load_config(cfg_name)
    llm_cfg = cfg["llm"]
    task_cfg = cfg["task"]
    hidden_test_cfg = cfg["hidden_test"]
    profiler_cfg = dict(cfg["profiler"])
    profiler_cfg["log_dir"] = str(resolve_repo_path(profiler_cfg["log_dir"]))

    llm = OpenAIAPI(
        base_url=os.environ.get(llm_cfg["base_url_env"], llm_cfg["base_url_default"]),
        api_key=os.environ[llm_cfg["api_key_env"]],
        model=os.environ.get(llm_cfg["model_env"], llm_cfg["model_default"]),
        timeout=llm_cfg["timeout"],
        temperature=llm_cfg.get("temperature"),
    )
    task = task_builder(task_cfg)
    profiler = profiler_factory(profiler_cfg)
    method = method_factory(llm, profiler, task, cfg["method"])

    method.run()
    token_usage = llm.token_usage()
    print(f"Token usage: {token_usage}")
    log_dir = Path(profiler._log_dir) if profiler._log_dir else Path(profiler_cfg["log_dir"])
    write_run_artifacts(log_dir, cfg, token_usage)

    final_population = method._population.population
    if post_eval_top_k is not None:
        final_population = select_top_k(final_population, post_eval_top_k)
        print(
            f"Final {method_label} population: {len(final_population)} functions "
            f"(top {post_eval_top_k} by training score); "
            f"post-evaluating on hidden ID/OOD datasets."
        )
    else:
        print(
            f"Final {method_label} population: {len(final_population)} functions; "
            f"post-evaluating on hidden ID/OOD datasets."
        )
    run_post_eval(
        log_dir,
        method_label,
        hidden_specs_fn(hidden_test_cfg),
        hidden_eval_factory_fn(hidden_test_cfg),
        final_population,
    )
    print(f"{method_label} logs written to {log_dir}")
    return method, log_dir


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "code"))
