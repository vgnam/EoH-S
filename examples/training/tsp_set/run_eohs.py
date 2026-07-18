import sys
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

import yaml

from llm4ad.task.optimization.tsp_construct_set import TSPSEvaluation
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from llm4ad.method.eohs import EoHS,EoHSProfiler
from post_train_open_world_eval import (
    load_hidden_tsp_dataset,
    print_hidden_utility_post_eval,
    save_hidden_utility_post_eval,
)

def load_config(name):
    return yaml.safe_load((REPO_ROOT / "cfg" / name).read_text(encoding="utf-8"))


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def hidden_dataset_paths(hidden_test_cfg):
    paths = hidden_test_cfg.get("datasets")
    if paths is None:
        paths = [hidden_test_cfg["dataset"]]
    return [resolve_repo_path(path) for path in paths]


def hidden_output_prefix(path):
    stem = Path(path).stem
    if stem.startswith("dataset_tsp_hidden_"):
        stem = stem[len("dataset_tsp_hidden_"):]
    return f"post_eval_hidden_{stem}"


def main():
    cfg = load_config("eohs.yaml")
    llm_cfg = cfg["llm"]
    task_cfg = cfg["task"]
    hidden_test_cfg = cfg["hidden_test"]
    profiler_cfg = cfg["profiler"]
    method_cfg = cfg["method"]

    llm = OpenAIAPI(
        base_url=os.environ.get(llm_cfg["base_url_env"], llm_cfg["base_url_default"]),
        api_key=os.environ[llm_cfg["api_key_env"]],
        model=os.environ.get(llm_cfg["model_env"], llm_cfg["model_default"]),
        timeout=llm_cfg["timeout"])

    task = TSPSEvaluation(
        timeout_seconds=task_cfg["timeout_seconds"],
        datasets=[str(resolve_repo_path(path)) for path in task_cfg["datasets"]],
        return_list=task_cfg["return_list"])

    profiler = EoHSProfiler(**profiler_cfg)
    method = EoHS(llm=llm,
                 profiler=profiler,
                 evaluation=task,
                 **method_cfg)

    method.run()
    token_usage = llm.token_usage()
    print(f"Token usage: {token_usage}")
    if profiler._log_dir:
        Path(profiler._log_dir, "run_config.json").write_text(
            json.dumps(cfg, indent=2),
            encoding="utf-8",
        )
        Path(profiler._log_dir, "token_usage.json").write_text(
            json.dumps(token_usage, indent=2),
            encoding="utf-8",
        )
        final_population = method._population.population
        for hidden_dataset_path in hidden_dataset_paths(hidden_test_cfg):
            output_prefix = hidden_output_prefix(hidden_dataset_path)
            try:
                hidden_dataset = load_hidden_tsp_dataset(hidden_dataset_path)
                portfolios_by_round = {
                    int(item["round_id"]): final_population
                    for item in hidden_dataset["rounds"]
                }
                utility_by_size, output_path = save_hidden_utility_post_eval(
                    profiler._log_dir,
                    "eohs",
                    portfolios_by_round,
                    hidden_dataset_path,
                    portfolio_protocol="fixed final EOHS population",
                    round_workers=1 if hidden_test_cfg.get("function_timeout_seconds", 0) else 6,
                    function_timeout_seconds=hidden_test_cfg.get("function_timeout_seconds"),
                    speed_probe_timeout_seconds=hidden_test_cfg.get("speed_probe_timeout_seconds"),
                    output_prefix="post_eval_hidden_utility",
                )
                print_hidden_utility_post_eval("eohs", utility_by_size, output_path)
            except Exception as exc:
                error_path = Path(profiler._log_dir) / f"{output_prefix}_error.json"
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


if __name__ == '__main__':
    main()
