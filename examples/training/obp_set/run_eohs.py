import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

import yaml

from llm4ad.method.eohs import EoHS, EoHSProfiler
from llm4ad.task.optimization.online_bin_packing_set import OBPSEvaluation
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config():
    return yaml.safe_load((REPO_ROOT / "cfg" / "obp_eohs.yaml").read_text(encoding="utf-8"))


def parse_args(available_sizes, default_size):
    parser = argparse.ArgumentParser(description="Run EoH-S on one OBP train size.")
    parser.add_argument(
        "--train-size",
        type=int,
        choices=available_sizes,
        default=default_size,
        help="Number of items per training instance.",
    )
    return parser.parse_args()


def main():
    cfg = load_config()
    llm_cfg = cfg["llm"]
    task_cfg = cfg["task"]
    datasets_by_size = {int(size): path for size, path in task_cfg["datasets_by_size"].items()}
    args = parse_args(sorted(datasets_by_size), int(task_cfg["default_train_size"]))
    train_dataset = resolve_repo_path(datasets_by_size[args.train_size])
    profiler_cfg = dict(cfg["profiler"])
    profiler_cfg["log_dir"] = str(
        resolve_repo_path(profiler_cfg["log_dir"]) / f"size{args.train_size}"
    )

    llm = OpenAIAPI(
        base_url=os.environ.get(llm_cfg["base_url_env"], llm_cfg["base_url_default"]),
        api_key=os.environ[llm_cfg["api_key_env"]],
        model=os.environ.get(llm_cfg["model_env"], llm_cfg["model_default"]),
        timeout=llm_cfg["timeout"],
    )

    task = OBPSEvaluation(
        timeout_seconds=task_cfg["timeout_seconds"],
        dataset=str(train_dataset),
        return_list=task_cfg["return_list"],
    )
    print(f"Training OBP with size={args.train_size}: {train_dataset}")

    profiler = EoHSProfiler(**profiler_cfg)
    method = EoHS(
        llm=llm,
        profiler=profiler,
        evaluation=task,
        **cfg["method"],
    )

    method.run()
    token_usage = llm.token_usage()
    print(f"Token usage: {token_usage}")
    if profiler._log_dir:
        log_dir = Path(profiler._log_dir)
        (log_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (log_dir / "token_usage.json").write_text(
            json.dumps(token_usage, indent=2), encoding="utf-8"
        )


if __name__ == '__main__':
    main()
