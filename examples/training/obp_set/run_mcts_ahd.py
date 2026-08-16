from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(SCRIPT_DIR.parent))  # examples/training shared modules

from llm4ad.method.mcts_ahd import MCTS_AHD, MAProfiler
from common import build_task, hidden_specs, hidden_eval_factory
from construct_run_common import run_construct_training


def main():
    run_construct_training(
        cfg_name="obp_mcts_ahd.yaml",
        method_label="mcts_ahd",
        task_builder=build_task,
        profiler_factory=lambda profiler_cfg: MAProfiler(**profiler_cfg),
        method_factory=lambda llm, profiler, task, method_cfg: MCTS_AHD(
            llm=llm,
            profiler=profiler,
            evaluation=task,
            **method_cfg,
        ),
        hidden_specs_fn=hidden_specs,
        hidden_eval_factory_fn=hidden_eval_factory,
        post_eval_top_k=10,
    )


if __name__ == "__main__":
    main()
