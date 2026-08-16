from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(SCRIPT_DIR.parent))  # examples/training shared modules

from llm4ad.method.eoh import EoH, EoHProfiler
from common import build_task, hidden_specs, hidden_eval_factory
from construct_run_common import run_construct_training


def main():
    run_construct_training(
        cfg_name="obp_eoh.yaml",
        method_label="eoh",
        task_builder=build_task,
        profiler_factory=lambda profiler_cfg: EoHProfiler(**profiler_cfg),
        method_factory=lambda llm, profiler, task, method_cfg: EoH(
            llm=llm,
            profiler=profiler,
            evaluation=task,
            **method_cfg,
        ),
        hidden_specs_fn=hidden_specs,
        hidden_eval_factory_fn=hidden_eval_factory,
    )


if __name__ == "__main__":
    main()
