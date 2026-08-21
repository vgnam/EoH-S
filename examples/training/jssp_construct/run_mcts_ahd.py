import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "code"), str(HERE.parent)]

from llm4ad.method.mcts_ahd import MCTS_AHD, MAProfiler
from common import build_task, hidden_eval_factory, hidden_specs
from construct_run_common import run_construct_training

if __name__ == "__main__":
    run_construct_training("jssp_mcts_ahd.yaml", "mcts_ahd", build_task,
        lambda cfg: MAProfiler(**cfg),
        lambda llm, profiler, task, cfg: MCTS_AHD(llm=llm, profiler=profiler, evaluation=task, **cfg),
        hidden_specs, hidden_eval_factory, post_eval_top_k=10)
