import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(ROOT/"code"),str(HERE.parent)]
from llm4ad.method.mcts_ahd import MCTS_AHD,MAProfiler
from common import build_task,hidden_eval_factory,hidden_specs
from construct_run_common import run_construct_training
if __name__=="__main__":run_construct_training("fssp_mcts_ahd.yaml","mcts_ahd",build_task,lambda c:MAProfiler(**c),lambda l,p,t,c:MCTS_AHD(llm=l,profiler=p,evaluation=t,**c),hidden_specs,hidden_eval_factory,post_eval_top_k=10)
