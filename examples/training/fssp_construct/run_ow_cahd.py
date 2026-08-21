import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(ROOT/"code"),str(HERE.parent)]
from common import build_wake_stream,fssp_descriptor,hidden_eval_factory,hidden_specs,is_valid_fssp_instance,make_evaluation
from construct_ow_run_common import run_construct_ow_cahd
if __name__=="__main__":run_construct_ow_cahd(cfg_name="fssp_ow_cahd.yaml",task_tag="fssp",descriptor=fssp_descriptor,validity_fn=is_valid_fssp_instance,make_evaluation=make_evaluation,build_wake_stream=build_wake_stream,hidden_specs=hidden_specs,hidden_eval_factory=hidden_eval_factory)
