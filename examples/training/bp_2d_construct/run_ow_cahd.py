from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(SCRIPT_DIR.parent))  # examples/training shared modules

from llm4ad.method.eohs import EoHSProfiler
from llm4ad.method.ow_cahd import OWCAHD, OWCAHDConfig
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from common import (
    bp2d_descriptor,
    is_valid_bp2d_instance,
    build_wake_stream,
    hidden_eval_factory,
    hidden_specs,
    make_evaluation,
)
from construct_run_common import load_config, resolve_repo_path
from ow_cahd_common import OWCAHDLogger
from post_eval_common import run_post_eval


def main():
    cfg = load_config("bp2d_ow_cahd.yaml")
    llm_cfg = cfg["llm"]
    method_cfg = dict(cfg["method"])
    stream_config = dict(cfg["stream"])
    hidden_test_cfg = cfg["hidden_test"]

    llm = OpenAIAPI(
        base_url=os.environ.get(llm_cfg["base_url_env"], llm_cfg["base_url_default"]),
        api_key=os.environ[llm_cfg["api_key_env"]],
        model=os.environ.get(llm_cfg["model_env"], llm_cfg["model_default"]),
        timeout=llm_cfg["timeout"],
    )
    config = OWCAHDConfig(**method_cfg)
    logger = OWCAHDLogger(resolve_repo_path(cfg["logger"]["root"]), "bp2d")
    logger.write_config(config, stream_config, hidden_test_cfg)

    def round_profiler(round_id):
        if not config.print_eohs_samples:
            return None
        print(f"\nBP2D OW-CAHD round={round_id} EOHS samples:")
        return EoHSProfiler(log_dir=None, log_style="simple", create_random_path=False)

    method = OWCAHD(
        llm=llm,
        descriptor=bp2d_descriptor,
        evaluation_factory=lambda instances: make_evaluation(instances),
        validity_fn=is_valid_bp2d_instance,
        config=config,
        profiler_factory=round_profiler,
    )

    wake_stream = build_wake_stream(**stream_config)
    for round_id, wake_batch in enumerate(wake_stream):
        result = method.step(wake_batch, round_id=round_id)
        logger.record_round(result, llm)
        print(
            f"round={result.round_id} novelty={result.novelty_score:.3f} "
            f"accepted={result.accepted_regime} belief={result.belief} "
            f"samples={result.eohs_total_samples_used}/{config.max_sample_nums} "
            f"tokens={llm.token_usage()} log_dir={logger.log_dir}"
        )
        if result.eohs_total_samples_used >= config.max_sample_nums:
            break
        if result.eohs_samples_used <= 0:
            raise RuntimeError("BP2D OW-CAHD made no EOHS sample progress.")

    final_portfolio = method.portfolio
    run_post_eval(
        logger.log_dir,
        "ow_cahd",
        hidden_specs(hidden_test_cfg),
        hidden_eval_factory(hidden_test_cfg),
        final_portfolio,
    )
    print(f"BP2D OW-CAHD logs written to {logger.log_dir}")


if __name__ == "__main__":
    main()
