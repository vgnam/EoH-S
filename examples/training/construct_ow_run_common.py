from __future__ import annotations

import os

from llm4ad.method.eohs import EoHSProfiler
from llm4ad.method.ow_cahd import OWCAHD, OWCAHDConfig
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from construct_run_common import load_config, resolve_repo_path
from ow_cahd_common import OWCAHDLogger
from post_eval_common import run_post_eval


def run_construct_ow_cahd(
    *,
    cfg_name,
    task_tag,
    descriptor,
    validity_fn,
    make_evaluation,
    build_wake_stream,
    hidden_specs,
    hidden_eval_factory,
):
    cfg = load_config(cfg_name)
    llm_cfg = cfg["llm"]
    config = OWCAHDConfig(**dict(cfg["method"]))
    stream_cfg = dict(cfg["stream"])
    hidden_cfg = cfg["hidden_test"]
    llm = OpenAIAPI(
        base_url=os.environ.get(llm_cfg["base_url_env"], llm_cfg["base_url_default"]),
        api_key=os.environ[llm_cfg["api_key_env"]],
        model=os.environ.get(llm_cfg["model_env"], llm_cfg["model_default"]),
        timeout=llm_cfg["timeout"],
        temperature=llm_cfg.get("temperature"),
        max_retries=llm_cfg.get("max_retries"),
    )
    logger = OWCAHDLogger(resolve_repo_path(cfg["logger"]["root"]), task_tag)
    logger.write_config(config, stream_cfg, hidden_cfg)

    def round_profiler(round_id):
        if not config.print_eohs_samples:
            return None
        print(f"\n{task_tag.upper()} OW-CAHD round={round_id} EOHS samples:")
        return EoHSProfiler(log_dir=None, log_style="simple", create_random_path=False)

    method = OWCAHD(
        llm=llm,
        descriptor=descriptor,
        evaluation_factory=lambda instances: make_evaluation(instances),
        validity_fn=validity_fn,
        config=config,
        profiler_factory=round_profiler,
    )
    for round_id, wake_batch in enumerate(build_wake_stream(**stream_cfg)):
        result = method.step(wake_batch, round_id=round_id)
        logger.record_round(result, llm)
        print(
            f"round={result.round_id} novelty={result.novelty_score:.3f} "
            f"accepted={result.accepted_regime} belief={result.belief} "
            f"samples={result.eohs_total_samples_used}/{config.max_sample_nums}"
        )
        if result.eohs_total_samples_used >= config.max_sample_nums:
            break
        if result.eohs_samples_used <= 0:
            raise RuntimeError(f"{task_tag} OW-CAHD made no EOHS sample progress.")

    run_post_eval(
        logger.log_dir,
        "ow_cahd",
        hidden_specs(hidden_cfg),
        hidden_eval_factory(hidden_cfg),
        method.portfolio,
    )
    print(f"{task_tag} OW-CAHD logs written to {logger.log_dir}")
    return method, logger.log_dir
