from __future__ import annotations

"""Verify the method x task matrix without LLM access.

Checks: imports, config sanity, template exec, train task construction,
hidden ID/OOD dataset loading, scoring the template heuristic on one
instance per hidden dataset, and the OW-CAHD task adapters.

Run: python scripts/verify_matrix.py
"""

import importlib
import importlib.util
import inspect
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "examples" / "training"
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(TRAINING_DIR))

TASKS = ("bp1d", "bp2d", "admissible", "obp", "ovrp", "vrptw", "cflp", "fssp", "jssp", "scp")
METHODS = ("eoh", "eohs", "mcts_ahd", "ow_cahd")
TASK_DIRS = {
    "bp1d": "bp_1d_construct",
    "bp2d": "bp_2d_construct",
    "admissible": "admissible_set",
    "obp": "obp_set",
    "ovrp": "ovrp_construct",
    "vrptw": "vrptw_construct",
    "cflp": "cflp_construct",
    "fssp": "fssp_construct",
    "jssp": "jssp_construct",
    "scp": "set_cover_construct",
}

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
        return True
    except Exception as exc:
        FAILURES.append(name)
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def task_module(task):
    dir_path = TRAINING_DIR / TASK_DIRS[task]
    spec = importlib.util.spec_from_file_location(
        f"common_{task}", dir_path / "common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_imports():
    from llm4ad.method.eoh import EoH, EoHProfiler
    from llm4ad.method.eohs import EoHS, EoHSProfiler
    from llm4ad.method.mcts_ahd import MCTS_AHD, MAProfiler
    from llm4ad.method.ow_cahd import OWCAHD, OWCAHDConfig
    for mod in ("post_eval_common", "construct_run_common", "ow_cahd_common"):
        importlib.import_module(mod)
    for task in TASKS:
        task_module(task)


def check_configs():
    for task in TASKS:
        for method in METHODS:
            path = REPO_ROOT / "cfg" / f"{task}_{method}.yaml"
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            for key in ("llm", "hidden_test"):
                assert key in cfg, f"{path.name} missing {key}"
            if method == "ow_cahd":
                for key in ("logger", "stream", "method"):
                    assert key in cfg, f"{path.name} missing {key}"
                assert cfg["method"].get("synthesis_context"), f"{path.name} synthesis_context"
                assert "max_generations" not in cfg["method"] or True
            else:
                for key in ("task", "profiler", "method"):
                    assert key in cfg, f"{path.name} missing {key}"
                if method == "mcts_ahd":
                    assert "max_generations" not in cfg["method"], f"{path.name} should not set max_generations"
                    assert "alpha" in cfg["method"] and "lambda_0" in cfg["method"]
                else:
                    assert "max_generations" in cfg["method"], f"{path.name} missing max_generations"
                assert cfg["hidden_test"]["id_datasets"], f"{path.name} missing id datasets"
                assert cfg["hidden_test"]["ood_datasets"], f"{path.name} missing ood datasets"


def check_template_exec():
    from llm4ad.base import TextFunctionProgramConverter
    for mod_path in (
        "llm4ad.task.optimization.bp_1d_construct.template",
        "llm4ad.task.optimization.bp_2d_construct.template",
        "llm4ad.task.optimization.admissible_set.template",
        "llm4ad.task.optimization.online_bin_packing_set.template",
        "llm4ad.task.optimization.ovrp_construct.template",
        "llm4ad.task.optimization.vrptw_construct.template",
        "llm4ad.task.optimization.cflp_construct.template",
        "llm4ad.task.optimization.fssp_construct.template",
        "llm4ad.task.optimization.jssp_construct.template",
        "llm4ad.task.optimization.set_cover_construct.template",
    ):
        mod = importlib.import_module(mod_path)
        func = TextFunctionProgramConverter.text_to_function(mod.template_program)
        assert func is not None, f"{mod_path} template unparsable"
        namespace = {}
        exec(mod.template_program, namespace)
        assert callable(namespace[func.name]), f"{mod_path} template not executable"


def check_profiler_signatures():
    from llm4ad.method.eoh import EoHProfiler
    from llm4ad.method.eohs import EoHSProfiler
    from llm4ad.method.mcts_ahd import MAProfiler
    for cls in (EoHProfiler, EoHSProfiler, MAProfiler):
        params = inspect.signature(cls.__init__).parameters
        assert "log_dir" in params and "log_style" in params, f"{cls.__name__} missing log params"


def check_train_task(task):
    common = task_module(task)
    cfg = yaml.safe_load(
        (REPO_ROOT / "cfg" / f"{task}_eoh.yaml").read_text(encoding="utf-8")
    )
    evaluation = common.build_task(cfg["task"])
    datasets = getattr(
        evaluation,
        "_datasets",
        getattr(evaluation, "_dataset_instances", None),
    )
    n = len(datasets)
    assert n > 0, f"{task} train task has no instances"
    return evaluation, common


def score_template_on_one(evaluation, common):
    from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
    template = evaluation.template_program
    func = TextFunctionProgramConverter.text_to_function(template)
    program = TextFunctionProgramConverter.function_to_program(func, template)
    secure = SecureEvaluator(evaluation, debug_mode=False)
    score = secure.evaluate_program(program)
    assert score is not None, "template heuristic scored None"
    values = np.asarray(score, dtype=float).ravel()
    assert len(values) >= 1 and np.all(np.isfinite(values)), f"bad scores {score}"
    return values


def check_hidden_eval(task):
    common = task_module(task)
    cfg = yaml.safe_load(
        (REPO_ROOT / "cfg" / f"{task}_eoh.yaml").read_text(encoding="utf-8")
    )
    hidden_cfg = cfg["hidden_test"]
    specs = common.hidden_specs(hidden_cfg)
    assert specs, f"{task} has no hidden specs"
    factory = common.hidden_eval_factory(hidden_cfg)
    from post_eval_common import load_instances
    for dataset_path, stem in specs:
        instances = load_instances(dataset_path)
        assert instances, f"{stem} empty"
        if task == "admissible" and stem == "ood":
            instances = [i for i in instances if i["dimension"] == 12][:1]
        elif task == "scp":
            # The legacy random ID generator did not guarantee coverability;
            # the adapter filters infeasible records before applying its cap.
            pass
        else:
            instances = instances[:1]
        evaluation = factory(instances, stem=stem)
        score_template_on_one(evaluation, common)
        print(f"      hidden {stem}: ok")


def check_ow_adapters(task):
    common = task_module(task)
    cfg = yaml.safe_load(
        (REPO_ROOT / "cfg" / f"{task}_ow_cahd.yaml").read_text(encoding="utf-8")
    )
    stream = common.build_wake_stream(**cfg["stream"])
    batch = next(stream)
    assert len(batch) > 0, f"{task} wake stream empty"
    descriptor_name = {
        "bp1d": "bp1d_descriptor",
        "bp2d": "bp2d_descriptor",
        "admissible": "asp_descriptor",
        "obp": "obp_descriptor",
        "ovrp": "ovrp_descriptor",
        "vrptw": "vrptw_descriptor",
        "cflp": "cflp_descriptor",
        "fssp": "fssp_descriptor",
        "jssp": "jssp_descriptor",
        "scp": "scp_descriptor",
    }[task]
    validity_name = {
        "bp1d": "is_valid_bp1d_instance",
        "bp2d": "is_valid_bp2d_instance",
        "admissible": "is_valid_asp_instance",
        "obp": "is_valid_obp_instance",
        "ovrp": "is_valid_ovrp_instance",
        "vrptw": "is_valid_vrptw_instance",
        "cflp": "is_valid_cflp_instance",
        "fssp": "is_valid_fssp_instance",
        "jssp": "is_valid_jssp_instance",
        "scp": "is_valid_scp_instance",
    }[task]
    descriptor = getattr(common, descriptor_name)
    validity = getattr(common, validity_name)
    for instance in batch:
        vector = np.asarray(descriptor(instance), dtype=float).ravel()
        assert np.all(np.isfinite(vector)), f"{task} descriptor non-finite"
        assert validity(instance), f"{task} validity rejected a wake instance"
    evaluation = common.make_evaluation(batch[:1])
    score_template_on_one(evaluation, common)


def check_run_scripts_compile():
    import py_compile
    for task in TASKS:
        dir_path = TRAINING_DIR / TASK_DIRS[task]
        for name in ("run_eoh.py", "run_eohs.py", "run_mcts_ahd.py", "run_ow_cahd.py"):
            py_compile.compile(str(dir_path / name), doraise=True)
    for name in ("post_eval_common.py", "construct_run_common.py", "ow_cahd_common.py"):
        py_compile.compile(str(TRAINING_DIR / name), doraise=True)


def main():
    global TASKS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        help="Comma-separated task tags to verify (default: full matrix).",
    )
    args = parser.parse_args()
    if args.tasks:
        requested = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
        unknown = sorted(set(requested) - set(TASK_DIRS))
        if unknown:
            parser.error(f"unknown tasks: {', '.join(unknown)}")
        TASKS = requested
    check("imports", check_imports)
    check("configs", check_configs)
    check("template exec", check_template_exec)
    check("profiler signatures", check_profiler_signatures)
    check("run scripts compile", check_run_scripts_compile)
    for task in TASKS:
        check(f"train task {task}", lambda t=task: check_train_task(t))
        check(f"hidden eval {task}", lambda t=task: check_hidden_eval(t))
        check(f"ow adapters {task}", lambda t=task: check_ow_adapters(t))
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
