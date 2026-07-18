import sys
import copy
import csv
import json
import math
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
LOCAL_PYTHON_PACKAGES = REPO_ROOT / ".python_packages"
if LOCAL_PYTHON_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PYTHON_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "code"))

import numpy as np
import yaml

from llm4ad.base import Evaluation
from llm4ad.method.eohs import EoHSProfiler
from llm4ad.method.ow_cahd import OWCAHD, OWCAHDConfig
from llm4ad.task.optimization.tsp_construct_set.template import template_program, task_description
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
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


def pairwise_distances(coords):
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


def tour_cost(distance_matrix, route):
    total = 0.0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i], route[i + 1]]
    total += distance_matrix[route[-1], route[0]]
    return float(total)


def nearest_neighbor_reference(distance_matrix):
    n = len(distance_matrix)
    route = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda node: distance_matrix[current, node])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour_cost(distance_matrix, route)


def lkh_reference(distance_matrix):
    try:
        import elkai
    except Exception:
        return nearest_neighbor_reference(distance_matrix)
    scale = 1_000_000.0
    integer_matrix = np.rint(np.asarray(distance_matrix, dtype=float) * scale).astype(int)
    integer_matrix = np.maximum(integer_matrix, 0)
    np.fill_diagonal(integer_matrix, 0)
    try:
        route = list(elkai.DistanceMatrix(integer_matrix.tolist()).solve_tsp(runs=10))
    except Exception:
        return nearest_neighbor_reference(distance_matrix)
    if len(route) > 1 and route[0] == route[-1]:
        route = route[:-1]
    if len(route) != len(distance_matrix):
        return nearest_neighbor_reference(distance_matrix)
    return tour_cost(distance_matrix, route)


def gap_score(length, baseline):
    baseline = float(baseline)
    if baseline <= 0 or not math.isfinite(baseline):
        return None
    return (baseline - float(length)) / baseline


def make_tsp_instance(coords):
    distance_matrix = pairwise_distances(coords)
    baseline = lkh_reference(distance_matrix)
    return coords, distance_matrix, baseline


def extract_tsp_coords(instance):
    if isinstance(instance, (tuple, list)) and len(instance) == 3:
        maybe_coords = np.asarray(instance[0])
        maybe_distances = np.asarray(instance[1])
        if maybe_coords.ndim == 2 and maybe_coords.shape[1] == 2 and maybe_distances.ndim == 2:
            instance = instance[0]
    coords = np.asarray(instance, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected TSP coords with shape (n, 2), got {coords.shape}.")
    if len(coords) < 3:
        raise ValueError("Expected at least 3 TSP cities.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("TSP coords must be finite.")
    return coords


def normalize_tsp_instance(instance):
    coords = extract_tsp_coords(instance)
    return make_tsp_instance(coords)


def is_valid_tsp_instance(instance):
    try:
        extract_tsp_coords(instance)
        return True
    except Exception:
        return False


def sample_uniform(rng, n_cities):
    return rng.random((n_cities, 2))


def sample_cluster(rng, n_cities, n_clusters=4, std=0.06):
    centers = rng.uniform(0.15, 0.85, size=(n_clusters, 2))
    assignment = rng.integers(0, n_clusters, size=n_cities)
    coords = centers[assignment] + rng.normal(0.0, std, size=(n_cities, 2))
    return np.clip(coords, 0.0, 1.0)


def sample_bezier_surprise(rng, n_cities):
    control = rng.uniform(0.08, 0.92, size=(4, 2))
    t = np.sort(rng.random(n_cities))[:, None]
    coords = (
        (1 - t) ** 3 * control[0]
        + 3 * (1 - t) ** 2 * t * control[1]
        + 3 * (1 - t) * t**2 * control[2]
        + t**3 * control[3]
    )
    coords += rng.normal(0.0, 0.025, size=(n_cities, 2))
    return np.clip(coords, 0.0, 1.0)


def load_train_dataset(path=None, paths=None):
    if paths is None:
        if path is None:
            raise ValueError("Either path or paths must be provided.")
        paths = [path]
    combined_instances = []
    combined_families = []
    family_order = []
    resolved_paths = []
    for item_path in paths:
        item_path = resolve_repo_path(item_path)
        resolved_paths.append(str(item_path))
        with item_path.open("rb") as handle:
            dataset = pickle.load(handle)
        if isinstance(dataset, dict) and "instances" in dataset:
            instances = list(dataset["instances"])
            instance_families = list(dataset.get("instance_families", []))
            families = list(dataset.get("families", []))
        else:
            instances = list(dataset)
            instance_families = []
            families = []
        if not instance_families and len(families) == 1:
            instance_families = [families[0]] * len(instances)
        combined_instances.extend(instances)
        combined_families.extend(instance_families)
        for family in families:
            if family not in family_order:
                family_order.append(family)
    if combined_families and len(combined_families) != len(combined_instances):
        raise ValueError("Training dataset family metadata length does not match instances.")
    return {
        "path": resolved_paths[0] if len(resolved_paths) == 1 else resolved_paths,
        "instances": combined_instances,
        "instance_families": combined_families,
        "families": family_order,
    }


def build_wake_stream(dataset=None, datasets=None, seed=2026, families=None, batch_size=None, shuffle=False):
    train = load_train_dataset(path=dataset, paths=datasets)
    instances = train["instances"]
    if not instances:
        raise ValueError("Training dataset contains no instances.")
    rng = np.random.default_rng(seed)
    round_size = min(int(batch_size or len(instances)), len(instances))
    while True:
        indices = rng.choice(len(instances), size=round_size, replace=False)
        yield [instances[int(idx)] for idx in indices]


def tsp_descriptor(instance):
    coords = extract_tsp_coords(instance)
    center = np.mean(coords, axis=0)
    spread = np.std(coords, axis=0)
    distances = pairwise_distances(coords)
    upper = distances[np.triu_indices(len(coords), k=1)]
    return np.array(
        [
            center[0],
            center[1],
            spread[0],
            spread[1],
            float(np.mean(upper)),
            float(np.std(upper)),
            float(np.percentile(upper, 10)),
            float(np.percentile(upper, 90)),
        ],
        dtype=float,
    )


def score_summary(score):
    if score is None:
        return None
    values = np.asarray(score if isinstance(score, (list, tuple, np.ndarray)) else [score], dtype=float).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return {
        "avg": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(len(values)),
    }


def function_record(function, rank):
    summary = score_summary(function.score)
    return {
        "rank": rank,
        "name": function.name,
        "score_summary": summary,
        "score": function.score,
        "algorithm": getattr(function, "algorithm", ""),
        "function": str(function),
    }


class OWCAHDRunLogger:
    def __init__(self, root="logs/ow_cahd"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(root) / f"{stamp}_tsp_ow_cahd"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_jsonl = self.log_dir / "history.jsonl"
        self.history_csv = self.log_dir / "history.csv"
        self.summary_json = self.log_dir / "summary.json"
        self._csv_initialized = False
        self._last_token_usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def write_config(self, config, stream_config, hidden_test_config=None):
        payload = {
            "stream": stream_config,
            "hidden_test": hidden_test_config,
            "config": dict(config.__dict__),
        }
        (self.log_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record_round(self, result, llm):
        portfolio = [function_record(function, idx + 1) for idx, function in enumerate(result.portfolio)]
        candidate_pool = [
            function_record(function, idx + 1)
            for idx, function in enumerate(result.candidate_pool)
        ]
        best_avg = None
        valid_summaries = [item["score_summary"] for item in portfolio if item["score_summary"] is not None]
        if valid_summaries:
            best_avg = max(item["avg"] for item in valid_summaries)
        token_usage = llm.token_usage() if hasattr(llm, "token_usage") else {}
        token_delta = {
            key: int(token_usage.get(key, 0)) - int(self._last_token_usage.get(key, 0))
            for key in ("requests", "prompt_tokens", "completion_tokens", "total_tokens")
        }
        self._last_token_usage = dict(token_usage)

        regime_generator_path = None
        regime_generator_paths = []
        if result.accepted_regime is not None:
            regimes_dir = self.log_dir / "regimes"
            regimes_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = regimes_dir / f"{result.accepted_regime}.json"
            generator_programs = list(result.accepted_regime_generator_programs)
            if not generator_programs and result.accepted_regime_generator_program:
                generator_programs = [result.accepted_regime_generator_program]
            for component_id, generator_program in enumerate(generator_programs):
                suffix = "" if len(generator_programs) == 1 else f"_{component_id:02d}"
                generator_path = regimes_dir / f"{result.accepted_regime}_generator{suffix}.py"
                generator_path.write_text(generator_program, encoding="utf-8")
                regime_generator_paths.append(str(generator_path.relative_to(self.log_dir)))
            if regime_generator_paths:
                regime_generator_path = regime_generator_paths[0]
            metadata_path.write_text(
                json.dumps(
                    {
                        "round_id": result.round_id,
                        "name": result.accepted_regime,
                        "description": result.accepted_regime_description,
                        "generator_path": regime_generator_path,
                        "generator_paths": regime_generator_paths,
                        "mixture_components": len(generator_programs),
                        "mixture_weights": result.accepted_regime_mixture_weights,
                        "mixture_n_fit": result.accepted_regime_mixture_n_fit,
                        "mixture_temperatures": result.accepted_regime_mixture_temperatures,
                        "mixture_mus": result.accepted_regime_mixture_mus,
                        "mixture_covs": result.accepted_regime_mixture_covs,
                        "llm_synthesized": bool(generator_programs),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        row = {
            "round_id": result.round_id,
            "novelty_score": float(result.novelty_score),
            "novelty_threshold": float(result.novelty_threshold),
            "novelty_triggered": bool(result.novelty_triggered),
            "accepted_regime": result.accepted_regime,
            "accepted_regime_description": result.accepted_regime_description,
            "regime_generator_path": regime_generator_path,
            "regime_generator_paths": regime_generator_paths,
            "regime_mixture_components": len(regime_generator_paths),
            "belief": result.belief,
            "sleep_instances": int(result.sleep_instances),
            "eohs_sample_budget": int(result.eohs_sample_budget),
            "eohs_samples_used": int(result.eohs_samples_used),
            "eohs_total_samples_used": int(result.eohs_total_samples_used),
            "candidate_pool_size": len(result.candidate_pool),
            "portfolio_size": len(result.portfolio),
            "portfolio_best_avg_score": best_avg,
            "token_usage": token_usage,
            "token_delta": token_delta,
        }
        with self.history_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        csv_row = dict(row)
        csv_row["belief"] = json.dumps(result.belief)
        csv_row["regime_generator_paths"] = json.dumps(regime_generator_paths)
        csv_row["token_usage"] = json.dumps(token_usage)
        csv_row["token_delta"] = json.dumps(token_delta)
        with self.history_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_row.keys()))
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(csv_row)

        portfolio_path = self.log_dir / f"portfolio_round_{result.round_id}.json"
        portfolio_path.write_text(json.dumps(portfolio, indent=2), encoding="utf-8")
        candidate_path = self.log_dir / f"candidate_pool_round_{result.round_id}.json"
        candidate_path.write_text(json.dumps(candidate_pool, indent=2), encoding="utf-8")
        self.summary_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
        (self.log_dir / "token_usage.json").write_text(json.dumps(token_usage, indent=2), encoding="utf-8")


class TSPInMemoryEvaluation(Evaluation):
    def __init__(self, instances, timeout_seconds=120, return_list=True):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )
        self.instances = [normalize_tsp_instance(instance) for instance in instances]
        self.return_list = return_list

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    @staticmethod
    def generate_neighborhood_matrix(coords):
        coords = np.asarray(coords)
        n = len(coords)
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            matrix[i] = np.argsort(np.linalg.norm(coords[i] - coords, axis=1))
        return matrix

    def evaluate(self, heuristic):
        scores = []
        for coords, distance_matrix, baseline in self.instances:
            n = len(coords)
            neighbor_matrix = self.generate_neighborhood_matrix(coords)
            destination_node = 0
            current_node = 0
            route = np.zeros(n, dtype=int)
            for i in range(1, n - 1):
                near_nodes = neighbor_matrix[current_node][1:]
                mask = ~np.isin(near_nodes, route[:i])
                unvisited = near_nodes[mask]
                next_node = heuristic(current_node, destination_node, unvisited, distance_matrix)
                if next_node in route[:i]:
                    return None
                current_node = int(next_node)
                route[i] = current_node

            mask = ~np.isin(np.arange(n), route[: n - 1])
            route[n - 1] = np.arange(n)[mask][0]
            length = tour_cost(distance_matrix, route.tolist())
            score = gap_score(length, baseline)
            if score is None:
                return None
            scores.append(score)
        return scores if self.return_list else float(np.mean(scores))


def main():
    cfg = load_config("ow_cahd.yaml")
    llm_cfg = cfg["llm"]
    method_cfg = cfg["method"]
    stream_config = cfg["stream"]
    hidden_test_config = cfg["hidden_test"]
    wake_stream = build_wake_stream(**stream_config)
    method_cfg.pop("total_rounds", None)

    llm = OpenAIAPI(
        base_url=os.environ.get(llm_cfg["base_url_env"], llm_cfg["base_url_default"]),
        api_key=os.environ[llm_cfg["api_key_env"]],
        model=os.environ.get(llm_cfg["model_env"], llm_cfg["model_default"]),
        timeout=llm_cfg["timeout"],
    )

    config = OWCAHDConfig(**method_cfg)
    logger = OWCAHDRunLogger(root=cfg["logger"]["root"])
    logger.write_config(config, stream_config, hidden_test_config)

    def round_profiler(round_id):
        if not config.print_eohs_samples:
            return None
        print(f"\nOW-CAHD round={round_id} EOHS samples:")
        return EoHSProfiler(
            log_dir=None,
            log_style="simple",
            create_random_path=False,
        )

    method = OWCAHD(
        llm=llm,
        descriptor=tsp_descriptor,
        evaluation_factory=lambda sleep_instances: TSPInMemoryEvaluation(
            [copy.deepcopy(instance) for instance in sleep_instances],
            return_list=True,
        ),
        validity_fn=is_valid_tsp_instance,
        config=config,
        profiler_factory=round_profiler,
    )

    history = []
    for round_id, wake_batch in enumerate(wake_stream):
        result = method.step(wake_batch, round_id=round_id)
        history.append(result)
        logger.record_round(result, llm)
        print(
            f"round={result.round_id} novelty={result.novelty_score:.3f} "
            f"threshold={result.novelty_threshold:.3f} accepted={result.accepted_regime} "
            f"belief={result.belief} samples={result.eohs_total_samples_used}/"
            f"{config.max_sample_nums} tokens={llm.token_usage()} log_dir={logger.log_dir}"
        )
        if result.eohs_total_samples_used >= config.max_sample_nums:
            break
        if result.eohs_samples_used <= 0:
            raise RuntimeError(
                "OW-CAHD made no EOHS sample progress before reaching its sample budget."
            )
    final_portfolio = method.portfolio
    for hidden_dataset_path in hidden_dataset_paths(hidden_test_config):
        output_prefix = hidden_output_prefix(hidden_dataset_path)
        try:
            hidden_dataset = load_hidden_tsp_dataset(hidden_dataset_path)
            portfolios_by_round = {
                int(item["round_id"]): final_portfolio
                for item in hidden_dataset["rounds"]
            }
            utility_by_size, output_path = save_hidden_utility_post_eval(
                logger.log_dir,
                "ow_cahd",
                portfolios_by_round,
                hidden_dataset_path,
                portfolio_protocol="fixed final OW-CAHD portfolio",
                round_workers=1 if hidden_test_config.get("function_timeout_seconds", 0) else 6,
                function_timeout_seconds=hidden_test_config.get("function_timeout_seconds"),
                speed_probe_timeout_seconds=hidden_test_config.get("speed_probe_timeout_seconds"),
                output_prefix="post_eval_hidden_utility",
            )
            print_hidden_utility_post_eval("ow_cahd", utility_by_size, output_path)
        except Exception as exc:
            error_path = logger.log_dir / f"{output_prefix}_error.json"
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
    print(f"OW-CAHD logs written to {logger.log_dir}")


if __name__ == "__main__":
    main()
