"""Shared runners and data adapters for EoH-S/OW-CAHD ACO experiments."""

from __future__ import annotations

import copy
import concurrent.futures
import csv
import json
import os
import pickle
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "examples" / "training" / "cvrp_set"))

from cvrp_common import normalize_cvrp_instance  # noqa: E402
from llm4ad.base import (  # noqa: E402
    Function,
    SecureEvaluator,
    TextFunctionProgramConverter,
)
from llm4ad.method.eohs import EoHS, EoHSProfiler  # noqa: E402
from llm4ad.method.ow_cahd import OWCAHD, OWCAHDConfig  # noqa: E402
from llm4ad.task.optimization.aco_pheromone_set import (  # noqa: E402
    CVRPACOPheromoneEvaluation,
    TSPACOPheromoneEvaluation,
)
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI  # noqa: E402


PROBLEMS = ("tsp", "cvrp")


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(name: str) -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / "cfg" / name).read_text(encoding="utf-8"))


def make_llm(config: dict[str, Any]) -> OpenAIAPI:
    return OpenAIAPI(
        base_url=os.environ.get(config["base_url_env"], config["base_url_default"]),
        api_key=os.environ[config["api_key_env"]],
        model=os.environ.get(config["model_env"], config["model_default"]),
        timeout=config["timeout"],
    )


def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :],
        axis=2,
    )


def _tour_cost(distances: np.ndarray, tour: list[int]) -> float:
    indices = np.asarray(tour, dtype=int)
    return float(np.sum(distances[indices, np.roll(indices, -1)]))


def _tsp_reference(distances: np.ndarray) -> float:
    """Deterministic nearest-neighbour plus 2-opt reference for synthetic replay."""

    n = len(distances)
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        current = tour[-1]
        next_node = min(unvisited, key=lambda node: distances[current, node])
        tour.append(next_node)
        unvisited.remove(next_node)
    improved = True
    while improved:
        improved = False
        for start in range(1, n - 1):
            for end in range(start + 1, n):
                before = tour[start - 1]
                first = tour[start]
                last = tour[end]
                after = tour[(end + 1) % n]
                delta = (
                    distances[before, last]
                    + distances[first, after]
                    - distances[before, first]
                    - distances[last, after]
                )
                if delta < -1e-12:
                    tour[start : end + 1] = reversed(tour[start : end + 1])
                    improved = True
                    break
            if improved:
                break
    return _tour_cost(distances, tour)


def _extract_coordinates(instance: Any, problem: str) -> np.ndarray:
    expected_length = 3 if problem == "tsp" else 5
    if isinstance(instance, (tuple, list)) and len(instance) == expected_length:
        possible_coordinates = np.asarray(instance[0])
        candidate = (
            instance[0]
            if possible_coordinates.ndim == 2
            and possible_coordinates.shape[1:] == (2,)
            else instance
        )
    else:
        candidate = instance
    coordinates = np.asarray(candidate, dtype=float)
    minimum = 3 if problem == "tsp" else 2
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < minimum:
        raise ValueError(f"{problem.upper()} coordinates must have shape (n>={minimum}, 2)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{problem.upper()} coordinates must be finite")
    return coordinates


def normalize_tsp_instance(instance: Any) -> tuple[np.ndarray, np.ndarray, float]:
    if isinstance(instance, (tuple, list)) and len(instance) == 3:
        try:
            coordinates = np.asarray(instance[0], dtype=float)
            distances = np.asarray(instance[1], dtype=float)
            baseline = float(instance[2])
            if (
                coordinates.ndim == 2
                and coordinates.shape[1:] == (2,)
                and distances.shape == (len(coordinates), len(coordinates))
                and baseline > 0
            ):
                return coordinates, distances, baseline
        except (TypeError, ValueError):
            pass
    coordinates = np.clip(_extract_coordinates(instance, "tsp"), 0.0, 1.0)
    distances = pairwise_distances(coordinates)
    return coordinates, distances, _tsp_reference(distances)


def normalize_instance(instance: Any, problem: str) -> Any:
    if problem == "tsp":
        return normalize_tsp_instance(instance)
    if problem == "cvrp":
        try:
            return normalize_cvrp_instance(instance)
        except (TypeError, ValueError):
            return normalize_cvrp_instance(_extract_coordinates(instance, problem))
    raise ValueError(f"Unsupported problem: {problem}")


def is_valid_instance(instance: Any, problem: str) -> bool:
    try:
        _extract_coordinates(instance, problem)
        return True
    except Exception:
        return False


def instance_descriptor(instance: Any, problem: str) -> np.ndarray:
    if problem == "tsp":
        coordinates, distances, _baseline = normalize_tsp_instance(instance)
        points = coordinates
        demand_features: list[float] = []
    else:
        coordinates, distances, demands, capacity, _baseline = normalize_instance(
            instance, "cvrp"
        )
        points = coordinates[1:]
        customer_demands = demands[1:].astype(float)
        demand_features = [
            float(np.mean(customer_demands)),
            float(np.std(customer_demands)),
            float(np.sum(customer_demands) / capacity),
        ]
    center = np.mean(points, axis=0)
    spread = np.std(points, axis=0)
    relevant = distances if problem == "tsp" else distances[1:, 1:]
    upper = relevant[np.triu_indices(len(relevant), k=1)]
    return np.asarray(
        [
            center[0],
            center[1],
            spread[0],
            spread[1],
            float(np.mean(upper)),
            float(np.std(upper)),
            float(np.percentile(upper, 10)),
            float(np.percentile(upper, 90)),
            *demand_features,
        ],
        dtype=float,
    )


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_train_instances(paths: Iterable[str | Path], problem: str) -> list[Any]:
    instances: list[Any] = []
    for path in paths:
        dataset = _load_pickle(resolve_repo_path(path))
        loaded = dataset.get("instances", dataset) if isinstance(dataset, dict) else dataset
        if isinstance(loaded, dict):
            loaded = loaded.values()
        instances.extend(normalize_instance(instance, problem) for instance in loaded)
    if not instances:
        raise ValueError(f"No {problem.upper()} ACO training instances were loaded")
    return instances


def build_wake_stream(
    paths: Iterable[str | Path],
    problem: str,
    seed: int,
    batch_size: int,
    shuffle: bool = True,
):
    instances = load_train_instances(paths, problem)
    batch_size = min(int(batch_size), len(instances))
    rng = np.random.default_rng(seed)
    while True:
        if shuffle:
            indices = rng.choice(len(instances), size=batch_size, replace=False)
        else:
            offset = int(rng.integers(len(instances)))
            indices = (np.arange(batch_size) + offset) % len(instances)
        yield [copy.deepcopy(instances[int(index)]) for index in indices]


def load_hidden_instances(path: str | Path, problem: str) -> list[Any]:
    dataset = _load_pickle(resolve_repo_path(path))
    if not isinstance(dataset, dict) or "rounds" not in dataset:
        raise ValueError(f"Unsupported hidden {problem.upper()} dataset: {path}")
    instances: list[Any] = []
    for hidden_round in dataset["rounds"]:
        if problem == "tsp":
            for coordinates, baseline in zip(
                hidden_round["coordinates"], hidden_round["baselines"]
            ):
                coordinates = np.asarray(coordinates, dtype=float)
                instances.append(
                    (coordinates, pairwise_distances(coordinates), float(baseline))
                )
        else:
            fields = zip(
                hidden_round["coordinates"],
                hidden_round["demands"],
                hidden_round["capacities"],
                hidden_round["baselines"],
            )
            for coordinates, demands, capacity, baseline in fields:
                coordinates = np.asarray(coordinates, dtype=float)
                instances.append(
                    (
                        coordinates,
                        pairwise_distances(coordinates),
                        np.asarray(demands, dtype=int),
                        int(capacity),
                        float(baseline),
                    )
                )
    return instances


def make_evaluation(
    problem: str,
    instances: Iterable[Any],
    aco_config: dict[str, Any],
    *,
    safe_evaluate: bool = True,
):
    evaluator_class = (
        TSPACOPheromoneEvaluation if problem == "tsp" else CVRPACOPheromoneEvaluation
    )
    return evaluator_class(
        instances=[normalize_instance(instance, problem) for instance in instances],
        safe_evaluate=safe_evaluate,
        return_list=True,
        **aco_config,
    )


def function_record(function: Function, rank: int) -> dict[str, Any]:
    score = function.score
    if score is None:
        values = np.asarray([], dtype=float)
    else:
        values = np.asarray(
            score if isinstance(score, (list, tuple, np.ndarray)) else [score],
            dtype=float,
        )
        values = values[np.isfinite(values)]
    return {
        "rank": rank,
        "name": function.name,
        "mean_score": float(np.mean(values)) if len(values) else None,
        "score": score,
        "algorithm": getattr(function, "algorithm", ""),
        "function": str(function),
    }


class ACORunLogger:
    def __init__(self, root: str | Path, problem: str, method: str):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = resolve_repo_path(root) / f"{stamp}_{problem}_aco_{method}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_jsonl = self.log_dir / "history.jsonl"
        self.history_csv = self.log_dir / "history.csv"
        self._csv_initialized = False

    def write_config(self, config: dict[str, Any]) -> None:
        (self.log_dir / "run_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    def record_round(self, result, llm) -> None:
        portfolio = [
            function_record(function, rank + 1)
            for rank, function in enumerate(result.portfolio)
        ]
        candidates = [
            function_record(function, rank + 1)
            for rank, function in enumerate(result.candidate_pool)
        ]
        generator_paths = []
        generator_programs = list(result.accepted_regime_generator_programs)
        if not generator_programs and result.accepted_regime_generator_program:
            generator_programs = [result.accepted_regime_generator_program]
        if result.accepted_regime and generator_programs:
            regimes_dir = self.log_dir / "regimes"
            regimes_dir.mkdir(parents=True, exist_ok=True)
            for component_id, program in enumerate(generator_programs):
                suffix = "" if len(generator_programs) == 1 else f"_{component_id:02d}"
                path = regimes_dir / f"{result.accepted_regime}_generator{suffix}.py"
                path.write_text(program, encoding="utf-8")
                generator_paths.append(str(path.relative_to(self.log_dir)))
            metadata = {
                "round_id": result.round_id,
                "name": result.accepted_regime,
                "description": result.accepted_regime_description,
                "generator_paths": generator_paths,
                "mixture_weights": result.accepted_regime_mixture_weights,
                "mixture_n_fit": result.accepted_regime_mixture_n_fit,
                "mixture_temperatures": result.accepted_regime_mixture_temperatures,
            }
            (regimes_dir / f"{result.accepted_regime}.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
        row = {
            "round_id": result.round_id,
            "novelty_score": float(result.novelty_score),
            "novelty_threshold": float(result.novelty_threshold),
            "accepted_regime": result.accepted_regime,
            "regime_generator_paths": generator_paths,
            "belief": result.belief,
            "sleep_instances": result.sleep_instances,
            "eohs_samples_used": result.eohs_samples_used,
            "eohs_total_samples_used": result.eohs_total_samples_used,
            "portfolio_size": len(portfolio),
            "candidate_pool_size": len(candidates),
            "token_usage": llm.token_usage(),
        }
        with self.history_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        csv_row = dict(row)
        for field in ("belief", "regime_generator_paths", "token_usage"):
            csv_row[field] = json.dumps(csv_row[field])
        with self.history_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_row))
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(csv_row)
        (self.log_dir / f"portfolio_round_{result.round_id}.json").write_text(
            json.dumps(portfolio, indent=2), encoding="utf-8"
        )
        (self.log_dir / f"candidate_pool_round_{result.round_id}.json").write_text(
            json.dumps(candidates, indent=2), encoding="utf-8"
        )
        (self.log_dir / "token_usage.json").write_text(
            json.dumps(llm.token_usage(), indent=2), encoding="utf-8"
        )


def evaluate_portfolio(
    problem: str,
    portfolio: list[Function],
    instances: list[Any],
    aco_config: dict[str, Any],
    workers: int = 1,
) -> tuple[list[float], list[dict[str, Any]]]:
    if not portfolio:
        raise ValueError("Cannot post-evaluate an empty portfolio")
    if workers < 1:
        raise ValueError("workers must be positive")

    def evaluate_function(item: tuple[int, Function]):
        rank, function = item
        evaluation = make_evaluation(problem, instances, aco_config, safe_evaluate=True)
        template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        program = TextFunctionProgramConverter.function_to_program(function, template)
        scores = SecureEvaluator(evaluation).evaluate_program(program)
        if scores is None:
            return rank, None, {"rank": rank, "valid": False, "mean_utility": None}
        vector = np.asarray(scores, dtype=float)
        if vector.shape != (len(instances),) or not np.all(np.isfinite(vector)):
            return rank, None, {"rank": rank, "valid": False, "mean_utility": None}
        return rank, vector, {
            "rank": rank,
            "valid": True,
            "mean_utility": float(np.mean(vector)),
        }

    indexed_portfolio = list(enumerate(portfolio, start=1))
    if workers == 1:
        evaluated = [evaluate_function(item) for item in indexed_portfolio]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(indexed_portfolio))
        ) as executor:
            evaluated = list(executor.map(evaluate_function, indexed_portfolio))
    evaluated.sort(key=lambda item: item[0])
    function_scores = [vector for _rank, vector, _record in evaluated if vector is not None]
    functions = [record for _rank, _vector, record in evaluated]
    if not function_scores:
        raise RuntimeError("Every portfolio function failed ACO post-evaluation")
    portfolio_utility = np.max(np.vstack(function_scores), axis=0)
    return portfolio_utility.tolist(), functions


def post_evaluate_hidden(
    problem: str,
    method: str,
    portfolio: list[Function],
    hidden_config: dict[str, Any],
    log_dir: str | Path,
) -> list[dict[str, Any]]:
    rows = []
    for dataset_path in hidden_config["datasets"]:
        dataset_name = Path(dataset_path).stem
        try:
            instances = load_hidden_instances(dataset_path, problem)
            utility, functions = evaluate_portfolio(
                problem,
                portfolio,
                instances,
                dict(hidden_config["aco"]),
                workers=int(hidden_config.get("function_workers", 1)),
            )
            size = len(instances[0][0]) - (1 if problem == "cvrp" else 0)
            split = "ood" if "_ood_" in dataset_name else "id"
            payload = {
                "format": "eohs-aco-hidden-post-eval-v1",
                "problem": problem,
                "method": method,
                "dataset": str(dataset_path),
                "split": split,
                "size": size,
                "instances": len(instances),
                "portfolio_protocol": "per-instance best utility over fixed final portfolio",
                "mean_utility": float(np.mean(utility)),
                "std_utility": float(np.std(utility)),
                "utility": utility,
                "functions": functions,
                "aco": hidden_config["aco"],
            }
            output_path = Path(log_dir) / f"post_eval_{dataset_name}.json"
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            rows.append(payload)
            print(
                f"{problem.upper()}-ACO {method} {split.upper()} size={size}: "
                f"utility={payload['mean_utility']:+.6f} ({len(instances)} instances)"
            )
        except Exception as error:
            error_payload = {
                "problem": problem,
                "method": method,
                "dataset": str(dataset_path),
                "error": f"{type(error).__name__}: {error}",
            }
            (Path(log_dir) / f"post_eval_{dataset_name}_error.json").write_text(
                json.dumps(error_payload, indent=2), encoding="utf-8"
            )
            print(
                f"Post-eval failed for {dataset_path}: "
                f"{type(error).__name__}: {error}"
            )
    summary_path = Path(log_dir) / "post_eval_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _save_eohs_metadata(
    log_dir: Path,
    config: dict[str, Any],
    llm,
    population: list[Function],
) -> None:
    (log_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (log_dir / "token_usage.json").write_text(
        json.dumps(llm.token_usage(), indent=2), encoding="utf-8"
    )
    (log_dir / "final_population.json").write_text(
        json.dumps(
            [
                function_record(function, rank + 1)
                for rank, function in enumerate(population)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def run_eohs(problem: str, config_name: str) -> None:
    config = load_config(config_name)
    train_instances = load_train_instances(config["task"]["datasets"], problem)
    aco_config = dict(config["task"]["aco"])
    max_instances = config["task"].get("max_instances")
    if max_instances:
        train_instances = train_instances[: int(max_instances)]
    evaluation = make_evaluation(problem, train_instances, aco_config)
    llm = make_llm(config["llm"])
    profiler = EoHSProfiler(**config["profiler"])
    method = EoHS(
        llm=llm,
        profiler=profiler,
        evaluation=evaluation,
        **config["method"],
    )
    method.run()
    log_dir = Path(profiler._log_dir)
    final_population = method._population.population
    _save_eohs_metadata(log_dir, config, llm, final_population)
    post_evaluate_hidden(
        problem,
        "eohs",
        final_population,
        config["hidden_test"],
        log_dir,
    )
    print(f"{problem.upper()}-ACO EoH-S logs written to {log_dir}")


def run_ow_cahd(problem: str, config_name: str) -> None:
    config = load_config(config_name)
    stream_config = config["stream"]
    wake_stream = build_wake_stream(
        stream_config["datasets"],
        problem,
        seed=stream_config["seed"],
        batch_size=stream_config["batch_size"],
        shuffle=stream_config.get("shuffle", True),
    )
    llm = make_llm(config["llm"])
    method_config = OWCAHDConfig(**config["method"])
    logger = ACORunLogger(config["logger"]["root"], problem, "ow_cahd")
    logger.write_config(config)

    def profiler_factory(round_id: int):
        if not method_config.print_eohs_samples:
            return None
        print(f"\n{problem.upper()}-ACO OW-CAHD round={round_id} EoH-S samples:")
        return EoHSProfiler(log_dir=None, log_style="simple", create_random_path=False)

    train_aco_config = dict(config["aco_train"])
    method = OWCAHD(
        llm=llm,
        descriptor=lambda instance: instance_descriptor(instance, problem),
        evaluation_factory=lambda replay: make_evaluation(
            problem,
            [normalize_instance(item, problem) for item in replay],
            train_aco_config,
        ),
        validity_fn=lambda instance: is_valid_instance(instance, problem),
        config=method_config,
        profiler_factory=profiler_factory,
    )
    for round_id, wake_batch in enumerate(wake_stream):
        result = method.step(wake_batch, round_id=round_id)
        logger.record_round(result, llm)
        print(
            f"round={round_id} accepted={result.accepted_regime} "
            f"samples={result.eohs_total_samples_used}/{method_config.max_sample_nums} "
            f"tokens={llm.token_usage()}"
        )
        if result.eohs_total_samples_used >= method_config.max_sample_nums:
            break
        if result.eohs_samples_used <= 0:
            raise RuntimeError("OW-CAHD made no EoH-S sample progress")
    (logger.log_dir / "final_portfolio.json").write_text(
        json.dumps(
            [
                function_record(function, rank + 1)
                for rank, function in enumerate(method.portfolio)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    post_evaluate_hidden(
        problem,
        "ow_cahd",
        method.portfolio,
        config["hidden_test"],
        logger.log_dir,
    )
    print(f"{problem.upper()}-ACO OW-CAHD logs written to {logger.log_dir}")
