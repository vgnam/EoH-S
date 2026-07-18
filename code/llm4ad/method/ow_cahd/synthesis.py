from __future__ import annotations

import ast
import copy
import math
import random
import re
from typing import Callable, Optional

import numpy as np

from ...base import LLM
from .regime import fit_regime_observation_model
from .types import DescriptorFn, Instance, OWCAHDConfig, Regime, ValidityFn


def extract_generate_instance_program(response: str) -> str:
    code = _trim_code_fence(response)
    start = code.find("def generate_instance")
    if start < 0:
        raise ValueError("LLM response does not contain def generate_instance(...).")
    lines = code[start:].splitlines()
    tree = None
    parsed_lines = lines
    while parsed_lines:
        try:
            tree = ast.parse("\n".join(parsed_lines) + "\n")
            break
        except SyntaxError:
            parsed_lines = parsed_lines[:-1]
    if tree is None:
        raise ValueError("Could not parse generate_instance function.")

    target = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "generate_instance":
            target = node
            break
    if target is None or target.end_lineno is None:
        raise ValueError("Could not parse generate_instance function.")
    return "\n".join(parsed_lines[: target.end_lineno]) + "\n"


def synthesize_regime_with_llm(
    llm: LLM,
    instances: list[Instance],
    descriptor: DescriptorFn,
    validity_fn: ValidityFn,
    config: OWCAHDConfig,
    *,
    regime_name: str,
    round_id: int,
) -> Optional[Regime]:
    component_count = max(1, int(config.llm_synthesis_mixture_components))
    fit_sample_count = max(1, int(config.fit_samples_per_regime))
    component_programs: list[str] = []
    component_generators: list[Callable[[int, int], list[Instance]]] = []
    component_samples: list[list[Instance]] = []
    component_mus: list[np.ndarray] = []
    component_covs: list[np.ndarray] = []
    component_n_fit: list[int] = []
    component_temperatures: list[float] = []
    configured_temperatures = [
        float(value) for value in config.llm_synthesis_temperatures
    ] or [1.0]
    min_successful_components = max(
        1,
        min(
            component_count,
            int(config.llm_synthesis_min_successful_components),
        ),
    )

    for component_id in range(component_count):
        temperature = configured_temperatures[component_id % len(configured_temperatures)]
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError(
                f"LLM synthesis temperature must be in [0, 2], got {temperature}."
            )
        last_error = ""
        component_succeeded = False
        for attempt in range(config.llm_synthesis_retries):
            prompt_instances = _sample_prompt_instances(
                instances,
                config.llm_synthesis_examples,
                round_id=round_id,
                component_id=component_id,
                attempt=attempt,
            )
            prompt = build_synthesis_prompt(
                prompt_instances,
                descriptor,
                config,
                round_id,
                attempt,
                last_error,
                component_id=component_id,
                component_count=component_count,
                temperature=temperature,
            )
            response = llm.draw_sample(prompt, temperature=temperature)
            try:
                program = extract_generate_instance_program(response)
                generator_func = compile_generator(program)
                component_seed = _bounded_generator_seed(
                    round_id * 100_000 + component_id * 1_000 + attempt
                )
                samples = list(generator_func(fit_sample_count, component_seed))
                if len(samples) != fit_sample_count:
                    raise ValueError(
                        "generate_instance returned "
                        f"{len(samples)} instances; expected exactly {fit_sample_count}."
                    )
                if not all(validity_fn(instance) for instance in samples):
                    raise ValueError("generate_instance produced invalid instances.")

                component_regime = Regime(
                    name=f"{regime_name}_component_{component_id:02d}",
                    generator=generator_func,
                    archive=list(samples),
                    generator_program=program,
                )
                fit_regime_observation_model(
                    component_regime,
                    descriptor,
                    covariance_reg=config.covariance_reg,
                    samples=list(instances) + samples,
                )
                component_programs.append(program)
                component_generators.append(generator_func)
                component_samples.append(samples)
                component_mus.append(np.asarray(component_regime.mu, dtype=float))
                component_covs.append(np.asarray(component_regime.cov, dtype=float))
                component_n_fit.append(component_regime.n_fit)
                component_temperatures.append(temperature)
                component_succeeded = True
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if config.debug_mode:
                    print(
                        "OWCAHD LLM regime synthesis "
                        f"component {component_id + 1}/{component_count}, "
                        f"attempt {attempt + 1} failed: {last_error}"
                    )
        if not component_succeeded:
            if len(component_programs) < min_successful_components:
                return None
            if config.debug_mode:
                print(
                    "OWCAHD LLM regime synthesis accepted a partial mixture: "
                    f"{len(component_programs)}/{component_count} components."
                )
            break

    if len(component_programs) < min_successful_components:
        return None

    component_count = len(component_programs)
    generator = _build_mixture_generator(component_generators)
    all_generated_samples = [
        instance
        for samples in component_samples
        for instance in samples
    ]
    regime = Regime(
        name=regime_name,
        generator=generator,
        description=(
            f"LLM-synthesized {component_count}-component mixture regime "
            f"at round {round_id}"
        ),
        archive=all_generated_samples,
        generator_program=(component_programs[0] if component_count == 1 else ""),
        generator_programs=component_programs,
        mixture_mus=component_mus,
        mixture_covs=component_covs,
        mixture_weights=np.full(component_count, 1.0 / component_count, dtype=float),
        mixture_n_fit=component_n_fit,
        mixture_temperatures=component_temperatures,
    )
    fit_regime_observation_model(
        regime,
        descriptor,
        covariance_reg=config.covariance_reg,
        samples=list(instances) + all_generated_samples,
    )
    return regime


def _sample_prompt_instances(
    instances: list[Instance],
    count: int,
    *,
    round_id: int,
    component_id: int,
    attempt: int,
) -> list[Instance]:
    """Draw fresh observed examples for every LLM synthesis call."""
    if not instances or count <= 0:
        return []
    sample_size = min(int(count), len(instances))
    seed = _bounded_generator_seed(
        round_id * 100_003 + component_id * 1_009 + attempt * 97 + 31
    )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(instances), size=sample_size, replace=False)
    return [instances[int(index)] for index in indices]


def _build_mixture_generator(
    component_generators: list[Callable[[int, int], list[Instance]]],
) -> Callable[[int, int], list[Instance]]:
    def generator(n: int, seed: int) -> list[Instance]:
        if n <= 0:
            return []
        rng = np.random.default_rng(seed)
        component_count = len(component_generators)
        complete_cycles, remainder = divmod(n, component_count)
        component_ids = np.tile(np.arange(component_count, dtype=int), complete_cycles)
        if remainder:
            extra_ids = rng.choice(component_count, size=remainder, replace=False)
            component_ids = np.concatenate([component_ids, extra_ids])
        rng.shuffle(component_ids)
        generated: list[Optional[Instance]] = [None] * n
        for component_id, component_generator in enumerate(component_generators):
            positions = np.flatnonzero(component_ids == component_id)
            if len(positions) == 0:
                continue
            component_seed = _bounded_generator_seed(
                seed * 1009 + component_id * 9176 + 17
            )
            samples = list(component_generator(len(positions), component_seed))
            if len(samples) != len(positions):
                raise ValueError(
                    f"Mixture component {component_id} returned {len(samples)} instances; "
                    f"expected {len(positions)}."
                )
            for position, instance in zip(positions, samples):
                generated[int(position)] = instance
        if any(instance is None for instance in generated):
            raise ValueError("Mixture generator failed to fill every requested instance slot.")
        return list(generated)

    return generator


def _bounded_generator_seed(seed: int) -> int:
    """Keep seeds small enough for legacy RNGs and common LLM-derived transforms."""
    return int(seed) % 1_000_003


def compile_generator(program: str) -> Callable[[int, int], list[Instance]]:
    namespace = {
        "np": np,
        "math": math,
        "copy": copy,
        "random": random,
        "__builtins__": __builtins__,
    }
    exec(compile(program, "<ow_cahd_llm_generator>", "exec"), namespace)
    generator = namespace.get("generate_instance")
    if not callable(generator):
        raise ValueError("Compiled program does not define callable generate_instance.")
    return generator


def build_synthesis_prompt(
    instances: list[Instance],
    descriptor: DescriptorFn,
    config: OWCAHDConfig,
    round_id: int,
    attempt: int,
    last_error: str,
    *,
    component_id: int = 0,
    component_count: int = 1,
    temperature: float = 1.0,
) -> str:
    examples = []
    for idx, instance in enumerate(instances):
        instance_descriptor = np.asarray(descriptor(instance), dtype=float).ravel().tolist()
        coords_source = instance
        if isinstance(instance, (tuple, list)) and len(instance) in (3, 5):
            candidate = np.asarray(instance[0])
            if candidate.ndim == 2 and candidate.shape[1] == 2:
                coords_source = instance[0]
        coords = np.asarray(coords_source, dtype=float)
        n_cities = int(len(coords))
        coords_observed = np.asarray(coords, dtype=float).round(6).tolist()
        examples.append(
            f"Example {idx}\n"
            f"descriptor = {instance_descriptor}\n"
            f"n_cities = {n_cities}\n"
            f"coords = {coords_observed}\n"
        )

    error_block = ""
    if last_error:
        error_block = (
            "\nThe previous generated program failed with this error. "
            "Repair the code and avoid the same failure:\n"
            f"{last_error}\n"
        )

    context = config.synthesis_context.strip() or (
        "Generate TSP coordinate arrays with shape (n_cities, 2). "
        "The downstream code will compute distance matrices and baselines."
    )
    mixture_block = ""
    if component_count > 1:
        focuses = (
            "global geometry and large-scale support",
            "local density and clustering structure",
            "anisotropy, orientation, and aspect ratios",
            "boundaries, holes, and occupied versus empty regions",
            "multimodality and separated spatial modes",
            "tails, outliers, and rare spatial patterns",
            "spacing, scale variation, and nearest-neighbor structure",
            "curves, manifolds, and topological structure",
            "heterogeneous mixtures of local mechanisms",
            "an alternative plausible stochastic explanation",
        )
        focus = focuses[component_id % len(focuses)]
        mixture_block = (
            f"Mixture component: {component_id + 1}/{component_count}.\n"
            f"Sampling temperature: {temperature:.3f}.\n"
            "The same observations are used for every component, but this prompt must "
            "produce a distinct plausible generator hypothesis. "
            f"Emphasize {focus} while remaining faithful to the observations.\n\n"
        )
    return (
        "You are implementing Module C of OW-CAHD: synthesize an executable "
        "instance generator for a newly observed optimization-instance regime.\n\n"
        f"Round: {round_id}. Attempt: {attempt + 1}.\n\n"
        + mixture_block
        + "Domain/context:\n"
        f"{context}\n\n"
        "Observed novel instances:\n"
        + "\n".join(examples)
        + error_block
        + "\nWrite Python code defining exactly one top-level function:\n\n"
        "def generate_instance(n: int, seed: int) -> list:\n"
        "    ...\n\n"
        "Requirements:\n"
        "- Return a list with exactly n valid instances.\n"
        "- Use only Python standard library, math, copy, random, and numpy as np.\n"
        "- The function must be deterministic for a fixed seed.\n"
        "- Normalize every seed passed to numpy RandomState or np.random.seed with "
        "int(derived_seed) % (2**32 - 1).\n"
        "- Return each instance as coordinates only: a numpy array or nested list "
        "with shape (n_cities, 2). Do not compute or return distance_matrix or baseline.\n"
        "- Do not call external solvers, files, network, or plotting APIs.\n"
        "- Include any helper logic inside generate_instance, not as separate top-level functions.\n"
        "- Output only a Python code block or raw Python code.\n"
    )


def _trim_code_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()
