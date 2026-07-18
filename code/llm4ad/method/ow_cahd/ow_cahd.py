from __future__ import annotations

import copy
import math
from typing import Callable, Iterable, Optional

import numpy as np

from ..eohs import EoHS, EoHSProfiler
from ...base import Evaluation, Function, LLM, SecureEvaluator, TextFunctionProgramConverter
from ...tools.profiler import ProfilerBase
from .portfolio import greedy_portfolio, score_vector
from .regime import bootstrap_regime_from_instances, fit_regime_observation_model
from .stats import (
    gaussian_logpdf,
    mahalanobis_sq,
    mean_descriptor,
    mmd_pvalue,
    normalize,
    novelty_threshold,
)
from .synthesis import compile_generator, extract_generate_instance_program, synthesize_regime_with_llm
from .types import (
    DescriptorFn,
    EvaluationFactory,
    Instance,
    OWCAHDConfig,
    OWCAHDRoundResult,
    Regime,
    RegimeSynthesizer,
    ValidityFn,
)


class OWCAHD:
    """Open-world continual AHSD controller using EoH-S as the inner backbone."""

    def __init__(
        self,
        llm: LLM,
        descriptor: DescriptorFn,
        evaluation_factory: EvaluationFactory,
        *,
        config: Optional[OWCAHDConfig] = None,
        initial_regimes: Optional[list[Regime]] = None,
        regime_synthesizer: Optional[RegimeSynthesizer] = None,
        validity_fn: Optional[ValidityFn] = None,
        profiler_factory: Optional[Callable[[int], Optional[ProfilerBase]]] = None,
    ):
        self.llm = llm
        self.descriptor = descriptor
        self.evaluation_factory = evaluation_factory
        self.config = config or OWCAHDConfig()
        self.regime_synthesizer = regime_synthesizer
        self.validity_fn = validity_fn or (lambda _: True)
        self.profiler_factory = profiler_factory
        self.regimes: list[Regime] = list(initial_regimes or [])
        self.belief: np.ndarray = np.ones(len(self.regimes), dtype=float)
        if len(self.belief):
            self.belief = normalize(self.belief)
        self._novel_streak = 0
        self._novel_buffer: list[Instance] = []
        self.portfolio: list[Function] = []
        # Keep the complete final EoHS population so evolution can continue
        # across rounds.  The portfolio is only the smaller deployment subset.
        self.population: list[Function] = []
        self.history: list[OWCAHDRoundResult] = []
        self._eohs_sample_budget_used = 0
        self._last_eohs_sample_budget = 0
        self._last_eohs_samples_used = 0

    def run(self, wake_stream: Iterable[list[Instance]]) -> list[OWCAHDRoundResult]:
        for round_id, wake_batch in enumerate(wake_stream):
            self.step(list(wake_batch), round_id=round_id)
        return self.history

    def step(self, wake_batch: list[Instance], *, round_id: Optional[int] = None) -> OWCAHDRoundResult:
        round_id = len(self.history) if round_id is None else round_id
        phi = mean_descriptor(wake_batch, self.descriptor)
        threshold = novelty_threshold(len(phi), self.config.novelty_alpha)
        if self.regimes:
            novelty_score = min(mahalanobis_sq(phi, regime) for regime in self.regimes)
            novelty_triggered = novelty_score > threshold
        else:
            novelty_score = 0.0
            novelty_triggered = False

        if self.config.always_synthesize_regime:
            new_regime = self._synthesize_regime(round_id, instances=wake_batch)
            if new_regime is None:
                raise RuntimeError(
                    f"LLM failed to synthesize regime z{len(self.regimes)} at round {round_id} "
                    f"after {self.config.llm_synthesis_retries} attempts."
                )
            self._accept_regime(new_regime)
            self._novel_streak = 0
            self._novel_buffer = []
            accepted_regime = new_regime.name
        elif not self.regimes:
            self._add_bootstrap_regime("z0", wake_batch, "initial bootstrap regime")
            accepted_regime = "z0" if self.config.auto_accept_regime else None
        else:
            accepted_regime = None

        if not self.config.always_synthesize_regime:
            if self.config.auto_accept_regime and accepted_regime is None:
                accepted_regime = self._accept_current_batch_as_regime(wake_batch)
            elif not self.config.auto_accept_regime:
                accepted_regime = self._maybe_accept_novel_regime(wake_batch, novelty_triggered, round_id)

        self._bayes_filter(phi)
        sleep_instances, sleep_weights = self._build_sleep_set(round_id)
        evaluation = self.evaluation_factory(sleep_instances)
        rescored_population = self._score_functions(self.population, evaluation)
        valid_seed_functions = [
            func for func in rescored_population
            if (
                (vector := score_vector(func, len(sleep_instances))) is not None
                and np.all(np.isfinite(vector))
            )
        ]
        initial_population = valid_seed_functions if len(valid_seed_functions) >= 2 else None
        candidates = self._run_eohs_backbone(
            evaluation,
            round_id,
            initial_population=initial_population,
        )
        # A fully failed inner run must not erase the inherited population.
        # Keep the last population so a transient LLM/evaluation failure in
        # one round does not reset continual evolution in subsequent rounds.
        if candidates:
            self.population = [copy.deepcopy(func) for func in candidates]

        # Preserve any previously deployed function that was not part of the
        # EoHS population (mainly relevant for older/restored controller state).
        rescored_portfolio = self._score_functions(self.portfolio, evaluation)
        candidate_programs = {str(func) for func in candidates}
        candidates.extend(
            func for func in rescored_portfolio
            if str(func) not in candidate_programs
        )
        self.portfolio = greedy_portfolio(candidates, sleep_weights, self.config.portfolio_size)

        accepted_regime_object = next(
            (regime for regime in self.regimes if regime.name == accepted_regime),
            None,
        )

        result = OWCAHDRoundResult(
            round_id=round_id,
            novelty_score=novelty_score,
            novelty_threshold=threshold,
            novelty_triggered=novelty_triggered,
            accepted_regime=accepted_regime,
            belief={regime.name: float(weight) for regime, weight in zip(self.regimes, self.belief)},
            sleep_instances=len(sleep_instances),
            portfolio=list(self.portfolio),
            candidate_pool=[copy.deepcopy(func) for func in candidates],
            eohs_sample_budget=self._last_eohs_sample_budget,
            eohs_samples_used=self._last_eohs_samples_used,
            eohs_total_samples_used=self._eohs_sample_budget_used,
            accepted_regime_description=(
                accepted_regime_object.description
                if accepted_regime_object is not None
                else None
            ),
            accepted_regime_generator_program=(
                accepted_regime_object.generator_program
                if accepted_regime_object is not None
                and accepted_regime_object.generator_program
                else None
            ),
            accepted_regime_generator_programs=(
                list(accepted_regime_object.generator_programs)
                if accepted_regime_object is not None
                and accepted_regime_object.generator_programs
                else []
            ),
            accepted_regime_mixture_mus=(
                [mu.tolist() for mu in accepted_regime_object.mixture_mus]
                if accepted_regime_object is not None
                else []
            ),
            accepted_regime_mixture_covs=(
                [cov.tolist() for cov in accepted_regime_object.mixture_covs]
                if accepted_regime_object is not None
                else []
            ),
            accepted_regime_mixture_weights=(
                accepted_regime_object.mixture_weights.tolist()
                if accepted_regime_object is not None
                and accepted_regime_object.mixture_weights is not None
                else []
            ),
            accepted_regime_mixture_n_fit=(
                list(accepted_regime_object.mixture_n_fit)
                if accepted_regime_object is not None
                else []
            ),
            accepted_regime_mixture_temperatures=(
                list(accepted_regime_object.mixture_temperatures)
                if accepted_regime_object is not None
                else []
            ),
        )
        self.history.append(result)
        return result

    def _maybe_accept_novel_regime(
        self,
        wake_batch: list[Instance],
        novelty_triggered: bool,
        round_id: int,
    ) -> Optional[str]:
        if novelty_triggered:
            self._novel_streak += 1
            self._novel_buffer.extend(wake_batch)
            self._novel_buffer = self._novel_buffer[-self.config.novelty_buffer_size :]
        else:
            self._novel_streak = 0
            self._novel_buffer = []

        if self._novel_streak < self.config.novelty_confirm_rounds:
            return None

        new_regime = self._synthesize_regime(round_id)
        if new_regime is None and self.config.auto_accept_regime:
            new_regime = bootstrap_regime_from_instances(
                f"z{len(self.regimes)}",
                list(self._novel_buffer),
                self.descriptor,
                description="auto-accepted bootstrap regime after LLM synthesis failed",
                covariance_reg=self.config.covariance_reg,
            )

        if new_regime is None:
            return None

        if not self.config.auto_accept_regime and not self._verify_regime(new_regime, self._novel_buffer, round_id):
            return None

        self._accept_regime(new_regime)
        self._novel_streak = 0
        self._novel_buffer = []
        return new_regime.name

    def _accept_current_batch_as_regime(self, wake_batch: list[Instance]) -> str:
        regime = bootstrap_regime_from_instances(
            f"z{len(self.regimes)}",
            wake_batch,
            self.descriptor,
            description="auto-accepted regime from current wake batch",
            covariance_reg=self.config.covariance_reg,
        )
        self._accept_regime(regime)
        self._novel_streak = 0
        self._novel_buffer = []
        return regime.name

    def _add_bootstrap_regime(self, name: str, instances: list[Instance], description: str) -> None:
        regime = bootstrap_regime_from_instances(
            name,
            instances,
            self.descriptor,
            description=description,
            covariance_reg=self.config.covariance_reg,
        )
        self.regimes.append(regime)
        self.belief = normalize(np.ones(len(self.regimes), dtype=float))

    def _synthesize_regime(
        self,
        round_id: int,
        *,
        instances: Optional[list[Instance]] = None,
    ) -> Optional[Regime]:
        buffer = list(self._novel_buffer if instances is None else instances)
        if self.regime_synthesizer is not None:
            regime = self.regime_synthesizer(buffer, self.descriptor, round_id)
        else:
            regime = synthesize_regime_with_llm(
                self.llm,
                buffer,
                self.descriptor,
                self.validity_fn,
                self.config,
                regime_name=f"z{len(self.regimes)}",
                round_id=round_id,
            )
            if regime is None and self.config.allow_bootstrap_fallback:
                regime = bootstrap_regime_from_instances(
                    f"z{len(self.regimes)}",
                    buffer,
                    self.descriptor,
                    description="LLM synthesis failed; bootstrap fallback regime",
                    covariance_reg=self.config.covariance_reg,
                )
        if regime is None:
            return None
        if regime.mu is None or regime.cov is None:
            samples = regime.archive or regime.sample(self.config.fit_samples_per_regime, seed=round_id)
            fit_regime_observation_model(
                regime,
                self.descriptor,
                covariance_reg=self.config.covariance_reg,
                samples=buffer + list(samples),
            )
        return regime

    def _verify_regime(self, regime: Regime, real_instances: list[Instance], round_id: int) -> bool:
        generated = regime.sample(min(self.config.mmd_samples, max(1, len(real_instances))), seed=round_id)
        if not generated or not all(self.validity_fn(instance) for instance in generated):
            return False
        if self.config.skip_mmd_verification:
            return True
        real_phi = np.vstack([np.asarray(self.descriptor(instance), dtype=float).ravel() for instance in real_instances])
        gen_phi = np.vstack([np.asarray(self.descriptor(instance), dtype=float).ravel() for instance in generated])
        _, pvalue = mmd_pvalue(real_phi, gen_phi, self.config.mmd_permutations, seed=round_id)
        return pvalue >= self.config.mmd_pvalue_threshold

    def _accept_regime(self, regime: Regime) -> None:
        self.regimes.append(regime)
        old = self.belief * (1.0 - self.config.new_regime_blend)
        self.belief = np.concatenate([old, np.array([self.config.new_regime_blend], dtype=float)])
        self.belief = self._normalize_belief(self.belief)

    def _normalize_belief(self, values: np.ndarray) -> np.ndarray:
        probabilities = normalize(values)
        k = len(probabilities)
        if k == 0:
            return probabilities
        floor = min(max(float(self.config.belief_floor), 0.0), 1.0 / k)
        if floor == 0.0:
            return probabilities
        return floor + (1.0 - floor * k) * probabilities

    def _bayes_filter(self, phi: np.ndarray) -> None:
        k = len(self.regimes)
        if k == 0:
            return
        sticky = self.config.sticky_transition
        if k == 1:
            transition = np.ones((1, 1), dtype=float)
        else:
            off_diag = (1.0 - sticky) / (k - 1)
            transition = np.full((k, k), off_diag, dtype=float)
            np.fill_diagonal(transition, sticky)
        predicted = transition.T @ self.belief
        log_likelihood = np.array([gaussian_logpdf(phi, regime) for regime in self.regimes], dtype=float)
        log_likelihood -= np.max(log_likelihood)
        self.belief = self._normalize_belief(predicted * np.exp(log_likelihood))

    def _posterior_for_descriptor(self, phi: np.ndarray) -> np.ndarray:
        log_likelihood = np.array([gaussian_logpdf(phi, regime) for regime in self.regimes], dtype=float)
        log_likelihood -= np.max(log_likelihood)
        return self._normalize_belief(self.belief * np.exp(log_likelihood))

    def _allocation_weights(self, round_id: int) -> np.ndarray:
        if len(self.regimes) == 1:
            return np.ones(1, dtype=float)
        gains = []
        for idx, regime in enumerate(self.regimes):
            samples = self._valid_regime_samples(
                regime,
                8,
                seed=round_id * 1009 + idx,
            )
            kl_values = []
            for instance in samples:
                try:
                    posterior = self._posterior_for_descriptor(
                        np.asarray(self.descriptor(instance), dtype=float).ravel()
                    )
                except Exception:
                    continue
                kl_values.append(
                    float(np.sum(posterior * np.log((posterior + 1e-12) / (self.belief + 1e-12))))
                )
            gains.append(float(np.mean(kl_values)) if kl_values else 0.0)
        scaled = np.exp(np.asarray(gains) / max(self.config.allocation_temperature, 1e-12))
        return normalize(scaled)

    def _build_sleep_set(self, round_id: int) -> tuple[list[Instance], np.ndarray]:
        weights = self._allocation_weights(round_id)
        total = max(self.config.sleep_instances_per_round, len(self.regimes) * self.config.min_sleep_per_regime)
        raw_counts = self._sleep_counts(weights, total)

        instances: list[Instance] = []
        instance_weights: list[float] = []
        for idx, (regime, count) in enumerate(zip(self.regimes, raw_counts)):
            samples = self._valid_regime_samples(
                regime,
                int(count),
                seed=round_id * 7919 + idx,
            )
            instances.extend(samples)
            instance_weights.extend([float(self.belief[idx]) / max(1, len(samples))] * len(samples))
        if not instances:
            raise RuntimeError("OW-CAHD could not sample any valid SLEEP instances from known regimes.")
        return instances, normalize(np.asarray(instance_weights, dtype=float))

    def _valid_regime_samples(self, regime: Regime, n: int, *, seed: int) -> list[Instance]:
        if n <= 0:
            return []
        valid: list[Instance] = []
        attempts = 0
        batch_size = n
        while len(valid) < n and attempts < 3:
            try:
                samples = regime.sample(batch_size, seed=seed + attempts * 104729)
            except Exception:
                samples = []
            for instance in samples:
                if self.validity_fn(instance):
                    valid.append(instance)
                    if len(valid) == n:
                        break
            attempts += 1
            batch_size = max(n, (n - len(valid)) * 2)

        if len(valid) < n:
            for instance in regime.archive:
                if self.validity_fn(instance):
                    valid.append(instance)
                    if len(valid) == n:
                        break
        return valid

    def _sleep_counts(self, weights: np.ndarray, total: int) -> np.ndarray:
        raw_counts = np.floor(weights * total).astype(int)
        raw_counts = np.maximum(raw_counts, self.config.min_sleep_per_regime)
        while int(np.sum(raw_counts)) > total:
            idx = int(np.argmax(raw_counts))
            if raw_counts[idx] > self.config.min_sleep_per_regime:
                raw_counts[idx] -= 1
            else:
                break
        while int(np.sum(raw_counts)) < total:
            raw_counts[int(np.argmax(weights))] += 1
        return raw_counts

    def _run_eohs_backbone(
        self,
        evaluation: Evaluation,
        round_id: int,
        *,
        initial_population: Optional[list[Function]] = None,
    ) -> list[Function]:
        round_sample_budget = self._round_sample_budget(round_id)
        self._last_eohs_sample_budget = round_sample_budget
        self._last_eohs_samples_used = 0
        if round_sample_budget <= 0:
            return [copy.deepcopy(func) for func in (initial_population or [])]
        profiler = self.profiler_factory(round_id) if self.profiler_factory else None
        if profiler is None and self.config.debug_mode:
            profiler = EoHSProfiler(log_dir=None)
        method = EoHS(
            llm=self.llm,
            evaluation=evaluation,
            profiler=profiler,
            max_sample_nums=round_sample_budget,
            max_generations=self.config.max_generations,
            pop_size=self.config.pop_size,
            num_samplers=self.config.num_samplers,
            num_evaluators=self.config.num_evaluators,
            initial_population=initial_population,
            debug_mode=self.config.debug_mode,
            **self.config.eohs_kwargs,
        )
        method.run()
        self._last_eohs_samples_used = int(getattr(method, "_tot_sample_nums", 0))
        self._eohs_sample_budget_used += self._last_eohs_samples_used
        return [copy.deepcopy(func) for func in method._population.population]

    def _round_sample_budget(self, round_id: int) -> int:
        total_budget = int(self.config.max_sample_nums)
        remaining_budget = max(0, total_budget - self._eohs_sample_budget_used)
        total_rounds = self.config.total_rounds
        if total_rounds is None:
            return remaining_budget
        remaining_rounds = max(1, int(total_rounds) - int(round_id))
        return int(math.ceil(remaining_budget / remaining_rounds))

    def _score_functions(self, functions: list[Function], evaluation: Evaluation) -> list[Function]:
        if not functions:
            return []
        secure_evaluator = SecureEvaluator(evaluation, debug_mode=self.config.debug_mode)
        scored: list[Function] = []
        template = evaluation.template_program
        for func in functions:
            program = TextFunctionProgramConverter.function_to_program(func, template)
            if program is None:
                continue
            copied = copy.deepcopy(func)
            copied.score = secure_evaluator.evaluate_program(program)
            scored.append(copied)
        return scored

    def _score_existing_portfolio(self, evaluation: Evaluation) -> list[Function]:
        """Backward-compatible wrapper for callers of the old helper."""
        return self._score_functions(self.portfolio, evaluation)


# Backward-compatible access for tests or users that imported these helpers
# from ow_cahd.py before the module split.
_extract_generate_instance_program = extract_generate_instance_program
