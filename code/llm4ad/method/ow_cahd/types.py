from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ...base import Evaluation, Function


Instance = Any
DescriptorFn = Callable[[Instance], np.ndarray]
GeneratorFn = Callable[[int, int], list[Instance]]
EvaluationFactory = Callable[[list[Instance]], Evaluation]
RegimeSynthesizer = Callable[[list[Instance], DescriptorFn, int], "Regime"]
ValidityFn = Callable[[Instance], bool]


@dataclass
class Regime:
    """A known instance regime in OW-CAHD."""

    name: str
    generator: GeneratorFn
    description: str = ""
    mu: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    n_fit: int = 0
    archive: list[Instance] = field(default_factory=list)
    generator_program: str = ""
    generator_programs: list[str] = field(default_factory=list)
    mixture_mus: list[np.ndarray] = field(default_factory=list)
    mixture_covs: list[np.ndarray] = field(default_factory=list)
    mixture_weights: Optional[np.ndarray] = None
    mixture_n_fit: list[int] = field(default_factory=list)
    mixture_temperatures: list[float] = field(default_factory=list)

    def sample(self, n: int, seed: int) -> list[Instance]:
        return list(self.generator(n, seed))


@dataclass
class OWCAHDConfig:
    """Configuration for the OW-CAHD controller."""

    portfolio_size: int = 5
    sleep_instances_per_round: int = 64
    min_sleep_per_regime: int = 2
    novelty_alpha: float = 0.01
    novelty_confirm_rounds: int = 2
    novelty_buffer_size: int = 64
    new_regime_blend: float = 0.35
    belief_floor: float = 0.0
    covariance_reg: float = 1e-6
    fit_samples_per_regime: int = 128
    mmd_samples: int = 64
    mmd_pvalue_threshold: float = 0.05
    mmd_permutations: int = 99
    skip_mmd_verification: bool = False
    auto_accept_regime: bool = False
    always_synthesize_regime: bool = False
    llm_synthesis_retries: int = 3
    llm_synthesis_mixture_components: int = 1
    llm_synthesis_min_successful_components: int = 1
    llm_synthesis_temperatures: list[float] = field(
        default_factory=lambda: [1.0]
    )
    llm_synthesis_examples: int = 4
    llm_synthesis_max_chars: int = 12000
    synthesis_context: str = ""
    allow_bootstrap_fallback: bool = True
    allocation_temperature: float = 0.5
    sticky_transition: float = 0.92
    max_sample_nums: int = 100
    max_generations: Optional[int] = 10
    total_rounds: Optional[int] = None
    pop_size: int = 5
    num_samplers: int = 1
    num_evaluators: int = 1
    debug_mode: bool = False
    print_eohs_samples: bool = False
    eohs_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OWCAHDRoundResult:
    round_id: int
    novelty_score: float
    novelty_threshold: float
    novelty_triggered: bool
    accepted_regime: Optional[str]
    belief: dict[str, float]
    sleep_instances: int
    portfolio: list[Function]
    candidate_pool: list[Function] = field(default_factory=list)
    eohs_sample_budget: int = 0
    eohs_samples_used: int = 0
    eohs_total_samples_used: int = 0
    accepted_regime_description: Optional[str] = None
    accepted_regime_generator_program: Optional[str] = None
    accepted_regime_generator_programs: list[str] = field(default_factory=list)
    accepted_regime_mixture_mus: list[list[float]] = field(default_factory=list)
    accepted_regime_mixture_covs: list[list[list[float]]] = field(default_factory=list)
    accepted_regime_mixture_weights: list[float] = field(default_factory=list)
    accepted_regime_mixture_n_fit: list[int] = field(default_factory=list)
    accepted_regime_mixture_temperatures: list[float] = field(default_factory=list)
