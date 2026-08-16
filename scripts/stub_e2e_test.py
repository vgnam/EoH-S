from __future__ import annotations

"""Mini end-to-end run with a stub LLM (no network): trains EoH, EoHS,
MCTS_AHD on OBP and post-evaluates the final populations on one hidden
ID and one OOD dataset."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "examples" / "training"))
sys.path.insert(0, str(REPO_ROOT / "examples" / "training" / "obp_set"))

from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.method.eohs import EoHS, EoHSProfiler
from llm4ad.method.mcts_ahd import MCTS_AHD, MAProfiler
from common import hidden_eval_factory, hidden_specs, make_evaluation
from post_eval_common import load_instances, run_post_eval, resolve_repo_path


class StubLLM:
    def __init__(self):
        self.debug_mode = False
        self._counter = 0
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def draw_sample(self, prompt, temperature=1.0):
        self._counter += 1
        # Each sample is a valid, slightly different priority function.
        # EoH/EoHS/MCTS prompts require a {thought} bracket before the code.
        return (
            "{stub thought " + str(self._counter) + "}\n"
            "def priority(item: float, bins) -> np.ndarray:\n"
            "    return item - bins + " + str(self._counter) + " * 0.0\n"
        )

    def token_usage(self):
        return dict(self._usage)

    def close(self):
        pass


def main():
    train_path = resolve_repo_path("datasets/obp/dataset_obp_train_size200.pkl")
    instances = load_instances(train_path)[:4]

    id_path = str(resolve_repo_path("datasets/obp/dataset_obp_hidden_id_size200.pkl"))
    ood_path = str(resolve_repo_path("datasets/obp/dataset_obp_hidden_ood_size200.pkl"))
    hidden_cfg = {
        "id_datasets": [id_path],
        "ood_datasets": [ood_path],
        "function_timeout_seconds": 120,
    }
    specs = hidden_specs(hidden_cfg)[:2]
    factory = hidden_eval_factory(hidden_cfg)

    for name, method_cls, profiler_cls, kwargs in [
        ("eoh", EoH, EoHProfiler, dict(max_sample_nums=6, max_generations=2, pop_size=2)),
        ("eohs", EoHS, EoHSProfiler, dict(max_sample_nums=6, max_generations=2, pop_size=2)),
        ("mcts_ahd", MCTS_AHD, MAProfiler, dict(max_sample_nums=6, pop_size=2)),
    ]:
        llm = StubLLM()
        evaluation = make_evaluation(instances, timeout_seconds=120)
        method = method_cls(
            llm=llm,
            evaluation=evaluation,
            profiler=profiler_cls(
                log_dir=None, log_style="simple", create_random_path=False
            ),
            num_samplers=1,
            num_evaluators=1,
            debug_mode=False,
            **kwargs,
        )
        method.run()
        population = method._population.population
        scores = [getattr(f, "score", None) for f in population]
        print(f"{name}: population={len(population)} scores={scores}")
        assert population, f"{name} produced an empty population"
        assert all(score is not None for score in scores), f"{name} produced unscored functions"
        summaries = run_post_eval(
            REPO_ROOT / "examples" / "training" / "obp_set" / "logs_stub",
            name,
            specs,
            factory,
            population,
        )
        assert len(summaries) == 2, f"{name} post-eval produced {len(summaries)} summaries"
        for summary in summaries:
            assert summary["best_mean"] is not None, f"{name} post-eval best_mean is None"
    print("STUB END-TO-END OK")


if __name__ == "__main__":
    main()
