# Open-world TSP surprise benchmark

This benchmark compares the released TSP heuristic sets, including EoH-S, on a
stream where one previously unseen instance regime appears in the middle.

It is intentionally separate from the training code:

- no LLM calls
- no changes to `llm4ad` evaluators
- no dependency on `elkai`
- uses the released `heuristics/heuristics/heuristics_tsp_*_top10.py` files

## Protocol

The benchmark builds a sequence of TSP rounds. Early rounds use known synthetic
regimes such as uniform, clustered, diagonal, and grid-like instances. A
`bezier_surprise` regime is inserted at a fixed middle round to test
out-of-distribution behavior.

Each heuristic is scored with the same convention as the existing TSP evaluator:

```text
score = -(heuristic_tour_length - reference_tour_length) / reference_tour_length
```

Higher is better. The reference tour is computed by a deterministic nearest
neighbor multi-start solver followed by 2-opt, so the benchmark can run without
external TSP solvers.

For a heuristic file containing top-k functions, the default score is
`best-of-k`, matching the EoH-S heuristic-set objective. Use `--mode single` to
evaluate only the first function in each file.

## Metrics

Per run:

- `mean_score`: average over all generated instances
- `pre_score`: average before the surprise round
- `surprise_score`: average on the first surprise round
- `dip_depth`: `max(0, pre_score - surprise_score)`
- `recovery_time`: first round offset after surprise where score returns to
  `pre_score - tolerance`; `NA` means not recovered within the stream

Across runs, `method_summary.csv` reports mean and standard deviation of those
metrics.

## Run

From the repository root:

```bash
python benchmarks/open_world_tsp_surprise/run_benchmark.py
```

On Windows, if `python` is not on `PATH`, use:

```bash
py -3 benchmarks/open_world_tsp_surprise/run_benchmark.py
```

Quick smoke test:

```bash
python benchmarks/open_world_tsp_surprise/run_benchmark.py --n-cities 20 --instances-per-round 2 --max-functions 3
```

Useful options:

```bash
python benchmarks/open_world_tsp_surprise/run_benchmark.py --mode single
python benchmarks/open_world_tsp_surprise/run_benchmark.py --methods eohs eoh funsearch reevo
python benchmarks/open_world_tsp_surprise/run_benchmark.py --output-dir results/open_world_tsp_surprise
```

Outputs:

- `per_round.csv`: score by method, run, and stream round
- `run_summary.csv`: metrics for each method run
- `method_summary.csv`: aggregate metrics by method
