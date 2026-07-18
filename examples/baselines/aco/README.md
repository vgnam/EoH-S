# ACO baselines: TSP, CVRP, and BPP

This directory extends the classical Ant System baseline from
[FeiLiu36/EoH](https://github.com/FeiLiu36/EoH/tree/main/examples/aco_pheromone)
to the three dataset schemas used in this repository. It is self-contained and
does not call an LLM.

For the LLM-driven experiments where EoH-S or OW-CAHD searches the pheromone
update rule, use `examples/training/tsp_aco_set` and
`examples/training/cvrp_aco_set`. Those search runners require an API key; this
directory remains the fixed, key-free Ant System reference.

## Methods

- **TSP:** each ant samples a Hamiltonian tour with transition probability
  proportional to `pheromone^alpha * (1 / distance)^beta`. All ants deposit
  `1 / tour_cost` after evaporation.
- **CVRP:** each ant constructs depot-to-depot routes and masks customers that
  exceed the remaining vehicle capacity. Feasible route edges use the same Ant
  System transition and update rule as TSP.
- **BPP:** position-based pheromone samples a complete item ordering, which is
  decoded by Best Fit. Best Fit Decreasing seeds the incumbent. This algorithm
  sees the complete sequence and is therefore an **offline BPP baseline**, not
  an online OBP policy. Its scores must not be presented as a like-for-like
  comparison with EoH-S/OW-CAHD online bin-priority heuristics.

TSP and CVRP report the repository's utility `(reference - cost) / reference`.
BPP uses the volume lower bound and reports `-(bins - lower_bound) / lower_bound`,
matching the sign convention of the OBP evaluator.

## Run

From the repository root:

```powershell
# Quick smoke benchmark: two instances, one stochastic run
py -3 examples\baselines\aco\run_aco.py `
  --problem all --split id --max-instances 2 --runs 1 `
  --ants 4 --iterations 5

# Full TSP ID/OOD protocol at n = 20, 50, 100
py -3 examples\baselines\aco\run_aco.py `
  --problem tsp --split all --max-instances 0 --runs 10

# CVRP OOD only
py -3 examples\baselines\aco\run_aco.py `
  --problem cvrp --split ood --sizes 20 50 100 --max-instances 0

# Offline BPP ACO at 200, 500, and 1000 items
py -3 examples\baselines\aco\run_aco.py `
  --problem bpp --split all --sizes 200 500 1000 --max-instances 0
```

`--split all` includes train, ID, and OOD. Routing train data has fixed size 30,
so use `--sizes 30` (or omit `--sizes`) when selecting it. By default the runner
uses the first 16 instances per dataset, 3 seeds, 20 ants, and 100 iterations.
Set `--max-instances 0` for every instance.

Each run writes a detailed JSON file and a flat CSV file under
`results/aco_baselines/`. Use `--output path/to/name.json` to choose a different
location.
