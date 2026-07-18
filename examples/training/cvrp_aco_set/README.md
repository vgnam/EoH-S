# CVRP ACO heuristic search

These runners use EoH-S or OW-CAHD to evolve `update_pheromone` inside a fixed,
capacity-feasible CVRP Ant Colony Optimisation harness. Every ant route starts
and ends at depot 0, and infeasible customer transitions are masked centrally.
It generalizes the pheromone-update contract from the official
[FeiLiu36/EoH TSP ACO example](https://github.com/FeiLiu36/EOH/tree/main/examples/aco_pheromone)
to nested capacity-feasible CVRP routes.

Set the LLM endpoint from the repository root:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
$env:OPENAI_BASE_URL="YOUR_BASE_URL"
$env:OPENAI_MODEL="YOUR_MODEL"
```

Run the two search methods:

```powershell
py -3 examples\training\cvrp_aco_set\run_eohs.py
py -3 examples\training\cvrp_aco_set\run_ow_cahd.py
```

The matched configs are `cfg/cvrp_aco_eohs.yaml` and
`cfg/cvrp_aco_ow_cahd.yaml`. Both train on only the uniform CVRP30 file by
default. Each runner automatically post-evaluates the fixed final population
or portfolio on ID/OOD sizes 20, 50, and 100 customers.

The evolved function receives all capacity-feasible ant solutions and costs,
the global-best solution, the evaporation parameter, and iteration progress.
The evaluator rejects a matrix with a wrong shape, non-finite values, or
negative pheromone.
