# TSP ACO heuristic search

These runners use EoH-S or OW-CAHD to evolve `update_pheromone` inside a fixed
TSP Ant Colony Optimisation harness. This is different from the key-free
classical Ant System runner under `examples/baselines/aco`.
The function contract follows the official
[FeiLiu36/EoH ACO example](https://github.com/FeiLiu36/EOH/tree/main/examples/aco_pheromone).

Set the LLM endpoint from the repository root:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
$env:OPENAI_BASE_URL="YOUR_BASE_URL"
$env:OPENAI_MODEL="YOUR_MODEL"
```

Run the two search methods:

```powershell
py -3 examples\training\tsp_aco_set\run_eohs.py
py -3 examples\training\tsp_aco_set\run_ow_cahd.py
```

The matched configs are `cfg/tsp_aco_eohs.yaml` and
`cfg/tsp_aco_ow_cahd.yaml`. Both train on only the uniform TSP30 file by
default. Each runner automatically post-evaluates the fixed final population
or portfolio on ID/OOD sizes 20, 50, and 100.

The evolved function receives all ant tours and costs, the global-best tour,
the evaporation parameter, and iteration progress. The evaluator rejects a
matrix with a wrong shape, non-finite values, or negative pheromone.
