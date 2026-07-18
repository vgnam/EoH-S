# CVRP open-world training

The CVRP pipeline mirrors the TSP protocol:

- training: 32 instances for each of `uniform`, `cluster`, `bezier`, and `grid_holes`, with 30 customers;
- hidden ID: 128 instances at each of 20, 50, and 100 customers, balanced across the four training families;
- hidden OOD: 128 instances at each size, all from `mixed_structures`;
- OOD component weights are sampled per instance from `Dirichlet(3.5, 2.5, 2.5, 1.5)` before multinomial customer allocation;
- customer demands follow the same distribution in ID and OOD; capacity scales with total demand;
- reference costs use Clarke-Wright Savings followed by route-level 2-opt.

From the repository root, configure `OPENAI_API_KEY` and optionally
`OPENAI_BASE_URL` / `OPENAI_MODEL`, then run:

```powershell
py -3 examples\training\cvrp_set\run_eohs.py
py -3 examples\training\cvrp_set\run_ow_cahd.py
```

Configuration lives in `cfg/cvrp_eohs.yaml` and `cfg/cvrp_ow_cahd.yaml`.
Results are written below `examples/training/cvrp_set/logs/`.
