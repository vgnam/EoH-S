# Method x Task training matrix

Every construct task here follows the TSP/CVRP construct protocol:

1. train on ONE IID train size (32-64 instances),
2. run the method (EoH / EoHS / MCTS_AHD / OW-CAHD),
3. post-evaluate on held-out ID (test) and OOD datasets across ALL sizes,
   writing post_eval_hidden_<stem>.csv (+ .json) per dataset.

Post-eval population policy:

- EoH / EoHS: the whole final population (like OW-CAHD uses its whole
  portfolio).
- MCTS_AHD: top-10 individuals by training score.

## Coverage

| task | dir | train (IID) | hidden test (ID + OOD) |
| --- | --- | --- | --- |
| tsp_construct_set | tsp_set | 4 fixed30 families | size 20/50/100 x128 |
| cvrp_construct_set | cvrp_set | 4 fixed30 families | size 20/50/100 x128 |
| bp_1d_construct | bp_1d_construct | size 100 (64) | size 100/500/1000/2000 (32/size) |
| bp_2d_construct | bp_2d_construct | size 50 (32) | size 50/100/200/500 (32/size) |
| admissible_set | admissible_set | n15w10 (32) | id n15w10 (32) + ood n12w7/n21w15/n24w17 (1/family) |
| online_bin_packing_set | obp_set | size 500 (64) | size 200/500/1000 (128/size) |

Train size / instance caps are tunable in cfg/<task>_<method>.yaml
(train_datasets, train_instances*, eval_instances, *_eval_instances).

## Running

Python 3.10 recommended (numpy 1.26 + numba 0.59). Put your API key in the
repo-root .env file (auto-loaded by every run script; not committed):

    OPENAI_API_KEY=sk-...
    OPENAI_BASE_URL=https://opencode.ai/zen/go/v1
    OPENAI_MODEL=deepseek-v4-flash

Note: use the py -3.10 launcher (py -3 resolves to Python 3.11 whose numpy
2.x breaks numba). From the repository root:

    py -3.10 examples\training\<task_dir>\run_<method>.py

## Verification (no LLM needed)

    py -3.10 scripts\verify_matrix.py
