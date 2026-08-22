# OVRP construct training

Open Vehicle Routing Problem (OVRP) constructive heuristics.

The construction task is: at each step, given the current node, remaining
capacity, unvisited nodes, demands, and the distance matrix, `select_next_node`
picks the next customer to serve (or the depot to start a new route). Routes
are open: the last customer of a vehicle does not return to the depot. The
objective is to minimize the total traveled distance; reported scores are the
gap vs a nearest-neighbor baseline: `(baseline - cost) / baseline` (0 = NN,
positive = better than NN).

Train on the IID train datasets — 4 coordinate families (uniform, cluster,
bezier, grid_holes) × 32 instances each = 128 instances, using a near-balanced
mixture of 20, 50, and 100 customers
(train_datasets/family_<family>_mixed_sizes.pkl). Post-evaluate on the ID test datasets
(test_datasets/size_20/50/100.pkl) and the OOD datasets
(ood_test_datasets/mixture/size_20/50/100.pkl), 128 instances per size each
(384 total per split).

ID instances are balanced across the same 4 train families (32 per family per
size); OOD instances are gaussian mixtures with random Dirichlet weights
(mixed rings/spiral/gaussian-blob/line structures), so they are out-of-family
relative to training. Regenerate the mixed-size train data with:

    py -3.10 scripts/generate_mixed_size_routing_train.py --tasks ovrp

Regenerate the fixed-size hidden ID/OOD data separately with:

    py -3.10 scripts/generate_ovrp_vrptw_datasets.py ovrp

Post-eval uses the whole population for EoH/EoHS and the top-10 train-score
individuals for MCTS_AHD. OW-CAHD synthesizes new coordinate regimes and
derives demands/capacity/distance matrices deterministically from the
generated coordinates.

Configs: cfg/ovrp_<method>.yaml for
method in eoh / eohs / mcts_ahd / ow_cahd.

Set OPENAI_API_KEY and optionally OPENAI_BASE_URL / OPENAI_MODEL, then from
the repository root (Python 3.10 recommended):

    py -3 examples\training\ovrp_construct\run_eoh.py
    py -3 examples\training\ovrp_construct\run_eohs.py
    py -3 examples\training\ovrp_construct\run_mcts_ahd.py
    py -3 examples\training\ovrp_construct\run_ow_cahd.py

Per run, the script writes run_config.json, token_usage.json, and one
post_eval_hidden_<stem>.csv (+ .json) per hidden ID/OOD dataset into
logs/ovrp/<method>/&lt;timestamp&gt;.
