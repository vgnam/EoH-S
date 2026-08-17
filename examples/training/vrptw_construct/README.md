# VRPTW construct training

Vehicle Routing Problem with Time Windows (VRPTW, the capacitated variant is
often called CVRPTW) constructive heuristics.

The construction task is: at each step, given the current node, remaining
capacity, current time, unvisited nodes, demands, the distance matrix, and the
time windows, `select_next_node` picks the next customer to serve (or the
depot to start a new route). Routes must respect capacity and time-window
constraints; the objective is to minimize the total traveled distance.

Train on the IID train datasets (train_datasets/size_50.pkl and
train_datasets/size_100.pkl, 16 instances each) and post-evaluate on the ID
test datasets (test_datasets/size_*.pkl) and OOD datasets
(ood_test_datasets/mixture/size_*.pkl) for ALL sizes 20/50/100/200.

Post-eval uses the whole population for EoH/EoHS and the top-10 train-score
individuals for MCTS_AHD. OW-CAHD synthesizes new coordinate regimes and
derives demands/capacity/service times/time windows/distance matrices
deterministically from the generated coordinates.

Configs: cfg/vrptw_<method>.yaml for
method in eoh / eohs / mcts_ahd / ow_cahd.

Set OPENAI_API_KEY and optionally OPENAI_BASE_URL / OPENAI_MODEL, then from
the repository root (Python 3.10 recommended):

    py -3 examples\training\vrptw_construct\run_eoh.py
    py -3 examples\training\vrptw_construct\run_eohs.py
    py -3 examples\training\vrptw_construct\run_mcts_ahd.py
    py -3 examples\training\vrptw_construct\run_ow_cahd.py

Per run, the script writes run_config.json, token_usage.json, and one
post_eval_hidden_<stem>.csv (+ .json) per hidden ID/OOD dataset into
logs/vrptw/<method>/&lt;timestamp&gt;.