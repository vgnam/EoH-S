# OBP construct training (TSP/CVRP-style IID train + ID/OOD hidden eval)

All four methods (EoH, EoHS, MCTS_AHD, OW-CAHD) follow the TSP/CVRP construct
protocol: train on ONE IID train size (size 500 with 64 instances by default),
then post-evaluate on the hidden ID and OOD datasets of ALL sizes (200/500/1000,
128 instances per size). EoH/EoHS post-eval the whole population; MCTS_AHD
post-evals its top-10 train-score individuals.

Configs: cfg/obp_eoh.yaml, cfg/obp_eohs.yaml, cfg/obp_mcts_ahd.yaml,
cfg/obp_ow_cahd.yaml.

Set OPENAI_API_KEY and optionally OPENAI_BASE_URL / OPENAI_MODEL, then run
from the repository root (Python 3.10 recommended):

    py -3 examples\training\obp_set\run_eoh.py
    py -3 examples\training\obp_set\run_eohs.py
    py -3 examples\training\obp_set\run_mcts_ahd.py
    py -3 examples\training\obp_set\run_ow_cahd.py

Per run, the script writes run_config.json, token_usage.json, and one
post_eval_hidden_<id|ood>_size<size>.csv (+ .json) per hidden dataset into
the configured log directory (examples/training/obp_set/logs/<method>).

See datasets/obp/README.md for the ID/OOD split definitions and the command
used to regenerate all data files.
