# Admissible set construct training

Train on the IID train manifest (datasets/admissible/train/asp_train.pkl, n15w10) with all 32 instances by default. Post-eval uses the whole population for EoH/EoHS and the top-10 train-score individuals for MCTS_AHD.

Post-evaluates on the ID manifest (asp_id.pkl, n15w10, 32 instances) and OOD manifest (asp_ood.pkl, n12w7 / n21w15 / n24w17). n21w15 and n24w17 are combinatorially expensive, so OOD eval defaults to 3 instances (1 per family, family-balanced) with a 6h timeout.

Configs: cfg/admissible_<method>.yaml for
method in eoh / eohs / mcts_ahd / ow_cahd.

Set OPENAI_API_KEY and optionally OPENAI_BASE_URL / OPENAI_MODEL, then from
the repository root (Python 3.10 recommended):

    py -3 examples\training\admissible_set\run_eoh.py
    py -3 examples\training\admissible_set\run_eohs.py
    py -3 examples\training\admissible_set\run_mcts_ahd.py
    py -3 examples\training\admissible_set\run_ow_cahd.py

Per run, the script writes run_config.json, token_usage.json, and one
post_eval_hidden_<stem>.csv (+ .json) per hidden ID/OOD dataset into
examples/training/admissible_set/logs/<method>.
