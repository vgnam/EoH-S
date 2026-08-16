# BP1D construct training

Train on ONE IID train size (beta family): size 100 with 64 instances by default. Post-eval uses the whole population for EoH/EoHS and the top-10 train-score individuals for MCTS_AHD.

Post-evaluates on the ID test datasets (test_datasets/size_*.pkl) and OOD datasets (ood_test_datasets/mixture/size_*.pkl) for ALL sizes 100/500/1000/2000 with 32 instances per size.

Configs: cfg/bp1d_<method>.yaml for
method in eoh / eohs / mcts_ahd / ow_cahd.

Set OPENAI_API_KEY and optionally OPENAI_BASE_URL / OPENAI_MODEL, then from
the repository root (Python 3.10 recommended):

    py -3 examples\training\bp_1d_construct\run_eoh.py
    py -3 examples\training\bp_1d_construct\run_eohs.py
    py -3 examples\training\bp_1d_construct\run_mcts_ahd.py
    py -3 examples\training\bp_1d_construct\run_ow_cahd.py

Per run, the script writes run_config.json, token_usage.json, and one
post_eval_hidden_<stem>.csv (+ .json) per hidden ID/OOD dataset into
examples/training/bp_1d_construct/logs/<method>.
