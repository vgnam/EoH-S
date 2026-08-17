# BP2D construct training

Train on ONE IID train size (uniform family): size 50 with 32 instances by default. BP2D packing is expensive in pure Python, so sizes above 50 are used only for hidden eval. Post-eval uses the whole population for EoH/EoHS and the top-10 train-score individuals for MCTS_AHD.

Post-evaluates on the ID test datasets (test_datasets/size_*.pkl) and OOD datasets (ood_test_datasets/mixture/size_*.pkl) for ALL sizes 50/100/200/500 with 32 instances per size (size 500 is slow: ~90s/instance).

Configs: cfg/bp2d_<method>.yaml for
method in eoh / eohs / mcts_ahd / ow_cahd.

Set OPENAI_API_KEY and optionally OPENAI_BASE_URL / OPENAI_MODEL, then from
the repository root (Python 3.10 recommended):

    py -3 examples\training\bp_2d_construct\run_eoh.py
    py -3 examples\training\bp_2d_construct\run_eohs.py
    py -3 examples\training\bp_2d_construct\run_mcts_ahd.py
    py -3 examples\training\bp_2d_construct\run_ow_cahd.py

Per run, the script writes run_config.json, token_usage.json, and one
post_eval_hidden_<stem>.csv (+ .json) per hidden ID/OOD dataset into
logs/bp2d/<method>/&lt;timestamp&gt;.
