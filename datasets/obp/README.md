# OBP train, ID, and OOD datasets

All instances use bin capacity 100. Dataset size means the number of items in
each online arrival stream.

| Split | Sizes | Instances per size | Families |
| --- | --- | ---: | --- |
| Train | 200, 500, 1000 | 128 | uniform, normal, Weibull, exponential |
| Test ID | 200, 500, 1000 | 128 | the four train families, generated with held-out seeds |
| Test OOD | 200, 500, 1000 | 128 | bimodal, U-shaped beta, complementary pairs, discrete spikes |

Each file is a pickle dictionary whose values retain the schema expected by
`OBPSEvaluation`: `capacity`, `num_items`, and `items`. It also includes
`split`, `family`, `parameters`, and `seed` metadata for auditing.

Regenerate every split deterministically from the repository root:

```powershell
py -3 datasets\obp\generate_obp_datasets.py
```

Change the sizes or sample counts without editing code:

```powershell
py -3 datasets\obp\generate_obp_datasets.py `
  --train-sizes 200 500 1000 `
  --test-sizes 200 500 1000 `
  --train-instances 128 `
  --test-instances 128 `
  --seed 32026
```

The split seeds occupy separate deterministic ranges: train starts at 32026,
ID at 42026, and OOD at 52026. Families are balanced within every file.
