# OBP multi-size training

The OBP EoH-S setup uses exactly one train size per run. Size 500 is the
default; the available independent train sets contain 200, 500, or 1000 items
per instance. Configuration is in `cfg/obp_eohs.yaml`.

Set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` / `OPENAI_MODEL`, then run
from the repository root:

```powershell
py -3 examples\training\obp_set\run_eohs.py
```

Select a different train set with `--train-size`:

```powershell
py -3 examples\training\obp_set\run_eohs.py --train-size 200
py -3 examples\training\obp_set\run_eohs.py --train-size 1000
```

Logs are separated into `size200`, `size500`, and `size1000` directories so
runs using different train sets do not overwrite each other.

See `datasets/obp/README.md` for the ID/OOD split definitions and the command
used to regenerate all data files.
