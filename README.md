# SSRK Synthetic Oracle Demo

This package contains a minimal SSRK-only code release for the synthetic oracle experiment.
It contains only the SSRK implementation needed for the demo and the archived reference outputs.

## Contents

- `ssrk/`: SSRK model, statistic, training utility, and oracle synthetic knockoff generator.
- `scripts/run_synthetic_symmetric_oracle.py`: runnable synthetic oracle demo.
- `results/reference/`: archived reference JSON/CSV from the manuscript-facing run.
- `requirements.txt`: minimal Python dependencies.

## Install

Create a Python environment, then install:

```bash
pip install -r requirements.txt
```

The experiments were generated on CPU. A GPU can run the script, but exact numeric
reproduction should use CPU with the fixed thread setting below.

## Run the Full Demo

```bash
python scripts/run_synthetic_symmetric_oracle.py --device cpu --out results/synthetic_symmetric_oracle_demo.json --csv-out results/synthetic_symmetric_oracle_demo.csv
```

The default demo runs four fixed settings with 10 seeds:

| Setting | Reference FDP | Reference power | Reference avg selected |
|---|---:|---:|---:|
| `p=100, s=50` | `0.088 +/- 0.029` | `1.000 +/- 0.000` | `54.9 +/- 1.8` |
| `p=200, s=60` | `0.097 +/- 0.032` | `1.000 +/- 0.000` | `66.5 +/- 2.4` |
| `p=300, s=90` | `0.078 +/- 0.025` | `1.000 +/- 0.000` | `97.7 +/- 2.6` |
| `p=500, s=150` | `0.090 +/- 0.034` | `1.000 +/- 0.000` | `165.1 +/- 6.5` |

FDP values are means over the fixed seeds. Individual seed-level FDP can exceed
`q = 0.10`; the reported statement is about repeated-run mean FDP/FDR behavior
under this fixed oracle protocol.

## Quick Smoke Test

For a fast installation check:

```bash
python scripts/run_synthetic_symmetric_oracle.py --device cpu --condition smoke:20:5 --seeds 0 --n-samples 128 --epochs 1 --batch-size 32 --out results/smoke.json --csv-out results/smoke.csv
```

This smoke test only verifies that the package executes; it is not a paper result.
