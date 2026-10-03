#!/usr/bin/env bash
# Full from-scratch reproduction of the paper's experiments: about 1 h 30 min on one RTX 4090 Laptop GPU,
# plus the UCI HAR download.
# Usage: bash run_all.sh            (uses `python` from PATH; override with PYTHON=/path/to/python)
set -euo pipefail
PY="${PYTHON:-python}"
case "$PY" in */*) PY="$(cd "$(dirname -- "$PY")" && pwd)/$(basename -- "$PY")";; esac  # absolute before cd
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
WORKERS="${UCI_EVAL_WORKERS:-4}"
step() { printf '\n=== %s  [%s] ===\n' "$*" "$(date '+%Y-%m-%d %H:%M:%S')"; }

step "0/6 data: verify shipped PBMC file, download + verify UCI HAR"
"$PY" data/download.py --all
step "1/6 unit tests"
"$PY" -m pytest -q tests
step "2/6 exact Gaussian fixed-target study (~40 min)"
"$PY" experiments/exact_gaussian/run.py main
"$PY" experiments/exact_gaussian/run.py ablation
"$PY" experiments/exact_gaussian/run.py tune
"$PY" experiments/exact_gaussian/run.py audit
"$PY" experiments/exact_gaussian/make_tables.py --check
"$PY" experiments/exact_gaussian/make_figure.py --check
step "3/6 PBMC 3k matched protocol (~11 min)"
"$PY" experiments/pbmc/run.py --units all
"$PY" experiments/pbmc/stability.py --strict
step "4/6 UCI HAR matched protocol (~31 min) and five-seed evaluator (~5 min)"
"$PY" experiments/uci_har/run.py --units all
"$PY" experiments/uci_har/evaluate.py --workers "$WORKERS"
step "5/6 paired statistics (Table A3, 70 contrasts)"
"$PY" experiments/paired/paired_stats.py
"$PY" experiments/paired/make_table.py
step "6/6 bitwise verification against reference/ and the paper"
"$PY" verify.py --require-all
step "done"
