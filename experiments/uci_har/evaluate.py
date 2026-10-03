"""Deterministic five-seed UCI HAR evaluator (paper Sec. 5.5 and App. A.3.3 "Matched comparison": "UCI HAR
accuracy and macro-F1 are averaged over five MLP initializations").

For every unit record ``units/NNN.json`` the selected real features are evaluated by an MLP
(hidden layers 128-64, 300 iterations, no early stopping) trained on the official training split and
scored on the official test split, with evaluator seeds ``unit seed + 41 + 1000 k`` (k = 0..4) and
single-thread BLAS/OpenMP (``threadpoolctl``), so the result does not depend on the core count.
``Accuracy``/``F1_macro`` are the means over the five seeds (the paper numbers);
``Accuracy_seed0`` is the single-seed value stored by ``run.py``.

Usage (from the package root)::

    python experiments/uci_har/evaluate.py --units-dir results/uci_har/units --workers 4
    python experiments/uci_har/evaluate.py --from-reference --units 0      # evaluate the shipped selections

Outputs in ``--out`` (default: the parent of ``--units-dir``): ``eval5.json``
(``{"000_SSRK": {unit, method, Accuracy, F1_macro, Accuracy_seed0, accs, f1s}, ...}``) and
``per_run.csv`` (unit, seed, five-seed Accuracy/F1_macro, Accuracy_seed0, F1_seed0, HNI, DomainBalance) and
``eval5_manifest.json``.
Workers are independent processes, each with single-thread BLAS; results are identical for any
``--workers``. The same evaluator re-scores the stored baseline selections in
``experiments/uci_har/rescore.py --with-evaluator --with-baselines``.
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import argparse  # noqa: E402
import time  # noqa: E402
from typing import Sequence  # noqa: E402

import numpy as np  # noqa: E402

from experiments.common import (  # noqa: E402
    DEFAULT_DATA_DIR,
    display_path,
    REFERENCE_DIR,
    load_json,
    parse_units,
    resolve_path,
    save_json,
    write_csv,
    write_manifest,
)

#: Evaluator seed offsets: seed + 41 + 1000 k, k = 0..4 (k = 0 is the single-seed evaluator of run.py).
EVAL_OFFSETS = [41 + 1000 * k for k in range(5)]
#: Unit seeds of the 20 matched UCI HAR runs (unit i uses 31000 + 997 i).
REPEAT_SEEDS = tuple(31_000 + 997 * i for i in range(20))


def evaluate_uci(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected: Sequence[int],
    seed: int,
) -> tuple[float, float]:
    """Accuracy and macro-F1 of an MLP (128-64, max_iter 300, random_state=seed) on the selected columns.

    Call inside ``threadpool_limits(limits=1)`` for thread-count-independent results.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.neural_network import MLPClassifier

    cols = np.asarray(sorted(selected), dtype=int)
    classifier = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=int(seed),
        early_stopping=False,
    )
    classifier.fit(X_train[:, cols], y_train)
    prediction = classifier.predict(X_test[:, cols])
    return (
        float(accuracy_score(y_test, prediction)),
        float(f1_score(y_test, prediction, average="macro")),
    )


def evaluate_five_seeds(data, unit: int, seed: int, selected: Sequence[int], method: str = "SSRK") -> dict:
    """Five-seed evaluation of one selection with single-thread BLAS; returns the eval5 record."""
    from threadpoolctl import threadpool_limits

    Xtr, ytr, Xte, yte = data
    accs, f1s = [], []
    with threadpool_limits(limits=1):
        for off in EVAL_OFFSETS:
            a, f = evaluate_uci(Xtr, ytr, Xte, yte, selected, seed + off)
            accs.append(a)
            f1s.append(f)
    return dict(unit=unit, method=method, Accuracy=sum(accs) / 5, F1_macro=sum(f1s) / 5,
                Accuracy_seed0=accs[0], accs=accs, f1s=f1s)


def _worker(args) -> list[tuple[str, dict, float]]:
    """Process-pool worker: evaluate a list of ``(key, unit, seed, selected[, method])`` jobs."""
    data_dir, jobs = args
    from experiments.uci_har.data import load_uci_har

    data = load_uci_har(Path(data_dir))
    out = []
    for key, unit, seed, sel, *rest in jobs:
        t0 = time.perf_counter()
        rec = evaluate_five_seeds(data, unit, seed, sel, *rest)
        out.append((key, rec, time.perf_counter() - t0))
        print(f"{key} five-seed accuracy {rec['Accuracy']:.4f} ({time.perf_counter() - t0:.1f}s)", flush=True)
    return out


def run_jobs(data_dir: Path, jobs: Sequence[tuple], workers: int = 1) -> tuple[dict, dict]:
    """Evaluate ``(key, unit, seed, selected[, method])`` jobs on ``workers`` processes.

    Returns ``({key: eval5 record} sorted by key, {key: seconds})``; the records do not depend on
    ``workers`` (every evaluation runs with single-thread BLAS in a fresh MLP).
    """
    jobs = list(jobs)
    n_workers = max(1, min(int(workers), len(jobs)))
    if n_workers == 1:
        results = _worker((str(data_dir), jobs))
    else:
        import multiprocessing as mp

        chunks = [(str(data_dir), jobs[w::n_workers]) for w in range(n_workers)]
        with mp.get_context("spawn").Pool(n_workers) as pool:
            results = [r for part in pool.map(_worker, chunks) for r in part]
    eval5 = {key: rec for key, rec, _ in sorted(results, key=lambda r: r[0])}
    seconds = {key: s for key, _, s in results}
    return eval5, seconds


PER_RUN_FIELDS = ["unit", "seed", "Accuracy", "F1_macro", "Accuracy_seed0", "F1_seed0", "HNI", "DomainBalance"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units-dir", default="results/uci_har/units", help="directory with NNN.json unit records")
    ap.add_argument("--from-reference", action="store_true", help="evaluate reference/uci_har/units (shipped records)")
    ap.add_argument("--units", default="all", help='"all" (every record found) or e.g. "0,7"')
    ap.add_argument("--data-dir", default=None, help="data directory (default: <package>/data)")
    ap.add_argument("--out", default=None, help="output directory (default: parent of --units-dir, or "
                                                 "results/uci_har_reference_eval with --from-reference)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (each single-thread BLAS)")
    ap.add_argument("--check", action="store_true",
                    help="compare with the shipped five-seed records (reference/uci_har/eval5.json) and, for all "
                         "20 units, the means with the paired-statistics records")
    a = ap.parse_args(argv)

    units_dir = REFERENCE_DIR / "uci_har" / "units" if a.from_reference else resolve_path(a.units_dir, Path())
    if a.out:
        out = resolve_path(a.out, Path())
    elif a.from_reference:
        out = resolve_path("results/uci_har_reference_eval", Path())
    else:
        out = units_dir.parent
    data_dir = resolve_path(a.data_dir, DEFAULT_DATA_DIR)
    wanted = set(parse_units(a.units, len(REPEAT_SEEDS)))

    records = {}
    for f in sorted(units_dir.glob("[0-9][0-9][0-9].json")):
        rec = load_json(f)
        u = int(rec["unit"])
        if u not in wanted:
            continue
        if int(rec["seed"]) != REPEAT_SEEDS[u]:
            raise RuntimeError(f"{f}: seed {rec['seed']} is not the seed {REPEAT_SEEDS[u]} of unit {u}")
        records[u] = rec
    if not records:
        raise FileNotFoundError(f"no unit records for units {sorted(wanted)} in {units_dir}")

    jobs = [(f"{u:03d}_SSRK", u, int(records[u]["seed"]), records[u]["selected_indices"]) for u in sorted(records)]
    n_workers = max(1, min(int(a.workers), len(jobs)))
    t_all = time.perf_counter()
    eval5, seconds = run_jobs(data_dir, jobs, n_workers)

    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "eval5.json", eval5, indent=None)
    rows = []
    for u in sorted(records):
        e = eval5[f"{u:03d}_SSRK"]
        rows.append({"unit": u, "seed": records[u]["seed"], "Accuracy": repr(e["Accuracy"]),
                     "F1_macro": repr(e["F1_macro"]), "Accuracy_seed0": repr(e["Accuracy_seed0"]),
                     "F1_seed0": repr(e["f1s"][0]), "HNI": repr(float(records[u]["HNI"])),
                     "DomainBalance": repr(float(records[u]["DomainBalance"]))})
    write_csv(out / "per_run.csv", rows, PER_RUN_FIELDS)
    means = {k: float(np.mean([float(r[k]) for r in rows])) for k in ("Accuracy", "F1_macro", "HNI", "DomainBalance")}
    checks = {}
    if a.check:
        ref = load_json(REFERENCE_DIR / "uci_har" / "eval5.json")
        for key in eval5:
            checks[key] = eval5[key] == ref[key]
            print(f"check {key}: {'IDENTICAL' if checks[key] else 'DIFFERENT'}")
        if len(rows) == len(REPEAT_SEEDS):
            expected = load_json(REFERENCE_DIR / "uci_har" / "expected_ssrk_means.json")["means"]
            checks["means_equal_paper_records"] = all(means[k] == expected[k] for k in expected)
            print("check 20-unit means vs paired records:", checks["means_equal_paper_records"], expected)
    write_manifest(out, {
        "protocol": "uci_har_matched_eval5",
        "units_dir": display_path(units_dir),
        "units": sorted(records),
        "eval_offsets": EVAL_OFFSETS,
        "workers": n_workers,
        "blas_threads": 1,
        "unit_seconds": seconds,
        "total_seconds": time.perf_counter() - t_all,
        "means": means,
        "checks": checks,
    }, name="eval5_manifest.json")
    print("means over", len(rows), "units:", means)
    if checks and not all(checks.values()):
        print("REFERENCE CHECK FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
