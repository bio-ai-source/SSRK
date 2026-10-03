#!/usr/bin/env python3
"""Rescore the PBMC 3k matched protocol from the shipped statistics W (Level A; CPU only, no torch).

For each of the 20 units this script recomputes, from ``NNN_W.npy`` alone, everything that
``experiments/pbmc/run.py`` derives after the fit (the same function, :func:`experiments.pbmc.run.score_unit`):

* the row bootstrap of the unit (``default_rng(seed)``, seed = 20000 + 997 unit);
* the top 100 genes of ``W`` (deterministic ranking, ties by the smaller index);
* ARI / NMI of randomized PCA (50 components) + k-means (k = 12, 10 initializations) on the selected
  genes against the Leiden clusters, evaluation seed = unit seed + 41, single-thread BLAS/OpenMP
  (``--eval-threads``);
* MarkerHits and CoverageScore (curated markers of paper Table A7, App. A.8 "Metric definitions").

It writes ``--out/units/NNN.json`` in the format of ``run.py`` (``meta`` copied from the source record)
plus byte copies of ``NNN_W.npy`` / ``NNN_Wall.npy``, so that ``experiments/paired/paired_stats.py
--pbmc-units --out/units`` (PBMC rows of Table A3) and ``experiments/pbmc/stability.py --units-dir
--out/units --strict`` (Table A8) run on the rescored records.

Checks (all must hold, otherwise the exit status is 1):

* every recomputed field (``selected_indices``, ARI, NMI, MarkerHits, CoverageScore) equals the source
  record exactly, and the written ``NNN.json`` is byte-identical to the source file;
* ``W`` equals the float64 mean over the restarts in ``Wall``, cast to float32, bitwise (PBMC: one restart);
* the 20-unit means equal the SSRK means of the paired-statistics record
  (``reference/pbmc/expected_ssrk_means.json``: ARI .3549, NMI .5819, marker hits 14.00, coverage 2.800);
* ``--with-baselines``: ARI, NMI, MarkerHits and CoverageScore of the five baselines, recomputed from their
  stored selections (``reference/pbmc/baseline_selections.csv``) with the same evaluator, equal the shipped
  records ``experiments/paired/records/pbmc_baselines_per_run.csv`` in all 100 rows.

Usage (from the package root)::

    python experiments/pbmc/rescore.py --out results/verify_from_reference/pbmc_rescore
    python experiments/pbmc/rescore.py --out results/pbmc_rescore --with-baselines

Outputs in ``--out``: ``units/NNN.json``, ``units/NNN_W.npy``, ``units/NNN_Wall.npy``, ``per_run.csv``,
``baselines_per_run.csv`` (with ``--with-baselines``), ``rescore.json`` (all checks) and
``rescore_manifest.json``. About 15 s (30 s with ``--with-baselines``) on one CPU core.
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from experiments.common import (  # noqa: E402
    DEFAULT_DATA_DIR,
    REFERENCE_DIR,
    blas_threads,
    display_path,
    load_json,
    resolve_path,
    restart_mean,
    same_array,
    save_json,
    write_manifest,
)
from experiments.pbmc.data import bootstrap_resample, load_pbmc, pbmc_marker_indices  # noqa: E402
from experiments.pbmc.run import (  # noqa: E402
    CHECK_KEYS,
    EVAL_SEED_OFFSET,
    EVAL_THREADS,
    PBMC_BOOTSTRAP_SEEDS,
    PCA_COMPONENTS,
    cluster_metrics,
    marker_metrics,
    score_unit,
    write_per_run,
)

N_UNITS = len(PBMC_BOOTSTRAP_SEEDS)
BASELINE_SELECTIONS = REFERENCE_DIR / "pbmc" / "baseline_selections.csv"
BASELINE_RECORDS = PKG / "experiments" / "paired" / "records" / "pbmc_baselines_per_run.csv"
BASELINE_FIELDS = ["bootstrap_id", "seed", "method", "ARI", "NMI", "MarkerHits", "CoverageScore"]
METRICS = ("ARI", "NMI", "MarkerHits", "CoverageScore")


def rescore_units(src: Path, dst: Path, X, labels, marker_groups, all_markers, eval_threads: int) -> dict:
    """Rescore the 20 units of ``src`` into ``dst``; returns ``{NNN: {check: bool}}``."""
    results = {}
    for unit in range(N_UNITS):
        tag = f"{unit:03d}"
        src_json = src / f"{tag}.json"
        source_bytes = src_json.read_bytes()
        source = json.loads(source_bytes)
        if int(source["unit"]) != unit or int(source["seed"]) != PBMC_BOOTSTRAP_SEEDS[unit]:
            raise RuntimeError(f"{src_json}: unit/seed {source['unit']}/{source['seed']} do not belong to unit {unit}")
        W = np.load(src / f"{tag}_W.npy")
        Wall = np.load(src / f"{tag}_Wall.npy")
        Xb, yb = bootstrap_resample(X, labels, PBMC_BOOTSTRAP_SEEDS[unit])
        rec = score_unit(unit, W, Xb, yb, marker_groups, all_markers, eval_threads)
        rec["meta"] = source["meta"]
        record_bytes = json.dumps(rec).encode("utf-8")
        (dst / f"{tag}.json").write_bytes(record_bytes)
        for name in ("W", "Wall"):
            shutil.copyfile(src / f"{tag}_{name}.npy", dst / f"{tag}_{name}.npy")
        checks = {k: rec[k] == source[k] for k in CHECK_KEYS}
        checks["W_is_restart_mean_of_Wall"] = same_array(W, restart_mean(Wall))
        checks["record_bytes_identical"] = record_bytes == source_bytes
        results[tag] = checks
        print(f"pbmc unit {unit:2d}: {'IDENTICAL' if all(checks.values()) else 'DIFFERENT'} "
              f"ARI={rec['ARI']:.4f} NMI={rec['NMI']:.4f} MarkerHits={rec['MarkerHits']} "
              f"CoverageScore={rec['CoverageScore']:.3f}"
              + ("" if all(checks.values()) else f"  {[k for k, v in checks.items() if not v]}"), flush=True)
    return results


def rescore_baselines(X, labels, marker_groups, all_markers, eval_threads: int, out_csv: Path) -> dict:
    """Recompute the five baselines' metrics from their stored selections and compare with the records."""
    with BASELINE_SELECTIONS.open(encoding="utf-8", newline="") as h:
        selections = {(int(r["bootstrap_id"]), r["method"]): r for r in csv.DictReader(h)}
    with BASELINE_RECORDS.open(encoding="utf-8", newline="") as h:
        records = list(csv.DictReader(h))
    resamples: dict[int, tuple] = {}
    rows, mismatches = [], []
    for r in records:
        b, method, seed = int(r["bootstrap_id"]), r["method"], int(r["seed"])
        s = selections[(b, method)]
        if seed != PBMC_BOOTSTRAP_SEEDS[b] or int(s["seed"]) != seed:
            raise RuntimeError(f"baseline {method} bootstrap {b}: seed {seed} / {s['seed']} "
                               f"!= {PBMC_BOOTSTRAP_SEEDS[b]}")
        sel = [int(i) for i in s["selected_indices"].split(";")]
        if len(sel) != int(s["selection_size"]) or len(set(sel)) != len(sel):
            raise RuntimeError(f"baseline {method} bootstrap {b}: malformed selection")
        if b not in resamples:
            resamples[b] = bootstrap_resample(X, labels, seed)
        Xb, yb = resamples[b]
        with blas_threads(eval_threads):
            ari, nmi = cluster_metrics(Xb, yb, sel, seed + EVAL_SEED_OFFSET, PCA_COMPONENTS)
        hits, cov = marker_metrics(sel, marker_groups, all_markers)
        ok = (ari == float(r["ARI"]) and nmi == float(r["NMI"]) and hits == int(r["MarkerHits"])
              and cov == float(r["CoverageScore"]))
        if not ok:
            mismatches.append(f"{method}/{b}")
        rows.append({"bootstrap_id": b, "seed": seed, "method": method, "ARI": repr(ari), "NMI": repr(nmi),
                     "MarkerHits": hits, "CoverageScore": repr(cov)})
    # same layout as the shipped record file (shortest round-trip floats, LF line ends)
    with out_csv.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=BASELINE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    n_ok = len(records) - len(mismatches)
    same_file = out_csv.read_bytes() == BASELINE_RECORDS.read_bytes()
    print(f"pbmc baselines: {n_ok}/{len(records)} rows (ARI, NMI, MarkerHits, CoverageScore) identical to the "
          f"shipped records" + (f"; different: {', '.join(mismatches[:10])}" if mismatches else "")
          + f"; recomputed CSV {'byte-identical to' if same_file else 'DIFFERS from'} {BASELINE_RECORDS.name}",
          flush=True)
    return {"rows": len(records), "identical": n_ok, "different": mismatches, "csv_bytes_identical": same_file}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units-dir", default=str(REFERENCE_DIR / "pbmc" / "units"),
                    help="directory with the 20 unit records NNN.json, NNN_W.npy, NNN_Wall.npy "
                         "(default: the shipped records reference/pbmc/units)")
    ap.add_argument("--out", required=True, help="output directory (the records are written to OUT/units)")
    ap.add_argument("--data-dir", default=None, help="data directory (default: <package>/data; PBMC is shipped)")
    ap.add_argument("--with-baselines", action="store_true",
                    help="also recompute the five baselines' metrics from reference/pbmc/baseline_selections.csv")
    ap.add_argument("--eval-threads", type=int, default=EVAL_THREADS,
                    help="BLAS/OpenMP threads for PCA + k-means (default 1, as run.py)")
    a = ap.parse_args(argv)

    src = resolve_path(a.units_dir, Path())
    out = resolve_path(a.out, Path())
    dst = out / "units"
    if dst.resolve() == src.resolve():
        raise SystemExit("--out/units must differ from --units-dir (the source records would be overwritten)")
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    X, labels, genes = load_pbmc(resolve_path(a.data_dir, DEFAULT_DATA_DIR))
    marker_groups, all_markers = pbmc_marker_indices(genes)

    units = rescore_units(src, dst, X, labels, marker_groups, all_markers, a.eval_threads)
    rows = write_per_run(dst, out / "per_run.csv")
    means = {k: float(np.mean([float(r[k]) for r in rows])) for k in METRICS}
    expected = load_json(REFERENCE_DIR / "pbmc" / "expected_ssrk_means.json")["means"]
    checks = {
        "units_identical": all(all(c.values()) for c in units.values()),
        "means_equal_paired_records": all(means[k] == expected[k] for k in expected),
    }
    report = {"units_dir": display_path(src), "units": units, "means": means, "expected_means": expected}
    if a.with_baselines:
        report["baselines"] = rescore_baselines(X, labels, marker_groups, all_markers, a.eval_threads,
                                                out / "baselines_per_run.csv")
        checks["baselines_identical"] = not report["baselines"]["different"]
        checks["baselines_csv_bytes_identical"] = report["baselines"]["csv_bytes_identical"]
    report["checks"] = checks
    save_json(out / "rescore.json", report)
    write_manifest(out, {"protocol": "pbmc_matched_rescore", "units_dir": display_path(src),
                         "eval_threads": a.eval_threads, "with_baselines": a.with_baselines, "checks": checks,
                         "total_seconds": time.perf_counter() - t0}, name="rescore_manifest.json")
    n_ok = sum(all(c.values()) for c in units.values())
    print(f"pbmc rescore: {n_ok}/{N_UNITS} units identical (selections, ARI, NMI, MarkerHits, CoverageScore, "
          f"record bytes); means " + ", ".join(f"{k}={means[k]:.4f}" for k in METRICS)
          + f" {'equal' if checks['means_equal_paired_records'] else 'DIFFER from'} the paired records")
    print(json.dumps(checks, indent=1))
    if not all(checks.values()):
        print("PBMC RESCORE CHECK FAILED (see rescore.json)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
