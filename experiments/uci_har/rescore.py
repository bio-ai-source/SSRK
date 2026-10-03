#!/usr/bin/env python3
"""Rescore the UCI HAR matched protocol from the shipped statistics W (Level A; CPU only, no torch).

For each of the 20 units this script recomputes, from ``NNN_W.npy`` / ``NNN_Wall.npy`` alone, what
``experiments/uci_har/run.py`` derives after the fit (the same function,
:func:`experiments.uci_har.run.rank_unit`):

* ``W`` = coordinatewise float64 mean of the 3 restarts in ``Wall``, stored as float32 (checked bitwise);
* ``selected_indices`` = top 384 of the 561 real features (the evaluated ranking) and
  ``calibration_topk_indices`` = top 384 of all 1,122 columns (deterministic ranking);
* HNI = share of the 561 appended permuted columns (indices 561..1121) in the all-column top 384;
* DomainBalance = ``|n_time/272 - n_freq/289|`` of the real top 384 (paper App. A.8). The feature names
  come from ``data/uci_har/UCI HAR Dataset/features.txt`` or, if the dataset has not been downloaded,
  from its byte copy ``reference/uci_har/features.txt`` (SHA-256 checked against ``data/checksums.json``).

It writes ``--out/units/NNN.json`` in the format of ``run.py`` (single-seed Accuracy / F1_macro and
``meta`` copied from the source record) plus byte copies of ``NNN_W.npy`` / ``NNN_Wall.npy``, and
``--out/eval5.json``, the five-seed MLP evaluator records that enter Table A3 (Sec. 5.5): a copy of
``--eval5`` (default ``reference/uci_har/eval5.json``) after consistency checks, or, with
``--with-evaluator``, recomputed by ``experiments/uci_har/evaluate.py``'s evaluator (needs the UCI HAR
download; about 8 min with ``--workers 4``). ``experiments/paired/paired_stats.py --uci-units
--out/units --uci-eval5 --out/eval5.json`` then recomputes the UCI HAR rows of Table A3.

Checks (all must hold, otherwise the exit status is 1):

* every recomputed field equals the source record exactly and the written ``NNN.json`` is
  byte-identical to the source file; ``W`` is the restart mean of ``Wall`` bitwise;
* every five-seed record belongs to its unit, its seed-0 values equal the unit's single-seed Accuracy /
  F1_macro, and its Accuracy / F1_macro equal the mean of the five per-seed values; with
  ``--with-evaluator`` the recomputed ``eval5.json`` equals the source records exactly (also as bytes);
* the 20-unit means equal the SSRK means of the paired-statistics record
  (``reference/uci_har/expected_ssrk_means.json``: accuracy .9489, macro-F1 .9485, HNI .0000, domain
  imbalance .1339);
* ``--with-baselines``: the DomainBalance of the five baselines, recomputed from their stored selections
  (``reference/uci_har/baseline_selections.csv``, the 384 real features of each run), equals the shipped
  records ``experiments/paired/records/uci_har_baselines_per_run.csv`` in all 100 rows; with
  ``--with-evaluator`` also their five-seed Accuracy / F1_macro (about 30 min more with ``--workers 4``).
  Baseline HNI cannot be re-derived (it needs the 1,122-column ranking, which was not stored) and stays a
  record.

Usage (from the package root)::

    python experiments/uci_har/rescore.py --out results/verify_from_reference/uci_har_rescore
    python experiments/uci_har/rescore.py --out results/uci_har_rescore --with-baselines
    python experiments/uci_har/rescore.py --out results/uci_har_rescore --with-evaluator --workers 4

Outputs in ``--out``: ``units/NNN.json``, ``units/NNN_W.npy``, ``units/NNN_Wall.npy``, ``eval5.json``,
``baselines_per_run.csv`` (with ``--with-baselines``), ``eval5_baselines.json`` (with both options),
``rescore.json`` (all checks) and ``rescore_manifest.json``.
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
    display_path,
    load_json,
    resolve_path,
    restart_mean,
    same_array,
    save_json,
    sha256_file,
    write_csv,
    write_manifest,
)
from experiments.uci_har.data import (  # noqa: E402
    SHIPPED_FEATURES,
    appended_columns,
    features_file,
    read_feature_names,
    uci_domain_balance,
)
from experiments.uci_har.evaluate import REPEAT_SEEDS, run_jobs  # noqa: E402
from experiments.uci_har.run import (  # noqa: E402
    APPENDED_NULL_RATIO,
    CHECK_KEYS,
    N_REAL,
    RESTARTS,
    UCI_K,
    rank_unit,
    unit_record,
)

N_UNITS = len(REPEAT_SEEDS)
BASELINE_SELECTIONS = REFERENCE_DIR / "uci_har" / "baseline_selections.csv"
BASELINE_RECORDS = PKG / "experiments" / "paired" / "records" / "uci_har_baselines_per_run.csv"
BASELINE_FIELDS = ["seed", "unit", "method", "Accuracy", "F1_macro", "HNI", "DomainBalance"]
FEATURES_KEY = "uci_har/UCI HAR Dataset/features.txt"


def feature_names(data_dir: Path) -> tuple[list[str], dict]:
    """The 561 feature names (dataset copy if downloaded, else the shipped copy) and their provenance."""
    path = features_file(data_dir, allow_shipped_copy=True)
    expected = load_json(PKG / "data" / "checksums.json")["datasets"]["uci_har"]["files"][FEATURES_KEY]["sha256"]
    info = {"path": display_path(path), "shipped_copy": path.resolve() == SHIPPED_FEATURES.resolve(),
            "sha256": sha256_file(path), "sha256_expected": expected}
    info["sha256_ok"] = info["sha256"] == expected
    if not info["shipped_copy"]:
        info["equals_shipped_copy"] = path.read_bytes() == SHIPPED_FEATURES.read_bytes()
    return read_feature_names(path), info


def rescore_units(src: Path, dst: Path, names, eval5: dict | None) -> tuple[dict, dict]:
    """Rescore the 20 units of ``src`` into ``dst``.

    ``eval5`` (recomputed five-seed records) replaces the single-seed Accuracy / F1_macro of the source
    records by the recomputed seed-0 values; without it they are copied. Returns ``(checks, records)``.
    """
    nulls = appended_columns(N_REAL, APPENDED_NULL_RATIO)
    results, records = {}, {}
    for unit in range(N_UNITS):
        tag = f"{unit:03d}"
        src_json = src / f"{tag}.json"
        source_bytes = src_json.read_bytes()
        source = json.loads(source_bytes)
        seed = REPEAT_SEEDS[unit]
        if int(source["unit"]) != unit or int(source["seed"]) != seed:
            raise RuntimeError(f"{src_json}: unit/seed {source['unit']}/{source['seed']} do not belong to unit {unit}")
        W = np.load(src / f"{tag}_W.npy")
        Wall = np.load(src / f"{tag}_Wall.npy")
        if W.shape != (N_REAL + len(nulls),) or Wall.shape != (RESTARTS, W.size):
            raise RuntimeError(f"{src}: unit {unit} has W {W.shape} / Wall {Wall.shape}, expected "
                               f"({N_REAL + len(nulls)},) / ({RESTARTS}, {N_REAL + len(nulls)})")
        ranked = rank_unit(W, names, nulls)
        if eval5 is None:
            acc, f1 = source["Accuracy"], source["F1_macro"]
        else:
            acc, f1 = eval5[f"{tag}_SSRK"]["accs"][0], eval5[f"{tag}_SSRK"]["f1s"][0]
        rec = unit_record(unit, seed, acc, f1, ranked)
        rec["meta"] = source["meta"]
        record_bytes = json.dumps(rec).encode("utf-8")
        (dst / f"{tag}.json").write_bytes(record_bytes)
        for name in ("W", "Wall"):
            shutil.copyfile(src / f"{tag}_{name}.npy", dst / f"{tag}_{name}.npy")
        checks = {k: rec[k] == source[k] for k in CHECK_KEYS}
        checks["W_is_restart_mean_of_Wall"] = same_array(W, restart_mean(Wall))
        checks["record_bytes_identical"] = record_bytes == source_bytes
        results[tag], records[unit] = checks, rec
        print(f"uci_har unit {unit:2d}: {'IDENTICAL' if all(checks.values()) else 'DIFFERENT'} "
              f"HNI={rec['HNI']:.4f} DomainBalance={rec['DomainBalance']:.4f}"
              + ("" if all(checks.values()) else f"  {[k for k, v in checks.items() if not v]}"), flush=True)
    return results, records


def eval5_consistency(eval5: dict, records: dict) -> dict:
    """Each five-seed record belongs to its unit, starts with the unit's single-seed values, and its
    reported Accuracy / F1_macro are the means of the five per-seed values."""
    out = {}
    for unit, rec in records.items():
        key = f"{unit:03d}_SSRK"
        e = eval5.get(key)
        out[key] = bool(
            e is not None and int(e["unit"]) == unit and e["method"] == "SSRK"
            and len(e["accs"]) == 5 and len(e["f1s"]) == 5
            and e["accs"][0] == e["Accuracy_seed0"] == rec["Accuracy"] and e["f1s"][0] == rec["F1_macro"]
            and e["Accuracy"] == sum(e["accs"]) / 5 and e["F1_macro"] == sum(e["f1s"]) / 5
        )
    return out


def load_baselines() -> tuple[list[dict], dict]:
    """Shipped baseline records and stored selections ``{(unit, method): [384 real-feature indices]}``."""
    with BASELINE_RECORDS.open(encoding="utf-8", newline="") as h:
        records = list(csv.DictReader(h))
    with BASELINE_SELECTIONS.open(encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    selections = {}
    for r in rows:
        unit, seed = int(r["unit"]), int(r["seed"])
        sel = [int(i) for i in r["selected_indices"].split(";")]
        if (seed != REPEAT_SEEDS[unit] or int(r["selection_size"]) != UCI_K or len(sel) != UCI_K
                or len(set(sel)) != UCI_K or not all(0 <= i < N_REAL for i in sel)):
            raise RuntimeError(f"{BASELINE_SELECTIONS}: malformed row unit {unit} method {r['method']}")
        selections[(unit, r["method"])] = sel
    for r in records:
        if (int(r["unit"]), r["method"]) not in selections or int(r["seed"]) != REPEAT_SEEDS[int(r["unit"])]:
            raise RuntimeError(f"{BASELINE_RECORDS}: no stored selection for unit {r['unit']} method {r['method']}")
    return records, selections


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units-dir", default=str(REFERENCE_DIR / "uci_har" / "units"),
                    help="directory with the 20 unit records NNN.json, NNN_W.npy, NNN_Wall.npy "
                         "(default: the shipped records reference/uci_har/units)")
    ap.add_argument("--eval5", default=str(REFERENCE_DIR / "uci_har" / "eval5.json"),
                    help="five-seed evaluator records of these units (default: reference/uci_har/eval5.json)")
    ap.add_argument("--out", required=True, help="output directory (the records are written to OUT/units)")
    ap.add_argument("--data-dir", default=None, help="data directory (default: <package>/data)")
    ap.add_argument("--with-evaluator", action="store_true",
                    help="recompute the five-seed MLP evaluator (needs the UCI HAR download)")
    ap.add_argument("--workers", type=int, default=1, help="evaluator processes (each single-thread BLAS)")
    ap.add_argument("--with-baselines", action="store_true",
                    help="also rescore the five baselines' stored selections (DomainBalance; with "
                         "--with-evaluator also the five-seed Accuracy / F1_macro)")
    a = ap.parse_args(argv)

    src = resolve_path(a.units_dir, Path())
    out = resolve_path(a.out, Path())
    dst = out / "units"
    if dst.resolve() == src.resolve():
        raise SystemExit("--out/units must differ from --units-dir (the source records would be overwritten)")
    dst.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_path(a.data_dir, DEFAULT_DATA_DIR)
    eval5_src = resolve_path(a.eval5, Path())
    t0 = time.perf_counter()
    names, features = feature_names(data_dir)
    eval5_source = load_json(eval5_src)
    checks = {"feature_names_checksum_ok": features["sha256_ok"]}
    if "equals_shipped_copy" in features:
        checks["feature_names_equal_shipped_copy"] = features["equals_shipped_copy"]

    baseline_records, selections = load_baselines() if a.with_baselines else ([], {})
    eval5_new, eval5_baselines, seconds = None, {}, {}
    if a.with_evaluator:
        jobs = []
        for unit in range(N_UNITS):
            rec = load_json(src / f"{unit:03d}.json")
            jobs.append((f"{unit:03d}_SSRK", unit, REPEAT_SEEDS[unit], rec["selected_indices"], "SSRK"))
        for r in baseline_records:
            unit = int(r["unit"])
            jobs.append((f"{unit:03d}_{r['method']}", unit, REPEAT_SEEDS[unit], selections[(unit, r["method"])],
                         r["method"]))
        print(f"five-seed MLP evaluator: {len(jobs)} selections on {max(1, a.workers)} worker(s)", flush=True)
        evaluated, seconds = run_jobs(data_dir, jobs, a.workers)
        eval5_new = {k: v for k, v in evaluated.items() if k.endswith("_SSRK")}
        eval5_baselines = {k: v for k, v in evaluated.items() if not k.endswith("_SSRK")}

    units, records = rescore_units(src, dst, names, eval5_new)
    checks["units_identical"] = all(all(c.values()) for c in units.values())
    if eval5_new is None:
        shutil.copyfile(eval5_src, out / "eval5.json")
        eval5 = eval5_source
    else:
        save_json(out / "eval5.json", eval5_new, indent=None)
        eval5 = eval5_new
        checks["eval5_recomputed_equals_source"] = eval5_new == eval5_source
        checks["eval5_bytes_identical"] = (out / "eval5.json").read_bytes() == eval5_src.read_bytes()
    consistency = eval5_consistency(eval5, records)
    checks["eval5_consistent_with_units"] = all(consistency.values())
    means = {"Accuracy": float(np.mean([eval5[f"{u:03d}_SSRK"]["Accuracy"] for u in range(N_UNITS)])),
             "F1_macro": float(np.mean([eval5[f"{u:03d}_SSRK"]["F1_macro"] for u in range(N_UNITS)])),
             "HNI": float(np.mean([records[u]["HNI"] for u in range(N_UNITS)])),
             "DomainBalance": float(np.mean([records[u]["DomainBalance"] for u in range(N_UNITS)]))}
    expected = load_json(REFERENCE_DIR / "uci_har" / "expected_ssrk_means.json")["means"]
    checks["means_equal_paired_records"] = all(means[k] == expected[k] for k in expected)
    report = {"units_dir": display_path(src), "eval5_source": display_path(eval5_src),
              "evaluator_recomputed": a.with_evaluator, "feature_names": features, "units": units,
              "eval5_consistency": consistency, "means": means, "expected_means": expected}

    if a.with_baselines:
        rows, db_diff, ev_diff = [], [], []
        for r in baseline_records:
            unit, method = int(r["unit"]), r["method"]
            db = uci_domain_balance(selections[(unit, method)], names)
            if db != float(r["DomainBalance"]):
                db_diff.append(f"{method}/{unit}")
            row = {"seed": r["seed"], "unit": unit, "method": method, "Accuracy": r["Accuracy"],
                   "F1_macro": r["F1_macro"], "HNI": r["HNI"], "DomainBalance": repr(db)}
            if a.with_evaluator:
                e = eval5_baselines[f"{unit:03d}_{method}"]
                if e["Accuracy"] != float(r["Accuracy"]) or e["F1_macro"] != float(r["F1_macro"]):
                    ev_diff.append(f"{method}/{unit}")
                row["Accuracy"], row["F1_macro"] = repr(e["Accuracy"]), repr(e["F1_macro"])
            rows.append(row)
        write_csv(out / "baselines_per_run.csv", rows, BASELINE_FIELDS)
        n = len(baseline_records)
        report["baselines"] = {"rows": n, "DomainBalance_identical": n - len(db_diff),
                               "DomainBalance_different": db_diff,
                               "HNI": "record (not recomputable: needs the 1,122-column ranking)",
                               "Accuracy_F1": "recomputed (five-seed evaluator)" if a.with_evaluator else "record"}
        checks["baselines_domain_balance_identical"] = not db_diff
        print(f"uci_har baselines: DomainBalance {n - len(db_diff)}/{n} rows identical to the shipped records",
              flush=True)
        if a.with_evaluator:
            save_json(out / "eval5_baselines.json", eval5_baselines, indent=None)
            report["baselines"]["Accuracy_F1_identical"] = n - len(ev_diff)
            report["baselines"]["Accuracy_F1_different"] = ev_diff
            checks["baselines_accuracy_f1_identical"] = not ev_diff
            print(f"uci_har baselines: five-seed Accuracy/F1_macro {n - len(ev_diff)}/{n} rows identical to the "
                  "shipped records", flush=True)

    report["checks"] = checks
    save_json(out / "rescore.json", report)
    write_manifest(out, {"protocol": "uci_har_matched_rescore", "units_dir": display_path(src),
                         "with_evaluator": a.with_evaluator, "workers": a.workers,
                         "with_baselines": a.with_baselines, "feature_names": features, "checks": checks,
                         "unit_seconds": seconds, "total_seconds": time.perf_counter() - t0},
                   name="rescore_manifest.json")
    n_ok = sum(all(c.values()) for c in units.values())
    print(f"uci_har rescore: {n_ok}/{N_UNITS} units identical (W = restart mean, both top-384 selections, HNI, "
          f"DomainBalance, record bytes); five-seed records {'recomputed' if a.with_evaluator else 'copied'}; "
          "means " + ", ".join(f"{k}={means[k]:.4f}" for k in means)
          + f" {'equal' if checks['means_equal_paired_records'] else 'DIFFER from'} the paired records")
    print(json.dumps(checks, indent=1))
    if not all(checks.values()):
        print("UCI HAR RESCORE CHECK FAILED (see rescore.json)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
