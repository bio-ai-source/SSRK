"""Matched paired statistics of SSRK against the five baselines (paper Table A3, LaTeX label ``tab:paired_summary``).

Protocol (paper, Sec. 5.1 and App. A.3.3 "Matched comparison"):

* 70 contrasts = 14 dataset--metric families x 5 baselines (Variance, STG, CAE, GAEFS, SSFS); every contrast
  pairs the 20 matched units of SSRK and the baseline (PBMC 3k: bootstrap refits ``bootstrap_id`` 0..19;
  MNIST, Fashion-MNIST, UCI HAR: training seeds ``31000 + 997 i``).
* For each contrast: mean of the paired differences (SSRK minus baseline), a 95% BCa bootstrap interval of the
  mean from 10,000 resamples (:func:`bca_ci`, seed ``71000 + 100 * metric_index + baseline_index``), and the
  two-sided paired sign-randomization p value ``(b + 1) / (10^5 + 1)`` from 100,000 random sign flips
  (:func:`paired_randomization_p`, seed ``81000 + 100 * metric_index + baseline_index``).
* Holm adjustment (:func:`holm_adjust`) over the five baselines within each dataset--metric family; decision
  ``SSRK`` / ``baseline`` if the Holm-adjusted p value is below 0.05 (direction from the sign of the mean
  difference; lower is better for HNI and DomainBalance), ``ns`` otherwise.

Inputs
------
* SSRK per-unit records produced by this package (``experiments/pbmc/run.py`` -> ``results/pbmc/units/NNN.json``;
  ``experiments/uci_har/run.py`` + ``evaluate.py`` -> ``results/uci_har/units/NNN.json`` and the five-seed MLP
  evaluator file ``results/uci_har/eval5.json``), or the reference copies of the original records
  (``--ssrk-source reference``).  UCI HAR accuracy and macro-F1 of SSRK are taken from the five-seed evaluator.
* Fixed records shipped in ``experiments/paired/records/`` (see ``README_records.md``): baseline per-run
  metrics for PBMC 3k and UCI HAR, and all six methods for MNIST/Fashion-MNIST (not retrained here).

Outputs (``--out``, default ``results/paired``): ``paired_statistics.csv`` (one row per contrast, same columns and
row order as the original ``paired_v3_final.csv``), ``paired_means.json`` (per-method means over the 20 units),
``ssrk_per_unit.csv`` (the SSRK values entering every contrast), ``decision_counts.json`` and ``manifest.json``.

This is a faithful port of the paired-statistics code of the original study (``bca_ci``,
``paired_randomization_p``, ``holm_adjust``): same arithmetic, same random streams, same order.  It needs only
numpy and scipy (no torch).

usage: python experiments/paired/paired_stats.py [--ssrk-source results|reference]
           [--pbmc-units DIR] [--uci-units DIR] [--uci-eval5 FILE_OR_DIR [...]] [--out DIR] [--allow-partial]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PKG = Path(__file__).resolve().parents[2]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np  # noqa: E402

RECORDS = Path(__file__).resolve().parent / "records"

#: Method order of the original study (SSRK first); fixes the baseline index ``bi`` of the seeds.
METHODS = ("SSRK", "Variance", "STG", "CAE", "GAEFS", "SSFS")
BASELINES = METHODS[1:]
#: Evaluation seeds of the matched units: PBMC bootstrap refits and the image / UCI HAR training seeds.
PBMC_BOOTSTRAP_SEEDS = tuple(20_000 + 997 * i for i in range(20))
REPEAT_SEEDS = tuple(31_000 + 997 * i for i in range(20))
N_UNITS = 20
LOWER_IS_BETTER = {"HNI", "DomainBalance"}
ALPHA = 0.05
N_BOOTSTRAP = 10_000
N_SIGN_FLIPS = 100_000
BCA_SEED_BASE = 71_000
SIGN_SEED_BASE = 81_000

#: (dataset, metrics in family order, shipped record file, unit key).  The metric index ``mi`` (position in the
#: list) and the baseline index ``bi`` (position in BASELINES) define the seeds of every contrast.
FAMILIES = (
    ("pbmc", ("ARI", "NMI", "MarkerHits", "CoverageScore"), "pbmc_baselines_per_run.csv", "bootstrap_id"),
    ("mnist", ("ARI", "NMI", "CoverageEff"), "image_per_run.csv", "seed"),
    ("fashion_mnist", ("ARI", "NMI", "CoverageEff"), "image_per_run.csv", "seed"),
    ("uci_har", ("Accuracy", "F1_macro", "HNI", "DomainBalance"), "uci_har_baselines_per_run.csv", "seed"),
)
CSV_FIELDS = ("dataset", "metric", "contrast", "n_pairs", "ssrk_mean", "baseline_mean", "mean_difference",
              "bca_ci_low", "bca_ci_high", "paired_randomization_p", "holm_adjusted_p", "decision")


# ----------------------------------------------------------------------------------------------- statistics
def bca_ci(values: Sequence[float], seed: int) -> tuple[float, float]:
    """95% BCa bootstrap interval of the mean paired difference (Efron, 1987), 10,000 resamples.

    Uses ``scipy.stats.bootstrap(method="BCa")`` with ``np.random.default_rng(seed)``.  A constant difference
    vector has no sampling variability; its interval is the degenerate ``[d, d]`` (e.g. UCI HAR HNI vs SSFS).
    """
    from scipy.stats import bootstrap

    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2 or np.allclose(arr, arr[0]):
        value = float(arr[0]) if arr.size else math.nan
        return value, value
    result = bootstrap(
        (arr,),
        np.mean,
        confidence_level=0.95,
        n_resamples=N_BOOTSTRAP,
        method="BCa",
        random_state=np.random.default_rng(seed),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def paired_randomization_p(values: Sequence[float], seed: int, n_draws: int = N_SIGN_FLIPS) -> float:
    """Two-sided paired sign-randomization p value ``(b + 1) / (n_draws + 1)``.

    ``b`` counts the random sign vectors ``s`` (i.i.d. uniform on {-1, +1}, drawn from
    ``np.random.default_rng(seed)`` in chunks of 10,000 rows) whose statistic ``|mean(s * d)|`` is at least the
    observed ``|mean(d)|`` (tolerance 1e-15).  An all-zero difference vector returns 1.
    """
    arr = np.asarray(values, dtype=np.float64)
    observed = abs(float(np.mean(arr)))
    if arr.size == 0 or np.allclose(arr, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = 10_000
    completed = 0
    while completed < n_draws:
        current = min(chunk, n_draws - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(current, arr.size))
        permuted = np.abs(np.mean(signs * arr[None, :], axis=1))
        extreme += int(np.sum(permuted >= observed - 1e-15))
        completed += current
    return float((extreme + 1) / (n_draws + 1))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm (1979) step-down adjusted p values: ``p_(i) <- max_{j <= i} min(1, (m - j + 1) p_(j))``.

    Ties keep their input order (``np.argsort`` default), which does not change the adjusted values.
    """
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def decide(mean_difference: float, holm_p: float, metric: str) -> str:
    """Decision of one contrast: ``SSRK`` / ``baseline`` if Holm p < 0.05 (by direction), else ``ns``."""
    if not holm_p < ALPHA:
        return "ns"
    favours_ssrk = mean_difference < 0 if metric in LOWER_IS_BETTER else mean_difference > 0
    return "SSRK" if favours_ssrk else "baseline"


# ----------------------------------------------------------------------------------------------- inputs
def load_records(csv_name: str, dataset: str, unit_key: str) -> dict[str, dict[int, dict[str, str]]]:
    """Shipped per-run records -> ``{method: {unit: row}}`` (cells kept as strings, parsed with ``float``)."""
    path = RECORDS / csv_name
    if not path.exists():
        raise FileNotFoundError(f"shipped record file missing: {path}")
    with path.open(encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    if "dataset" in rows[0]:
        rows = [r for r in rows if r["dataset"] == dataset]
    by_method: dict[str, dict[int, dict[str, str]]] = {m: {} for m in METHODS}
    for r in rows:
        unit = int(r[unit_key])
        if unit in by_method[r["method"]]:
            raise RuntimeError(f"{path}: duplicate record {dataset}/{r['method']}/{unit}")
        if unit_key == "seed" and int(r["seed"]) not in REPEAT_SEEDS:
            raise RuntimeError(f"{path}: seed {r['seed']} is not an evaluation seed")
        if unit_key == "bootstrap_id" and int(r["seed"]) != PBMC_BOOTSTRAP_SEEDS[unit]:
            raise RuntimeError(f"{path}: bootstrap {unit} has seed {r['seed']}")
        if "unit" in r and REPEAT_SEEDS[int(r["unit"])] != int(r["seed"]):
            raise RuntimeError(f"{path}: unit {r['unit']} has seed {r['seed']}")
        by_method[r["method"]][unit] = r
    return by_method


def load_eval5(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Merge five-seed evaluator records ``{"NNN_SSRK": {unit, method, Accuracy, F1_macro, ...}}``.

    Each path is a JSON file or a directory of JSON files.  A key present twice with different content raises.
    """
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(p.glob("*.json"))
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(f"five-seed evaluator record not found: {p}")
    merged: dict[str, dict[str, Any]] = {}
    for f in files:
        for k, v in json.loads(f.read_text(encoding="utf-8")).items():
            if k in merged and merged[k] != v:
                raise RuntimeError(f"{f}: conflicting duplicate five-seed record {k}")
            merged[k] = v
    return merged


def load_ssrk_units(folder: Path, dataset: str, metrics: Sequence[str],
                    eval5: dict[str, dict[str, Any]] | None) -> tuple[dict[int, dict[str, Any]], list[Path]]:
    """SSRK per-unit records ``folder/NNN.json`` -> ``{unit key: {metric: value}}``.

    Guard of the original code: every record's ``seed`` must equal the evaluation seed of its ``unit``
    (PBMC: bootstrap seed; UCI HAR: training seed).  PBMC records are keyed by ``unit`` (= bootstrap_id),
    UCI HAR records by ``seed``.  For UCI HAR, Accuracy and F1_macro are replaced by the five-seed evaluator
    values of ``eval5[f"{unit:03d}_SSRK"]``.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"SSRK unit directory not found: {folder}")
    files = sorted(folder.glob("[0-9][0-9][0-9].json"))
    new: dict[int, dict[str, Any]] = {}
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        expected = (PBMC_BOOTSTRAP_SEEDS if dataset == "pbmc" else REPEAT_SEEDS)[rec["unit"]]
        if int(rec["seed"]) != int(expected):
            raise RuntimeError(f"{f}: seed {rec['seed']} is not the evaluation seed {expected} of unit {rec['unit']}")
        key = rec["unit"] if dataset == "pbmc" else rec["seed"]
        new[int(key)] = {k: rec[k] for k in metrics}
        if dataset == "uci_har":
            key5 = f"{rec['unit']:03d}_SSRK"
            if key5 not in eval5:
                raise KeyError(f"five-seed evaluator record {key5} (for {f}) not found in --uci-eval5 inputs")
            e = eval5[key5]
            if int(e["unit"]) != int(rec["unit"]) or e.get("method", "SSRK") != "SSRK":
                raise RuntimeError(f"five-seed record {rec['unit']:03d}_SSRK does not belong to {f}")
            new[int(key)]["Accuracy"], new[int(key)]["F1_macro"] = e["Accuracy"], e["F1_macro"]
    return new, files


# ----------------------------------------------------------------------------------------------- main
def compute(sources: dict[str, Path], eval5_paths: Sequence[Path], allow_partial: bool = False):
    """Recompute all 70 contrasts.  Returns ``(rows, means, ssrk_units, inputs)``."""
    eval5 = load_eval5(eval5_paths)
    out_rows: list[dict[str, Any]] = []
    means: dict[str, dict[str, dict[str, float]]] = {}
    ssrk_units: dict[str, dict[int, dict[str, Any]]] = {}
    inputs: list[Path] = [RECORDS / f for f in sorted({fam[2] for fam in FAMILIES})]
    for dataset, metrics, csv_name, unit_key in FAMILIES:
        by_method = load_records(csv_name, dataset, unit_key)
        if dataset in sources:  # SSRK refitted by this package (PBMC, UCI HAR)
            by_method["SSRK"], files = load_ssrk_units(sources[dataset], dataset, metrics, eval5)
            inputs += files
        ssrk = by_method["SSRK"]
        if not allow_partial:
            for m in METHODS:
                if len(by_method[m]) != N_UNITS:
                    raise RuntimeError(f"{dataset}/{m}: {len(by_method[m])} units found, {N_UNITS} required "
                                       f"(use --allow-partial for a partial check)")
        ssrk_units[dataset] = ssrk
        means[dataset] = {m: {met: float(np.mean([float(v[met]) for v in by_method[m].values()])) for met in metrics}
                          for m in METHODS if by_method[m]}
        for mi, metric in enumerate(metrics):
            group = []
            for bi, base in enumerate(BASELINES):
                shared = sorted(set(ssrk) & set(by_method[base]))
                diffs = [float(ssrk[u][metric]) - float(by_method[base][u][metric]) for u in shared]
                lo, hi = bca_ci(diffs, BCA_SEED_BASE + mi * 100 + bi)
                group.append(dict(dataset=dataset, metric=metric, contrast=f"SSRK - {base}", n_pairs=len(shared),
                                  ssrk_mean=float(np.mean([float(ssrk[u][metric]) for u in shared])),
                                  baseline_mean=float(np.mean([float(by_method[base][u][metric]) for u in shared])),
                                  mean_difference=float(np.mean(diffs)), bca_ci_low=lo, bca_ci_high=hi,
                                  paired_randomization_p=paired_randomization_p(diffs, SIGN_SEED_BASE + mi * 100 + bi)))
            adj = holm_adjust([g["paired_randomization_p"] for g in group])
            for g, p in zip(group, adj):
                g["holm_adjusted_p"] = p
                g["decision"] = decide(g["mean_difference"], p, metric)
                out_rows.append(g)
    return out_rows, means, ssrk_units, inputs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict[str, str]:
    import scipy

    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform()}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PKG).as_posix()
    except ValueError:
        return str(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paired statistics of SSRK vs the five baselines (Table A3).")
    ap.add_argument("--ssrk-source", choices=("results", "reference"), default="results",
                    help="default SSRK inputs: this package's results/ (default) or the reference/ copies of the "
                         "original records")
    ap.add_argument("--pbmc-units", type=Path, help="directory of PBMC SSRK unit records NNN.json")
    ap.add_argument("--uci-units", type=Path, help="directory of UCI HAR SSRK unit records NNN.json")
    ap.add_argument("--uci-eval5", type=Path, nargs="+",
                    help="five-seed evaluator records of the SSRK units (JSON files or directories)")
    ap.add_argument("--out", type=Path, default=PKG / "results" / "paired", help="output directory (default results/paired)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="do not require all 20 units per method (debugging only; not the paper's table)")
    a = ap.parse_args(argv)
    if a.ssrk_source == "reference":
        defaults = (PKG / "reference" / "pbmc" / "units", PKG / "reference" / "uci_har" / "units",
                    PKG / "reference" / "uci_har" / "eval5.json")
    else:
        defaults = (PKG / "results" / "pbmc" / "units", PKG / "results" / "uci_har" / "units",
                    PKG / "results" / "uci_har" / "eval5.json")
    a.pbmc_units = a.pbmc_units or defaults[0]
    a.uci_units = a.uci_units or defaults[1]
    a.uci_eval5 = a.uci_eval5 or [defaults[2]]
    return a


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    t0 = time.perf_counter()
    rows, means, ssrk_units, inputs = compute({"pbmc": a.pbmc_units, "uci_har": a.uci_units}, a.uci_eval5,
                                              a.allow_partial)
    seconds = time.perf_counter() - t0
    a.out.mkdir(parents=True, exist_ok=True)
    # csv's default "\r\n" terminator and str(float) cells, as the original writer: byte-identical output
    with (a.out / "paired_statistics.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        w.writerows(rows)
    (a.out / "paired_means.json").write_text(json.dumps(means, indent=1), encoding="utf-8")
    with (a.out / "ssrk_per_unit.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h, lineterminator="\n")
        w.writerow(["dataset", "unit_key", "metric", "value"])
        for (dataset, metrics, _, _) in FAMILIES:
            for u in sorted(ssrk_units[dataset]):
                for met in metrics:
                    w.writerow([dataset, u, met, repr(float(ssrk_units[dataset][u][met]))])
    counts = Counter(r["decision"] for r in rows)
    summary = {"contrasts": len(rows), "SSRK": counts["SSRK"], "baseline": counts["baseline"], "ns": counts["ns"]}
    (a.out / "decision_counts.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    eval5_files = [f for p in a.uci_eval5 for f in (sorted(p.glob("*.json")) if p.is_dir() else [p])]
    manifest = {
        "command": [Path(sys.executable).name, *sys.argv],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ssrk_source": a.ssrk_source,
        "pbmc_units": rel(a.pbmc_units), "uci_units": rel(a.uci_units), "uci_eval5": [rel(p) for p in a.uci_eval5],
        "protocol": {"methods": METHODS, "n_units": N_UNITS, "bca_resamples": N_BOOTSTRAP,
                     "sign_flips": N_SIGN_FLIPS, "bca_seed": "71000 + 100*metric_index + baseline_index",
                     "sign_seed": "81000 + 100*metric_index + baseline_index", "holm_family": "dataset x metric",
                     "alpha": ALPHA},
        "inputs_sha256": {rel(p): sha256(p) for p in [*inputs, *eval5_files]},
        "decision_counts": summary,
        "seconds": seconds,
        "environment": environment(),
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(rows)} contrasts: SSRK {counts['SSRK']}, baseline {counts['baseline']}, ns {counts['ns']}")
    for r in rows:
        if r["decision"] != "SSRK":
            print(f"  {r['dataset']:13s} {r['metric']:13s} {r['contrast']:16s} {r['mean_difference']:+.4f} "
                  f"[{r['bca_ci_low']:+.4f},{r['bca_ci_high']:+.4f}] p_holm={r['holm_adjusted_p']:.3g} {r['decision']}")
    print(f"wrote {rel(a.out)}/paired_statistics.csv ({seconds:.1f} s)")


if __name__ == "__main__":
    main()
