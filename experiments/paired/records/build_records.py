"""Provenance tool: how the shipped baseline/image records were extracted from the original research archive.

Not needed for reproduction.  The three CSV files it writes are shipped in this directory, and
``paired_stats.py`` reads those shipped files; nothing in the reproduction pipeline (``run_all``, ``verify.py``,
the tests) calls this script.  It runs only with the original research archive (``--orig``), which is not part of
this package.  It documents, and lets an auditor with access to that archive re-derive byte for byte, the exact
transformation from the original per-run records to the shipped ones (source paths relative to the archive
root):

* ``pbmc_baselines_per_run.csv``  <- ``revised/results/local_minimal_v2_exact/01_pbmc_exact_top100_per_run.csv``
  rows with ``status == "ok"`` and ``method != "SSRK"``; columns bootstrap_id, seed, method, ARI, NMI,
  MarkerHits, CoverageScore (cells copied verbatim, i.e. the original shortest round-trip float strings).
* ``uci_har_baselines_per_run.csv`` <- ``revised/results/local_minimal_v2_exact/03_uci_exact_per_run.csv``
  rows with ``status == "ok"`` and ``method != "SSRK"``; HNI and DomainBalance copied verbatim; Accuracy and
  F1_macro REPLACED by the deterministic five-seed MLP evaluator applied to each stored selection
  (``revised/cr2/results/uci_eval5/stored_*.json``, key ``f"{unit:03d}_{method}"`` with
  ``unit = REPEAT_SEEDS.index(seed)``), written with ``repr(float)`` (exact round trip).  This is the
  ``--uci-eval eval5`` branch of the original ``revised/cr2/scripts/paired_v3.py``.
* ``image_per_run.csv`` <- ``revised/results/local_minimal_v2_exact/02_image_exact_per_run.csv``
  rows with ``status == "ok"`` for all six methods (SSRK included: the image runs are not retrained by
  this package); columns dataset, seed, method, ARI, NMI, CoverageEff (verbatim).

Row order is the order of the original files.

usage (auditors with the original archive only):
    python experiments/paired/records/build_records.py --orig <ORIGINAL_ARCHIVE_ROOT> [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

METHODS = ("SSRK", "Variance", "STG", "CAE", "GAEFS", "SSFS")
REPEAT_SEEDS = tuple(31_000 + 997 * i for i in range(20))

PBMC_FIELDS = ["bootstrap_id", "seed", "method", "ARI", "NMI", "MarkerHits", "CoverageScore"]
UCI_FIELDS = ["seed", "unit", "method", "Accuracy", "F1_macro", "HNI", "DomainBalance"]
IMAGE_FIELDS = ["dataset", "seed", "method", "ARI", "NMI", "CoverageEff"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ok(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    bad = [r for r in rows if r.get("status") != "ok"]
    if bad:
        raise RuntimeError(f"{path}: {len(bad)} rows with status != ok")
    return rows


def write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({f: r[f] for f in fields})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orig", required=True, type=Path,
                    help="root of the original research archive (not part of this package)")
    ap.add_argument("--out", type=Path, default=HERE, help="output directory (default experiments/paired/records)")
    a = ap.parse_args()
    stored = a.orig / "revised" / "results" / "local_minimal_v2_exact"
    eval5_dir = a.orig / "revised" / "cr2" / "results" / "uci_eval5"
    a.out.mkdir(parents=True, exist_ok=True)
    sources = {}

    src = stored / "01_pbmc_exact_top100_per_run.csv"
    rows = [r for r in read_ok(src) if r["method"] != "SSRK"]
    write(a.out / "pbmc_baselines_per_run.csv", rows, PBMC_FIELDS)
    sources["pbmc_baselines_per_run.csv"] = {src.name: sha256(src)}

    eval5: dict[str, dict] = {}
    files = sorted(eval5_dir.glob("stored_*.json"))
    for f in files:
        for k, v in json.loads(f.read_text()).items():
            if k in eval5 and eval5[k] != v:
                raise RuntimeError(f"conflicting duplicate key {k} in {f}")
            eval5[k] = v
    src = stored / "03_uci_exact_per_run.csv"
    rows = [r for r in read_ok(src) if r["method"] != "SSRK"]
    for r in rows:
        unit = REPEAT_SEEDS.index(int(r["seed"]))
        e = eval5[f"{unit:03d}_{r['method']}"]
        assert e["unit"] == unit and e["method"] == r["method"], e
        r["unit"] = str(unit)
        r["Accuracy"], r["F1_macro"] = repr(float(e["Accuracy"])), repr(float(e["F1_macro"]))
    write(a.out / "uci_har_baselines_per_run.csv", rows, UCI_FIELDS)
    sources["uci_har_baselines_per_run.csv"] = {src.name: sha256(src), **{f.name: sha256(f) for f in files}}

    src = stored / "02_image_exact_per_run.csv"
    rows = read_ok(src)
    write(a.out / "image_per_run.csv", rows, IMAGE_FIELDS)
    sources["image_per_run.csv"] = {src.name: sha256(src)}

    for name, srcs in sources.items():
        print(f"{name}  sha256={sha256(a.out / name)}")
        for s, h in srcs.items():
            print(f"    source {s}  sha256={h}")


if __name__ == "__main__":
    main()
