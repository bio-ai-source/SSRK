#!/usr/bin/env python3
"""Check reproduced results against the paper.

Two levels:

``python verify.py --from-reference``   Level A (CPU only, no torch, under 2 minutes).
    Recomputes every reported number covered by this package from the records of the original runs shipped in
    ``reference/``: from the per-run statistics W, the knockoff+ selections, FDP/power and bootstrap intervals of
    the exact Gaussian study and the PBMC and UCI HAR selections and metrics; from the stored per-dataset
    selections, the tuning summary and selection (the tuning W were not stored); the paired statistics.  Every
    rebuilt record is compared with the shipped one, and the regenerated tables and in-text numbers are compared
    with the paper text.

``python verify.py``                    Level B (after ``run_all``).
    Compares the outputs of a full retraining in ``results/`` with ``reference/`` bit for bit (raw statistic
    arrays, selections, per-run metrics) and regenerates the paper tables from ``results/``.  Areas that have
    not been run are reported as NOT RUN.

Options: ``--areas exact_gaussian,pbmc,uci_har,paired`` restricts the checks; ``--require-all`` turns NOT RUN
into a failure; ``--with-uci-evaluator`` (Level A only) additionally re-runs the five-seed MLP evaluator on the
shipped UCI HAR selections of SSRK (20) and of the baselines (100) and compares it with the shipped records (needs
the UCI HAR data and roughly 30 minutes with 4 workers).

A JSON report is written to ``results/verification_report.json`` (Level B) or
``results/verification_report_from_reference.json`` (Level A).  The exit status is 0 only if no check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

AREAS = ("exact_gaussian", "pbmc", "uci_har", "paired")
REF = PKG / "reference"
RES = PKG / "results"
LEVEL_A = RES / "verify_from_reference"
PY = sys.executable

PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT RUN"


class Report:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, area: str, name: str, status: str, detail: str = "") -> None:
        self.rows.append({"area": area, "check": name, "status": status, "detail": detail})
        print(f"  [{status:7s}] {area:14s} {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def run_script(args: list[str], log_name: str, log_dir: Path) -> tuple[int, str]:
    """Run a package script with the current interpreter; return (exit code, captured output)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([PY, *args], cwd=PKG, capture_output=True, text=True, env=env)
    text = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr.strip() else "")
    (log_dir / f"{log_name}.log").write_text(text, encoding="utf-8")
    return proc.returncode, text


def last_lines(text: str, n: int = 1) -> str:
    lines = [ln.strip() for ln in text.split("[stderr]")[0].strip().splitlines() if ln.strip()]
    return " | ".join(lines[-n:])


def first_line_with(text: str, needle: str) -> str:
    for ln in text.splitlines():
        if needle in ln:
            return ln.strip()
    return last_lines(text)


def json_checks_detail(text: str) -> str:
    """Summarize a printed JSON block of boolean checks (``"name": true/false``)."""
    keys = re.findall(r'"([a-z0-9_]+)": (true|false)', text)
    bad = [k for k, v in keys if v == "false"]
    return f"{len(keys) - len(bad)}/{len(keys)} checks true" + (f"; false: {', '.join(bad)}" if bad else "")


def status(code: int) -> str:
    return PASS if code == 0 else FAIL


# ----------------------------------------------------------------------------------- exact Gaussian study
EG_TAGS = ("eval_main", "eval_low", "eval_all", "ablation_R1", "ablation_R5", "ablation_nosharpen",
           "ablation_noentropy", "ablation_noctx", "tuning", "swap_audit")


def check_exact_gaussian(rep: Report, level_a: bool, logs: Path) -> None:
    area = "exact_gaussian"
    if level_a:
        records = LEVEL_A / area / "records"
        code, text = run_script(["experiments/exact_gaussian/run.py", "rescore", "--src", "reference/exact_gaussian",
                                 "--dst", str(records)], "exact_gaussian_rescore", logs)
        rep.add(area, "records rebuilt from the shipped W and tuning/audit rows (knockoff+, FDP/power, intervals, "
                "merge, tuning selection)",
                status(code), first_line_with(text, "rescore:"))
        if code != 0:
            return
        code, text = run_script(["experiments/exact_gaussian/make_tables.py", "--results", str(records), "--check",
                                 "--compare-timings", "--out", str(LEVEL_A / area / "tables")],
                                "exact_gaussian_tables", logs)
        rep.add(area, "Tables 1, A4, A5, A6 byte-identical to the paper; 80 numbers of the Abstract, Sec. 5.2, App. A.4 "
                "and table captions",
                status(code), first_line_with(text, "numbers:"))
        code, text = run_script(["experiments/exact_gaussian/make_figure.py", "--results", str(records), "--check",
                                 "--no-plot", "--out", str(LEVEL_A / area / "figure")], "exact_gaussian_figure", logs)
        rep.add(area, "Figure A1 panel data", status(code), first_line_with(text, "panel data"))
        return

    from experiments.exact_gaussian.run import compare_dirs

    root = RES / area
    have_all = True
    for tag in EG_TAGS:
        run_dir, ref_dir = root / tag, REF / area / tag
        if not run_dir.exists():
            rep.add(area, f"{tag}: bitwise vs reference", NOT_RUN)
            have_all = False
            continue
        try:
            res = compare_dirs(run_dir, ref_dir)
        except Exception as exc:  # interrupted or failed run: missing or truncated outputs
            rep.add(area, f"{tag}: incomplete run directory", FAIL, f"{type(exc).__name__}: {exc}")
            have_all = False
            continue
        bad = [k for k, v in res["checks"].items() if not v]
        rep.add(area, f"{tag}: bitwise vs reference ({len(res['checks'])} items)",
                PASS if res["all_identical"] else FAIL, ", ".join(bad[:6]))
    if not have_all:
        rep.add(area, "Tables 1, A4, A5, A6 from results/", NOT_RUN, "needs main, ablation, tune and audit")
        return
    code, text = run_script(["experiments/exact_gaussian/make_tables.py", "--results", str(root), "--check",
                             "--out", str(logs / "exact_gaussian_tables")], "exact_gaussian_tables", logs)
    rep.add(area, "Tables 1, A4, A5, A6 byte-identical to the paper; 78 numbers (the 2 compute times excepted)",
            status(code), first_line_with(text, "numbers:"))
    code, text = run_script(["experiments/exact_gaussian/make_figure.py", "--results", str(root), "--check",
                             "--no-plot", "--out", str(logs / "exact_gaussian_figure")], "exact_gaussian_figure", logs)
    rep.add(area, "Figure A1 panel data", status(code), first_line_with(text, "panel data"))


# -------------------------------------------------------------------------------------------- PBMC 3k
def pbmc_rescore(rep: Report, logs: Path) -> bool:
    out = LEVEL_A / "pbmc_rescore"
    code, text = run_script(["experiments/pbmc/rescore.py", "--out", str(out), "--with-baselines"],
                            "pbmc_rescore", logs)
    rep.add("pbmc", "20 bootstrap refits rescored from the shipped W (selections, ARI/NMI, markers); "
            "baseline rows from their shipped selections", status(code), first_line_with(text, "pbmc rescore:"))
    return code == 0


def check_pbmc(rep: Report, level_a: bool, logs: Path) -> None:
    area = "pbmc"
    if level_a:
        if not pbmc_rescore(rep, logs):
            return
        code, text = run_script(["experiments/pbmc/stability.py", "--units-dir", str(LEVEL_A / "pbmc_rescore" / "units"),
                                 "--strict", "--out", str(LEVEL_A / area)], "pbmc_stability", logs)
        rep.add(area, "Table A8 identical to the paper rows; 5 stability claims of Sec. 5.3 and App. A.5", status(code),
                json_checks_detail(text))
        return
    from experiments.common import compare_unit_with_reference
    from experiments.pbmc.run import CHECK_KEYS

    units_dir, ref_dir = RES / area / "units", REF / area / "units"
    present = [u for u in range(20) if (units_dir / f"{u:03d}.json").exists()]
    if not present:
        rep.add(area, "20 bootstrap refits: bitwise vs reference", NOT_RUN)
        rep.add(area, "Table A8 from results/", NOT_RUN)
        return
    bad = []
    for u in present:
        try:
            res = compare_unit_with_reference(units_dir, ref_dir, u, CHECK_KEYS)
        except Exception as exc:
            bad.append(f"unit {u}: {type(exc).__name__}")
            continue
        if not all(res.values()):
            bad.append(f"unit {u}: " + ",".join(k for k, v in res.items() if not v))
    st = FAIL if bad else (PASS if len(present) == 20 else NOT_RUN)
    rep.add(area, f"bootstrap refits: W, Wall, selections, ARI/NMI/markers bitwise ({len(present)}/20 units)",
            st, "; ".join(bad[:5]))
    if len(present) == 20:
        code, text = run_script(["experiments/pbmc/stability.py", "--units-dir", str(units_dir), "--strict",
                                 "--out", str(logs / "pbmc_stability")], "pbmc_stability", logs)
        rep.add(area, "Table A8 identical to the paper rows; 5 stability claims of Sec. 5.3 and App. A.5", status(code),
                json_checks_detail(text))
    else:
        rep.add(area, "Table A8 from results/", NOT_RUN, "needs all 20 units")


# ----------------------------------------------------------------------------------------- UCI HAR
def uci_rescore(rep: Report, logs: Path, with_evaluator: bool) -> bool:
    out = LEVEL_A / "uci_har_rescore"
    args = ["experiments/uci_har/rescore.py", "--out", str(out), "--with-baselines"]
    if with_evaluator:
        args += ["--with-evaluator", "--workers", "4"]
    code, text = run_script(args, "uci_har_rescore", logs)
    what = ("20 training seeds rescored from the shipped W (restart mean, selections, HNI, domain imbalance)"
            + ("; five-seed MLP evaluator re-run" if with_evaluator else ""))
    rep.add("uci_har", what, status(code), json_checks_detail(text))
    return code == 0


def check_uci_har(rep: Report, level_a: bool, logs: Path, with_evaluator: bool) -> None:
    area = "uci_har"
    if level_a:
        uci_rescore(rep, logs, with_evaluator)
        return
    from experiments.common import compare_unit_with_reference, load_json
    from experiments.uci_har.run import CHECK_KEYS

    units_dir, ref_dir = RES / area / "units", REF / area / "units"
    present = [u for u in range(20) if (units_dir / f"{u:03d}.json").exists()]
    if not present:
        rep.add(area, "20 training seeds: bitwise vs reference", NOT_RUN)
        rep.add(area, "five-seed evaluator vs reference", NOT_RUN)
        return
    bad = []
    for u in present:
        try:
            res = compare_unit_with_reference(units_dir, ref_dir, u, CHECK_KEYS)
        except Exception as exc:
            bad.append(f"unit {u}: {type(exc).__name__}")
            continue
        if not all(res.values()):
            bad.append(f"unit {u}: " + ",".join(k for k, v in res.items() if not v))
    st = FAIL if bad else (PASS if len(present) == 20 else NOT_RUN)
    rep.add(area, f"training seeds: W, Wall (3 restarts), selections, HNI, domain imbalance bitwise "
            f"({len(present)}/20)", st, "; ".join(bad[:5]))
    eval_path = RES / area / "eval5.json"
    if not eval_path.exists():
        rep.add(area, "five-seed evaluator vs reference", NOT_RUN)
        return
    ev, ref_eval5 = load_json(eval_path), load_json(REF / area / "eval5.json")
    diff = [k for k in ev if ev[k] != ref_eval5.get(k)]
    complete = all(f"{u:03d}_SSRK" in ev for u in range(20))
    rep.add(area, f"five-seed evaluator records identical ({len(ev)}/20 units)",
            FAIL if diff else (PASS if complete else NOT_RUN), ", ".join(diff[:5]))


# ---------------------------------------------------------------------------------------- paired statistics
def check_paired(rep: Report, level_a: bool, logs: Path, with_evaluator: bool) -> None:
    area = "paired"
    if level_a:
        pb, uc = LEVEL_A / "pbmc_rescore", LEVEL_A / "uci_har_rescore"
        if not (pb / "units" / "019.json").exists() and not pbmc_rescore(rep, logs):
            return
        if not (uc / "eval5.json").exists() and not uci_rescore(rep, logs, with_evaluator):
            return
        out = LEVEL_A / area
        args = ["experiments/paired/paired_stats.py", "--pbmc-units", str(pb / "units"), "--uci-units",
                str(uc / "units"), "--uci-eval5", str(uc / "eval5.json"), "--out", str(out)]
    else:
        needed = [RES / "pbmc" / "units" / "019.json", RES / "uci_har" / "units" / "019.json",
                  RES / "uci_har" / "eval5.json"]
        if not all(p.exists() for p in needed):
            rep.add(area, "Table A3 from results/", NOT_RUN, "needs the PBMC and UCI HAR runs")
            return
        out = RES / area
        args = ["experiments/paired/paired_stats.py", "--out", str(out)]
    code, text = run_script(args, "paired_stats", logs)
    if code != 0:
        rep.add(area, "paired statistics", FAIL, last_lines(text, 2))
        return
    same = (out / "paired_statistics.csv").read_bytes() == (REF / area / "paired_v3_final.csv").read_bytes()
    rep.add(area, "70 contrasts (gap, BCa interval, sign-randomization p, Holm) identical to the original record",
            PASS if same else FAIL)
    code, text = run_script(["experiments/paired/make_table.py", "--csv", str(out / "paired_statistics.csv"),
                             "--out", str(out)], "paired_table", logs)
    rep.add(area, "Table A3 byte-identical; 52/4/14 decisions; 16 claims (Abstract, Sec. 5.1, 5.3-5.5, App. A.3.2)",
            status(code), last_lines(text, 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-reference", action="store_true", help="Level A: recompute from the shipped records")
    ap.add_argument("--areas", default=",".join(AREAS), help="comma-separated subset of " + ",".join(AREAS) + " (default: all)")
    ap.add_argument("--require-all", action="store_true", help="treat NOT RUN as a failure")
    ap.add_argument("--with-uci-evaluator", action="store_true",
                    help="Level A: also re-run the five-seed UCI HAR evaluator on the shipped selections")
    a = ap.parse_args()
    areas = [x.strip() for x in a.areas.split(",") if x.strip()]
    unknown = sorted(set(areas) - set(AREAS))
    if unknown:
        raise SystemExit(f"unknown areas {unknown}; choose from {AREAS}")
    level = "A (from shipped records)" if a.from_reference else "B (full retraining in results/)"
    logs = LEVEL_A / "logs" if a.from_reference else RES / "verify_logs"
    print(f"SSRK reproduction check, level {level}")
    rep = Report()
    t0 = time.perf_counter()
    if "exact_gaussian" in areas:
        check_exact_gaussian(rep, a.from_reference, logs)
    if "pbmc" in areas:
        check_pbmc(rep, a.from_reference, logs)
    if "uci_har" in areas:
        check_uci_har(rep, a.from_reference, logs, a.with_uci_evaluator)
    if "paired" in areas:
        check_paired(rep, a.from_reference, logs, a.with_uci_evaluator)
    n = {s: sum(r["status"] == s for r in rep.rows) for s in (PASS, FAIL, NOT_RUN)}
    print(f"\n{n[PASS]} passed, {n[FAIL]} failed, {n[NOT_RUN]} not run ({time.perf_counter() - t0:.0f} s); "
          f"logs in {logs.relative_to(PKG)}")
    RES.mkdir(exist_ok=True)
    name = "verification_report_from_reference.json" if a.from_reference else "verification_report.json"
    (RES / name).write_text(json.dumps({"level": level, "counts": n, "checks": rep.rows}, indent=1), encoding="utf-8")
    failed = n[FAIL] > 0 or (a.require_all and n[NOT_RUN] > 0)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
