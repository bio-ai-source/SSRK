#!/usr/bin/env python3
"""Tables and in-text numbers of the exact Gaussian study from the run records.

Writes, into ``--out`` (default ``<results>/tables``):

* ``gaussian_main.tex``      Table 1 (Sec. 5.2; LaTeX label tab:gauss_main)
* ``gaussian_full.tex``      Table A5 (bootstrap CIs, sizes, empty rates; App. A.4)
* ``gaussian_tuning.tex``    Table A4 (predeclared tuning grid; App. A.4)
* ``gaussian_ablation.tex``  Table A6 (component ablation; App. A.4)
* ``numbers.json``         every number of the exact Gaussian study printed in the paper text
                           (Abstract, Sec. 5.2, App. A.4 "Exact Gaussian fixed-target study",
                           table captions), recomputed from the records, with the printed
                           string and a verbatim excerpt of the sentence in which it appears

(The file names are those of the paper's ``tables/`` directory.)

Inputs: ``eval_all``, ``tuning``, ``ablation_*`` and ``swap_audit`` under
``--results`` (default ``results/exact_gaussian``; ``--from-reference`` uses the
shipped records in ``reference/exact_gaussian``; ``run.py rescore --dst DIR`` writes a
complete record set rebuilt from the shipped W, usable as ``--results DIR``).
``--check`` compares the four tables byte for byte with ``reference/paper/tables``
and every number with its printed string (exit status 1 on any difference);
``--paper-tex`` additionally confirms that every excerpt occurs verbatim in the
paper source.

Numbers are read with ``pandas.read_csv`` (as for the published tables) and
formatted to the printed precision; a number "matches" when its formatted value
equals the printed string.  The bound "No cell mean exceeds its level by more than
0.011" is checked as the inequality itself; numbers.json stores the unrounded largest
excess and its cell next to it.  The outputs of this script do
not depend on pandas' float parser.  (The default C parser is not round-trip, e.g.
'0.14285714285714285' -> 0.1428571428571428; the byte identity of the stored
eval_all and tuning summaries depends on it, see ``run.merge_evals``.)
The two compute times of App. A.4 are wall-clock measurements of the original
runs: they are compared with ``--from-reference`` or ``--compare-timings`` (records
that carry the original manifests, e.g. a ``run.py rescore`` output), and reported
but not compared for a rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.exact_gaussian.protocol import (  # noqa: E402
    ABLATIONS, PROTOCOL, TUNING_AMPLITUDES, TUNING_GRID, TUNING_NULL_SEEDS, TUNING_SIGNAL_SEEDS,
)
from ssrk.ft_config import FixedTargetConfig  # noqa: E402  (imports no torch)
from experiments.common import display_path  # noqa: E402

RESULTS = PKG / "results" / "exact_gaussian"
REFERENCE = PKG / "reference" / "exact_gaussian"
PAPER_TABLES = PKG / "reference" / "paper" / "tables"
TABLE_FILES = ["gaussian_main.tex", "gaussian_full.tex", "gaussian_tuning.tex", "gaussian_ablation.tex"]

# Rows of the ablation table: (run directory, label).  The first row is the selected
# configuration evaluated in eval_all (its swap check counts all 70 checked datasets).
ABLATION_ROWS = [
    ("eval_all", "Selected configuration ($R=20$)"),
    ("ablation_R5", "$R=5$ restarts"),
    ("ablation_R1", "Single fit ($R=1$)"),
    ("ablation_nosharpen", "No Stage-II sharpening ($\\lambda_{\\rm II}=0$)"),
    ("ablation_noentropy", "No entropy term ($\\lambda_{\\rm I}=\\lambda_{\\rm II}=0$)"),
    ("ablation_noctx", "No visible-target context"),
]


# ----------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------

def fmt(x: float, nd: int = 3) -> str:
    """Table format: fixed decimals without the leading zero (``.093``, ``1.000``)."""
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("0.") else ("-" + s[2:] if s.startswith("-0.") else s)


def txt(x: float, nd: int = 3) -> str:
    """Text format: fixed decimals with the leading zero (``0.167``)."""
    return f"{x:.{nd}f}"


def sci_tex(x: float) -> str:
    """``0.02 -> $2\\times10^{-2}$`` (mantissa printed as an integer when exact)."""
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    ms = str(int(round(m))) if abs(m - round(m)) < 1e-9 else f"{m:g}"
    return f"${ms}\\times10^{{{e}}}$"


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------

def load_records(root: Path) -> dict:
    """Read every record needed for the tables and numbers."""
    ev = root / "eval_all"
    rec = {
        "summary": pd.read_csv(ev / "summary.csv"),
        "rows": pd.read_csv(ev / "per_seed_q.csv"),
        "manifest": json.loads((ev / "manifest.json").read_text(encoding="utf-8")),
        "infos": json.loads((ev / "per_seed_info.json").read_text(encoding="utf-8")),
        "tuning": pd.read_csv(root / "tuning" / "tuning_summary.csv"),
        "selection": json.loads((root / "tuning" / "selection.json").read_text(encoding="utf-8")),
        "audit": pd.read_csv(root / "swap_audit" / "summary.csv"),
        "ablation": {},
    }
    for tag, _ in ABLATION_ROWS:
        d = root / tag
        rec["ablation"][tag] = {
            "summary": pd.read_csv(d / "summary.csv"),
            "manifest": json.loads((d / "manifest.json").read_text(encoding="utf-8")),
        }
    return rec


def _cell(summary: pd.DataFrame, a: float, q: float, col: str):
    sig = summary[summary.model == "signal"]
    return sig[(np.isclose(sig.signal, a)) & (np.isclose(sig.q, q))].iloc[0][col]


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------

def main_table(summary: pd.DataFrame) -> str:
    """Table 1: mean FDP / mean power at three levels, |S| at q=0.10, empty rate at q=0.05."""
    sig = summary[summary.model == "signal"]
    null = summary[summary.model == "all_null"]
    levels = sorted(sig.signal.unique())
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Mean FDP / mean power} & & \\\cmidrule(lr){2-4}",
        r"$a$ & $q=0.05$ & $q=0.10$ & $q=0.15$ & $|\widehat S|$, $q=0.10$ & Empty, $q=0.05$\\\midrule",
    ]
    for a in levels:
        cells = []
        for q in (0.05, 0.10, 0.15):
            r = sig[(sig.signal == a) & (np.isclose(sig.q, q))].iloc[0]
            cells.append(f"{fmt(r.FDP_mean)} / {fmt(r.power_mean)}")
        r10 = sig[(sig.signal == a) & (np.isclose(sig.q, 0.10))].iloc[0]
        r05 = sig[(sig.signal == a) & (np.isclose(sig.q, 0.05))].iloc[0]
        lines.append(f"{a:.2f} & " + " & ".join(cells)
                     + f" & {r10.selection_size_mean:.1f} & {fmt(r05.empty_rate, 2)}" + r"\\")
    ne = []
    for q in (0.05, 0.10, 0.15):
        r = null[np.isclose(null.q, q)].iloc[0]
        ne.append(f"{int(r.nonempty_count)}/{int(r.datasets)} nonempty")
    lines.append(r"\midrule")
    lines.append("All null & " + " & ".join(ne) + r" & -- & --\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def appendix_table(summary: pd.DataFrame) -> str:
    """Table A5: means with percentile-bootstrap CIs, sizes (SD), empty rates."""
    sig = summary[summary.model == "signal"]
    null = summary[summary.model == "all_null"]
    lines = [
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Amplitude & $q$ & Mean FDP [95\% CI] & Mean power [95\% CI] & Mean $|\widehat S|$ (SD) & Empty rate\\\midrule",
    ]
    for a in sorted(sig.signal.unique()):
        for q in (0.05, 0.10, 0.15):
            r = sig[(sig.signal == a) & (np.isclose(sig.q, q))].iloc[0]
            lines.append(
                f"{a:.2f} & {q:.2f} & {fmt(r.FDP_mean)} [{fmt(r.FDP_ci_low)}, {fmt(r.FDP_ci_high)}] & "
                f"{fmt(r.power_mean)} [{fmt(r.power_ci_low)}, {fmt(r.power_ci_high)}] & "
                f"{r.selection_size_mean:.1f} ({r.selection_size_sd:.1f}) & {fmt(r.empty_rate, 2)}\\\\")
        lines.append(r"\addlinespace[2pt]")
    for q in (0.05, 0.10, 0.15):
        r = null[np.isclose(null.q, q)].iloc[0]
        lines.append(
            f"All null & {q:.2f} & {fmt(r.FDP_mean)} ({int(r.nonempty_count)}/{int(r.datasets)}; "
            f"bound {fmt(r.nonempty_upper95)}) & -- & {r.selection_size_mean:.2f} & {fmt(r.empty_rate, 2)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _variant(override: dict) -> str:
    if override.get("lambda_stage2", None) == 0.0:
        return "no sharpening"
    if override.get("use_target_context", True) is False:
        return "no target context"
    return "full"


def tuning_table(ts: pd.DataFrame, chosen: str) -> str:
    """Table A4: mean power at q=0.10 per tuning amplitude, their mean,
    and nonempty all-null selections at q=0.10 (one fit of the all-null tuning
    datasets: the run at the largest tuning amplitude)."""
    sig = ts[ts.model == "signal"]
    nul = ts[ts.model == "all_null"]
    amps = sorted(sig.tuning_amplitude.unique())
    lines = [r"\begin{tabular}{rcrl" + "c" * len(amps) + "cc}", r"\toprule",
             r"$R$ & Gate learning rate & Epochs & Variant & "
             + " & ".join(f"$a={a:.2f}$" for a in amps)
             + r" & Mean & All-null nonempty\\",
             r"\midrule"]
    for key in sorted(sig.config_key.unique()):
        override, R_grid = TUNING_GRID[key]
        cfg = FixedTargetConfig(**override)
        g = sig[sig.config_key == key]
        R = int(g.restarts.iloc[0])
        assert R == R_grid, (key, R, R_grid)
        powers = [float(g[(np.isclose(g.tuning_amplitude, a)) & (np.isclose(g.q, 0.10))].iloc[0].power_mean)
                  for a in amps]
        n = nul[(nul.config_key == key) & (np.isclose(nul.q, 0.10))
                & (np.isclose(nul.tuning_amplitude, max(amps)))].iloc[0]
        ne = f"{int(n.nonempty_count)}/{int(n.datasets)}"
        tag = f"{R}" + (r"$^\star$" if key == chosen else "")
        lines.append(f"{tag} & {sci_tex(cfg.lr_gate)} & {cfg.epochs} & {_variant(override)} & "
                     + " & ".join(fmt(x) for x in powers) + f" & {fmt(np.mean(powers))} & {ne}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def ablation_table(abl: dict) -> str:
    """Table A6: FDP and power at q=0.10 (a = 0.20, 0.30) and the swap check
    (bitwise mismatches over all checked restarts / number of checked datasets)."""
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{$a=0.20$} & \multicolumn{2}{c}{$a=0.30$} & \\\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Variant & FDP & Power & FDP & Power & Swap check\\\midrule"]
    for tag, name in ABLATION_ROWS:
        summ = abl[tag]["summary"]
        cells = []
        for a in (0.20, 0.30):
            cells += [fmt(float(_cell(summ, a, 0.10, "FDP_mean"))), fmt(float(_cell(summ, a, 0.10, "power_mean")))]
        sa = abl[tag]["manifest"]["swap_audits"]
        lines.append(f"{name} & " + " & ".join(cells)
                     + f" & {sa['total_per_run_bitwise_mismatches']} / {sa['count']}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# in-text numbers
# ----------------------------------------------------------------------------

def compute_numbers(rec: dict) -> list[dict]:
    """Every number of the exact Gaussian study in the paper text, recomputed from the records.

    Each entry: id, location, paper (printed string), computed, match, source,
    context (verbatim excerpt of the paper sentence), machine_dependent; an
    inequality claim checked at printed precision also carries its unrounded
    ``value``, the ``cell`` where it is attained and a ``note``.
    """
    S, rows, man = rec["summary"], rec["rows"], rec["manifest"]
    sig, nul = S[S.model == "signal"], S[S.model == "all_null"]
    levels = sorted(sig.signal.unique())
    cfg, R = man["config"], int(man["restarts"])
    sa = man["swap_audits"]
    aud = rec["audit"]
    abl = rec["ablation"]
    ts, sel = rec["tuning"], rec["selection"]
    out: list[dict] = []

    def add(id_, location, paper, computed, source, context, machine_dependent=False, **extra):
        out.append({"id": id_, "location": location, "paper": paper, "computed": computed,
                    "match": paper == computed, "machine_dependent": machine_dependent,
                    "source": source, "context": context, **extra})

    def pow10(a):
        return float(_cell(S, a, 0.10, "power_mean"))

    def null_ne(q):
        return int(nul[np.isclose(nul.q, q)].iloc[0].nonempty_count)

    words = {3: "three", 6: "six"}
    n_sig = sorted(set(sig.datasets.astype(int)))
    n_null = int(nul.datasets.iloc[0])
    ci_cover = all(r.FDP_ci_low <= r.q + 1e-12 for _, r in sig.iterrows())
    mism_total = sa["total_per_run_bitwise_mismatches"] + sa["total_aggregate_bitwise_mismatches"]
    amp_list = ", ".join(f"{a:.2f}" for a in levels)

    # ---------------- Abstract ----------------
    A = "Abstract"
    add("abstract.n_amplitudes", A, "six", words.get(len(levels), str(len(levels))),
        "eval_all/summary.csv: signal amplitudes", "with six signal amplitudes (50 datasets each)")
    add("abstract.datasets_per_amplitude", A, "50", "/".join(map(str, n_sig)),
        "eval_all/summary.csv: datasets per signal cell", "with six signal amplitudes (50 datasets each)")
    add("abstract.fdp_at_or_below_q", A, "true", str(ci_cover).lower(),
        "eval_all/summary.csv: every FDP_ci_low <= q",
        r"stays at or below $q\in\{0.05,0.10,0.15\}$ up to Monte Carlo error")
    add("abstract.power10_from", A, "0.167", txt(pow10(min(levels))),
        "eval_all/summary.csv: power_mean, a=0.15, q=0.10", "power at $q=0.10$ rises from 0.167 to 1.000")
    add("abstract.power10_to", A, "1.000", txt(pow10(max(levels))),
        "eval_all/summary.csv: power_mean, a=0.75, q=0.10", "power at $q=0.10$ rises from 0.167 to 1.000")
    add("abstract.null_datasets", A, "100", str(n_null), "eval_all/summary.csv: all-null datasets",
        "100 all-null datasets yield no nonempty selection at $q=0.10$")
    add("abstract.null_nonempty_q10", A, "0", str(null_ne(0.10)),
        "eval_all/summary.csv: all-null nonempty_count, q=0.10",
        "100 all-null datasets yield no nonempty selection at $q=0.10$")
    add("abstract.swap_mismatches", A, "0", str(mism_total),
        "eval_all/manifest.json: per-run + aggregate bitwise mismatches",
        "reproduces the sign-flipped statistic exactly in every checked run")

    # ---------------- Sec. 5.2 ----------------
    L = "Sec. 5.2"
    ctx_design = (r"Each dataset has $n=800$ observations, $p=80$ AR(1) Gaussian candidates ($\rho=0.3$), "
                  r"and a held-out target view of $r=12$ coordinates")
    add("sec42.n", L, "800", str(man["protocol"]["n"]), "manifest protocol n", ctx_design)
    add("sec42.p", L, "80", str(man["protocol"]["p"]), "manifest protocol p", ctx_design)
    add("sec42.rho", L, "0.3", str(man["protocol"]["rho"]), "manifest protocol rho", ctx_design)
    add("sec42.r", L, "12", str(man["protocol"]["target_dim"]), "manifest protocol target_dim", ctx_design)
    ctx_sup = (r"each of 30 nonnull candidates enters three targets with coefficient $\pm a/\sqrt3$, "
               r"and the other 50 candidates are exact conditional nulls")
    sup_sizes = sorted({len(i["support"]) for i in rec["infos"] if i["model"] == "signal"})
    add("sec42.nonnull", L, "30", "/".join(map(str, sup_sizes)), "eval_all/per_seed_info.json: |support|", ctx_sup)
    add("sec42.targets_per_signal", L, "three", words[man["protocol"]["targets_per_signal"]],
        "manifest protocol targets_per_signal", ctx_sup)
    add("sec42.nulls", L, "50", str(man["protocol"]["p"] - sup_sizes[0]), "p - |support|", ctx_sup)
    add("sec42.widths", L, "256--128--32--128--256", "--".join(map(str, cfg["hidden"])), "manifest config hidden",
        "The gate network (widths 256--128--32--128--256)")
    add("sec42.restarts", L, "20", str(R), "eval_all/manifest.json restarts",
        "$W$ averages 20 restarts on the same paired design")
    ctx_amps = (r"Each amplitude $a\in\{0.15, 0.20, 0.30, 0.40, 0.50, 0.75\}$ is evaluated on the same 50 "
                r"datasets, and 100 all-null datasets set every coefficient to zero")
    add("sec42.amplitudes", L, "0.15, 0.20, 0.30, 0.40, 0.50, 0.75", amp_list, "eval_all amplitudes", ctx_amps)
    add("sec42.null_datasets", L, "100", str(n_null), "eval_all all-null datasets", ctx_amps)
    add("sec42.fdp_ci_contains_or_below_q", L, "true", str(ci_cover).lower(),
        "eval_all/summary.csv: every FDP_ci_low <= q",
        r"In every cell, the 95\% bootstrap interval for mean FDP contains $q$ or lies below it")
    worst = sig.sort_values("FDP_mean").iloc[-1]
    add("sec42.largest_cell_mean", L, "0.160 at $a=0.20$, $q=0.15$",
        f"{txt(worst.FDP_mean)} at $a={worst.signal:.2f}$, $q={worst.q:.2f}$",
        "eval_all/summary.csv: max FDP_mean over signal cells",
        "the largest cell mean is 0.160 at $a=0.20$, $q=0.15$")
    grid = np.array([[float(_cell(S, a, q, "power_mean")) for q in (0.05, 0.10, 0.15)] for a in levels])
    monotone = bool(np.all(np.diff(grid, axis=0) >= 0) and np.all(np.diff(grid, axis=1) >= 0))
    add("sec42.power_increases_with_a_and_q", L, "true", str(monotone).lower(),
        "eval_all/summary.csv: power_mean nondecreasing in a and in q",
        "power increases with both amplitude and $q$")
    ctx_pow = (r"at $q=0.10$ it rises from 0.167 at $a=0.15$ and 0.614 at $a=0.20$ to 0.972 at $a=0.30$ "
               r"and 1.000 from $a=0.50$ on")
    for a, s in ((0.15, "0.167"), (0.20, "0.614"), (0.30, "0.972")):
        add(f"sec42.power10_a{int(round(a * 100)):02d}", L, s, txt(pow10(a)),
            f"eval_all/summary.csv: power_mean, a={a:.2f}, q=0.10", ctx_pow)
    ones = [a for a in levels if all(txt(pow10(b)) == "1.000" for b in levels if b >= a)]
    add("sec42.power10_one_from", L, "1.000 from $a=0.50$",
        f"{txt(pow10(min(ones)))} from $a={min(ones):.2f}$",
        "eval_all/summary.csv: smallest a from which power_mean(q=0.10) prints as 1.000", ctx_pow)
    add("sec42.null_nonempty", L, "0, 0, and 1", f"{null_ne(0.05)}, {null_ne(0.10)}, and {null_ne(0.15)}",
        "eval_all/summary.csv: all-null nonempty_count at q=0.05/0.10/0.15",
        "The all-null datasets give 0, 0, and 1 nonempty selections at the three levels")
    ctx_sw = r"retraining on randomly swapped subsets reproduced $T_SW$ exactly in all 70 checked datasets"
    add("sec42.swap_datasets", L, "70", str(sa["count"]), "eval_all/manifest.json swap_audits.count", ctx_sw)
    add("sec42.swap_mismatches", L, "0", str(mism_total), "eval_all/manifest.json bitwise mismatches", ctx_sw)

    # Component ablation paragraph (Sec. 5.2)
    fdp10 = {tag: [float(_cell(abl[tag]["summary"], a, 0.10, "FDP_mean")) for a in (0.20, 0.30)]
             for tag, _ in ABLATION_ROWS}
    pw = {tag: {a: float(_cell(abl[tag]["summary"], a, 0.10, "power_mean")) for a in (0.20, 0.30)}
          for tag, _ in ABLATION_ROWS}
    allf = [x for v in fdp10.values() for x in v]
    ctx_ab = (r"every variant keeps mean FDP at or below $q=0.10$ up to Monte Carlo error (0.002--0.108) "
              r"and passes the swap check")
    add("sec42.ablation_fdp_range", L, "0.002--0.108", f"{txt(min(allf))}--{txt(max(allf))}",
        "ablation_* and eval_all summary.csv: FDP_mean at q=0.10, a in {0.20, 0.30}", ctx_ab)
    abl_ci = all(float(_cell(abl[tag]["summary"], a, 0.10, "FDP_ci_low")) <= 0.10
                 for tag, _ in ABLATION_ROWS for a in (0.20, 0.30))
    add("sec42.ablation_fdp_ci_at_or_below_q", L, "true", str(abl_ci).lower(),
        "ablation_* summary.csv: FDP_ci_low <= 0.10", ctx_ab)
    abl_mm = sum(abl[tag]["manifest"]["swap_audits"]["total_per_run_bitwise_mismatches"]
                 + abl[tag]["manifest"]["swap_audits"]["total_aggregate_bitwise_mismatches"]
                 for tag, _ in ABLATION_ROWS)
    add("sec42.ablation_swap_mismatches", L, "0", str(abl_mm), "ablation_* manifest.json swap_audits", ctx_ab)
    ctx_ab2 = "at $a=0.20$, a single fit reaches 0.014, five restarts 0.312, and the 20-restart aggregate 0.614"
    add("sec42.ablation_power_a20_R1", L, "0.014", txt(pw["ablation_R1"][0.20]), "ablation_R1 power, a=0.20", ctx_ab2)
    add("sec42.ablation_power_a20_R5", L, "0.312", txt(pw["ablation_R5"][0.20]), "ablation_R5 power, a=0.20", ctx_ab2)
    add("sec42.ablation_power_a20_R20", L, "0.614", txt(pw["eval_all"][0.20]), "eval_all power, a=0.20", ctx_ab2)
    ctx_ab3 = ("removing Stage-II sharpening, the entropy schedule, or the visible-target context lowers power "
               "to 0.597, 0.538, and 0.568")
    add("sec42.ablation_power_a20_components", L, "0.597, 0.538, and 0.568",
        f"{txt(pw['ablation_nosharpen'][0.20])}, {txt(pw['ablation_noentropy'][0.20])}, and "
        f"{txt(pw['ablation_noctx'][0.20])}", "ablation_{nosharpen,noentropy,noctx} power, a=0.20", ctx_ab3)

    # ---------------- Appendix: data-generating process ----------------
    L = "App. A.4, Data-generating process"
    s_vals = sorted({i["s_value"] for i in rec["infos"]})
    add("app.s_value", L, "0.99", "/".join(f"{s:.2f}" for s in s_vals), "per_seed_info.json s_value",
        "(here $s=0.99$)")
    add("app.null_rows", L, "50", str(man["protocol"]["p"] - sup_sizes[0]), "p - |support|",
        "The null set is therefore exactly the 50 zero rows of $B$")
    add("app.null_datasets", L, "100", str(n_null), "eval_all all-null datasets",
        "The all-null model sets $B=0$ on 100 further datasets")

    # ---------------- Appendix: gate network and training ----------------
    L = "App. A.4, Gate network and training"
    ctx_b = (r"Inputs are clipped at $\pm6$ identically for both pair members, the standardized target at "
             r"$\pm6$, and outputs at $\pm10$; the affine weights and biases are constrained so that "
             r"$\|[A_l\ b_l;\,0\ 1]\|_F\le30$, with the homogeneous row fixed, and the logits are truncated to "
             r"$[-8,8]$")

    def g(x):
        return f"{x:g}"

    add("app.input_clip", L, "6", g(cfg["input_clip"]), "config input_clip", ctx_b)
    add("app.target_clip", L, "6", g(cfg["target_clip"]), "config target_clip", ctx_b)
    add("app.output_clip", L, "10", g(cfg["output_clip"]), "config output_clip", ctx_b)
    add("app.frobenius_cap", L, "30", "/".join(sorted({g(c) for c in cfg["frobenius_caps"]})),
        "config frobenius_caps", ctx_b)
    add("app.logit_cap", L, "8", g(cfg["logit_cap"]), "config logit_cap", ctx_b)
    add("app.restarts", L, "20", str(R), "eval_all restarts",
        r"The reported statistic is the mean $\overline W_j=20^{-1}\sum_{r=1}^{20}W_j^{(r)}$ over 20 restarts")

    # ---------------- Appendix: predeclared tuning ----------------
    L = "App. A.4, Predeclared tuning"
    ctx_t = ("20 signal datasets, each evaluated at the tuning amplitudes 0.20 and 0.30, and 20 all-null "
             "datasets")
    tsig = ts[ts.model == "signal"]
    tnul = ts[ts.model == "all_null"]
    add("app.tuning_signal_datasets", L, "20", "/".join(map(str, sorted(set(tsig.datasets.astype(int))))),
        "tuning_summary.csv datasets (signal)", ctx_t)
    add("app.tuning_amplitudes", L, "0.20 and 0.30",
        " and ".join(f"{a:.2f}" for a in sorted(tsig.tuning_amplitude.unique())), "tuning_summary.csv", ctx_t)
    add("app.tuning_null_datasets", L, "20", "/".join(map(str, sorted(set(tnul.datasets.astype(int))))),
        "tuning_summary.csv datasets (all-null)", ctx_t)
    chosen = sel["chosen"]
    full = _variant(TUNING_GRID[chosen][0]) == "full"
    add("app.tuning_selected", L, "full configuration at 20 restarts",
        f"{'full' if full else _variant(TUNING_GRID[chosen][0])} configuration at {sel['restarts']} restarts",
        "tuning/selection.json (selection rule applied to tuning_summary.csv)",
        "It selected the full configuration, with Stage-II sharpening and visible-target context, at 20 restarts.")
    add("app.tuning_selected_is_evaluated", L, "true",
        str(chosen == man["config_key"] and sel["override"] == {} and sel["restarts"] == R).lower(),
        "selection.json chosen == eval_all config_key, override {} and R",
        "It selected the full configuration, with Stage-II sharpening and visible-target context, at 20 restarts.")

    # ---------------- Appendix: results ----------------
    L = "App. A.4, Results"
    ex = sig.FDP_mean - sig.q
    excess = float(ex.max())
    arg = sig.loc[ex.idxmax()]
    bound = 0.011
    add("app.max_excess_over_q", L, "0.011", "0.011" if excess <= bound else txt(excess),
        "eval_all/summary.csv: max(FDP_mean - q) <= 0.011",
        "No cell mean exceeds its level by more than 0.011.",
        value=excess, cell={"a": float(arg.signal), "q": float(arg.q)}, holds_strictly=bool(excess <= bound),
        note=f"largest excess {excess:.6f} at a={arg.signal:.2f}, q={arg.q:.2f} (bound 0.011)")
    hi = sig[sig.signal >= 0.30 - 1e-12]
    f10 = hi[np.isclose(hi.q, 0.10)].FDP_mean
    f15 = hi[np.isclose(hi.q, 0.15)].FDP_mean
    ctx_r = r"(0.099--0.108 at $q=0.10$, 0.150--0.154 at $q=0.15$)"
    add("app.fdp_range_q10_a30plus", L, "0.099--0.108", f"{txt(f10.min())}--{txt(f10.max())}",
        "eval_all/summary.csv: FDP_mean, a >= 0.30, q=0.10", ctx_r)
    add("app.fdp_range_q15_a30plus", L, "0.150--0.154", f"{txt(f15.min())}--{txt(f15.max())}",
        "eval_all/summary.csv: FDP_mean, a >= 0.30, q=0.15", ctx_r)
    sr = rows[rows.model == "signal"].sort_values(["signal", "seed", "q"])
    nested = True
    for _, grp in sr.groupby(["signal", "seed"]):
        p = grp.power.to_numpy()
        sets = [set() if not isinstance(s, str) else set(map(int, s.split())) for s in grp.selected]
        nested &= bool(np.all(np.diff(p) >= 0)) and all(sets[i] <= sets[i + 1] for i in range(len(sets) - 1))
    add("app.power_nondecreasing_in_q_per_dataset", L, "true", str(nested).lower(),
        "eval_all/per_seed_q.csv: per dataset, selections nested and power nondecreasing in q",
        r"so power is nondecreasing in $q$ dataset by dataset")
    add("app.clopper_pearson_q10", L, "0.030", txt(float(nul[np.isclose(nul.q, 0.10)].iloc[0].nonempty_upper95)),
        "eval_all/summary.csv: all-null nonempty_upper95, q=0.10",
        r"the one-sided 95\% Clopper--Pearson upper bound on the FDR at $q=0.10$ is 0.030")
    empty_free = [a for a in levels if all(float(_cell(S, b, q, "empty_rate")) == 0.0
                                           for b in levels if b >= a for q in (0.05, 0.10, 0.15))]
    add("app.nonempty_from", L, "0.30", f"{min(empty_free):.2f}",
        "eval_all/summary.csv: smallest a from which every empty_rate is 0",
        "from $a=0.30$ on, every selection is nonempty")
    nonempty05 = rows[(rows.model == "signal") & np.isclose(rows.q, 0.05) & (rows.selection_size > 0)]
    min_size = int(nonempty05.selection_size.min())
    add("app.min_nonempty_size_q05", L, "20",
        str(math.ceil(1 / 0.05)) if min_size >= math.ceil(1 / 0.05) else f"violated (min {min_size})",
        "ceil(1/q) at q=0.05; checked: smallest nonempty selection at q=0.05 in per_seed_q.csv >= 20",
        r"a nonempty knockoff+ set needs at least $\lceil1/q\rceil=20$ selections")
    ctx_e = "the empty-selection rates of 0.94 and 0.76 at $a=0.15$ and $0.20$"
    add("app.empty_rate_q05_a15", L, "0.94", txt(float(_cell(S, 0.15, 0.05, "empty_rate")), 2),
        "eval_all/summary.csv: empty_rate, a=0.15, q=0.05", ctx_e)
    add("app.empty_rate_q05_a20", L, "0.76", txt(float(_cell(S, 0.20, 0.05, "empty_rate")), 2),
        "eval_all/summary.csv: empty_rate, a=0.20, q=0.05", ctx_e)

    # ---------------- Appendix: swap checks and compute ----------------
    L = "App. A.4, Swap checks and compute"
    ctx_s1 = ("For 70 evaluation datasets (every fifth signal dataset at each amplitude and every tenth all-null "
              "dataset)")
    exp_count = len(levels) * len([s for s in PROTOCOL["signal_seeds"] if s % 5 == 0]) + len(
        [s for s in PROTOCOL["all_null_seeds"] if s % 10 == 0])
    add("app.swap_datasets", L, "70", str(sa["count"]), "eval_all/manifest.json swap_audits.count", ctx_s1)
    add("app.swap_datasets_rule", L, "70", str(exp_count), "every 5th signal seed x amplitudes + every 10th null",
        ctx_s1)
    sizes = sa["subset_sizes"]
    add("app.swap_subset_sizes_within", L, "40",
        str(PROTOCOL["p"] // 2) if 1 <= min(sizes) and max(sizes) <= PROTOCOL["p"] // 2 else "violated",
        "audit subset size law (p/2); checked: all stored subset sizes in {1,...,40}",
        r"($|S|$ uniform on $\{1,\ldots,40\}$, coordinates uniform)")
    ctx_s2 = ("Over these 1,400 gate fits, the retrained statistics agreed with $T_SW$ exactly, both for single "
              "restarts and for the aggregate.")
    add("app.swap_fits", L, "1,400", f"{sa['count'] * R:,}", "swap_audits.count x restarts", ctx_s2)
    add("app.swap_per_run_mismatches", L, "0", str(sa["total_per_run_bitwise_mismatches"]),
        "eval_all/manifest.json total_per_run_bitwise_mismatches", ctx_s2)
    add("app.swap_aggregate_mismatches", L, "0", str(sa["total_aggregate_bitwise_mismatches"]),
        "eval_all/manifest.json total_aggregate_bitwise_mismatches", ctx_s2)
    ctx_s3 = ("On 100 further paired training runs, each with a fresh dataset of the design at $a=0.75$, a swap "
              r"subset of uniformly drawn size in $\{1,\ldots,80\}$, and a single fit, the centered form of "
              r"Remark~\ref{rem:centered} also reproduced $T_SW$ exactly, in both single and double precision.")
    sym = aud[aud.parameterization == "symmetric"]
    add("app.trajectories", L, "100", "/".join(map(str, sorted(set(aud.trajectories.astype(int))))),
        "swap_audit/summary.csv trajectories", ctx_s3)
    for dt in ("float32", "float64"):
        r = sym[sym.dtype == dt].iloc[0]
        add(f"app.trajectories_exact_{dt}", L, "100", str(int(r.exact_trajectories)),
            f"swap_audit/summary.csv symmetric {dt} exact_trajectories", ctx_s3)
    ctx_s4 = (r"no norm projection occurred, the largest value of $\|[A_l\ b_l;\,0\ 1]\|_F$ was 23.3 (bound 30), "
              r"and no gate logit reached $\pm8$")
    add("app.cap_projections", L, "0", str(man["cap_projections_total"]), "eval_all cap_projections_total", ctx_s4)
    add("app.max_frobenius", L, "23.3", f"{max(man['max_frobenius_over_runs']):.1f}",
        "eval_all max_frobenius_over_runs (max over layers)", ctx_s4)
    add("app.frobenius_bound", L, "30", "/".join(sorted({g(c) for c in cfg["frobenius_caps"]})),
        "config frobenius_caps", ctx_s4)
    add("app.logit_clamp_hits", L, "0", str(man["logit_clamp_active_total"]), "eval_all logit_clamp_active_total",
        ctx_s4)
    ctx_s5 = ("took 17 minutes on one 16\\,GB RTX 4090-class GPU, and the component ablation about 29 minutes")
    add("app.eval_minutes", L, "17", f"{man['gpu_seconds'] / 60:.0f}", "eval_all gpu_seconds / 60", ctx_s5,
        machine_dependent=True)
    abl_secs = sum(abl[f"ablation_{k}"]["manifest"]["gpu_seconds"] for k in ABLATIONS)
    add("app.ablation_minutes", L, "29", f"{abl_secs / 60:.0f}", "sum of ablation_* gpu_seconds / 60", ctx_s5,
        machine_dependent=True)

    # ---------------- Appendix: component ablation ----------------
    L = "App. A.4, Component ablation"
    ctx_c = ("power rises from 0.453 for a single fit to 0.891 for $R=5$ and 0.972 for $R=20$, whereas each of the "
             "other components changes it by at most 0.011")
    add("app.ablation_power_a30_R1", L, "0.453", txt(pw["ablation_R1"][0.30]), "ablation_R1 power, a=0.30", ctx_c)
    add("app.ablation_power_a30_R5", L, "0.891", txt(pw["ablation_R5"][0.30]), "ablation_R5 power, a=0.30", ctx_c)
    add("app.ablation_power_a30_R20", L, "0.972", txt(pw["eval_all"][0.30]), "eval_all power, a=0.30", ctx_c)
    dmax = max(abs(pw[t][0.30] - pw["eval_all"][0.30])
               for t in ("ablation_nosharpen", "ablation_noentropy", "ablation_noctx"))
    add("app.ablation_max_change_a30", L, "0.011", txt(dmax),
        "max |power(variant) - power(selected)| at a=0.30 over the three component ablations", ctx_c)

    # ---------------- table captions ----------------
    L = "Table captions (Tables A4, A5)"
    add("caption.tuning_datasets", L, "20 signal datasets at amplitudes 0.20 and 0.30; 20 all-null datasets",
        f"{len(TUNING_SIGNAL_SEEDS)} signal datasets at amplitudes "
        + " and ".join(f"{a:.2f}" for a in TUNING_AMPLITUDES) + f"; {len(TUNING_NULL_SEEDS)} all-null datasets",
        "protocol tuning seeds and amplitudes",
        "(20 signal datasets at amplitudes 0.20 and 0.30; 20 all-null datasets)")
    add("caption.bootstrap_resamples", L, "20,000", f"{20_000:,}", "protocol.bootstrap_ci draws",
        r"Brackets are percentile-bootstrap 95\% intervals over datasets (20,000 resamples)")
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def build(root: Path, out: Path) -> tuple[dict[str, str], list[dict]]:
    rec = load_records(root)
    tables = {
        "gaussian_main.tex": main_table(rec["summary"]),
        "gaussian_full.tex": appendix_table(rec["summary"]),
        "gaussian_tuning.tex": tuning_table(rec["tuning"], rec["selection"]["chosen"]),
        "gaussian_ablation.tex": ablation_table(rec["ablation"]),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, text in tables.items():
        _write(out / name, text)
    numbers = compute_numbers(rec)
    try:
        records = root.resolve().relative_to(PKG).as_posix()
    except ValueError:
        records = str(root)
    _write(out / "numbers.json", json.dumps({"records": records, "numbers": numbers}, indent=2) + "\n")
    return tables, numbers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS, help="root of the run directories")
    ap.add_argument("--from-reference", action="store_true", help="use the shipped records (reference/exact_gaussian)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default <results>/tables)")
    ap.add_argument("--check", action="store_true", help="compare with the paper tables and printed numbers")
    ap.add_argument("--paper-tex", type=Path, default=None, help="paper source; verify every excerpt occurs in it")
    ap.add_argument("--compare-timings", action="store_true",
                    help="also compare the two compute times of App. A.4 (implied by --from-reference); use it for "
                         "records that carry the original manifests, e.g. the output of 'run.py rescore'")
    a = ap.parse_args()
    root = REFERENCE if a.from_reference else a.results
    out = a.out or ((RESULTS / "tables_from_reference") if a.from_reference else (root / "tables"))
    tables, numbers = build(root, out)
    print(f"wrote {', '.join(tables)} and numbers.json to {display_path(out)}")
    timings = a.from_reference or a.compare_timings
    ok = True
    if a.check:
        for name, text in tables.items():
            same = (PAPER_TABLES / name).read_bytes() == text.encode("utf-8")
            ok &= same
            print(f"{name}: {'byte-identical to the paper table' if same else 'DIFFERS from the paper table'}")
        checked = [n for n in numbers if timings or not n["machine_dependent"]]
        ok &= all(n["match"] for n in checked)
        bad = [n for n in checked if not n["match"]]
        md = [n for n in numbers if n["machine_dependent"] and n not in checked]
        rounded = [n for n in checked if n["match"] and n.get("holds_strictly") is False]
        print(f"numbers: {len(checked) - len(bad)}/{len(checked)} equal to the printed strings"
              + "".join(f"; {n['id']} at printed precision ({n['value']:.6f} at a={n['cell']['a']:.2f}, "
                        f"q={n['cell']['q']:.2f})" for n in rounded)
              + ("" if not md else f"; {len(md)} machine-dependent timings not compared ("
                 + ", ".join(f"{n['id']}: {n['computed']} here vs {n['paper']} printed" for n in md) + ")"))
        for n in bad:
            print(f"  MISMATCH {n['id']}: paper {n['paper']!r} computed {n['computed']!r}")
    if a.paper_tex is not None:
        tex = a.paper_tex.read_text(encoding="utf-8")
        missing = [n["id"] for n in numbers if n["context"] not in tex]
        ok &= not missing
        print(f"paper excerpts found verbatim: {len(numbers) - len(missing)}/{len(numbers)}"
              + (f"; missing: {missing}" if missing else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
