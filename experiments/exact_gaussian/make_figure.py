#!/usr/bin/env python3
"""Figure A1 of the paper (exact Gaussian study, App. A.4), regenerated from the run records.

Panels (paper caption of Figure A1):
  (a) mean FDP versus q for each amplitude, percentile-bootstrap 95% intervals over the
      50 datasets; dashed line = nominal level;
  (b) mean power versus q with the same intervals;
  (c) the reported statistic W_j (20-restart mean) at amplitude 0.40 for null and nonnull
      coordinates, pooled over the 50 datasets.

Inputs: ``eval_all/summary.csv`` and ``eval_all/W_per_seed.npz`` under ``--results``
(default ``results/exact_gaussian``; ``--from-reference`` uses ``reference/exact_gaussian``).
Outputs, in ``--out`` (default ``<results>/figures``):

* ``FigureA1_Gaussian.pdf`` and ``.png``: a visual reproduction (fonts and PDF metadata depend on
  the local installation); compare with the paper's figure, shipped as
  ``reference/paper/figures/FigureA1_Gaussian.pdf``;
* ``figureA1_panel_data.csv``: the plotted data -- for panels (a) and (b) the mean and the
  95% interval per (a, q); for panel (c) the histogram bin edges and the null / nonnull
  counts.  ``--check`` compares it byte for byte with
  ``reference/exact_gaussian/figureA1_panel_data.csv`` (generated from the shipped records
  with ``--from-reference --no-plot``); exit status 1 on any difference.

``--no-plot`` writes only the CSV (no matplotlib needed).  The summary is read with
``float_precision="round_trip"``, so the CSV carries the record's values exactly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.common import display_path  # noqa: E402

RESULTS = PKG / "results" / "exact_gaussian"
REFERENCE = PKG / "reference" / "exact_gaussian"
PANEL_CSV = "figureA1_panel_data.csv"
PANEL_FIELDS = ["panel", "series", "a", "q", "mean", "ci_low", "ci_high", "bin_left", "bin_right", "count"]
QS = (0.05, 0.10, 0.15)

# Ordinal single-hue ramps (monotone lightness).
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
RAMP6 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0b2a52"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NULL_GRAY = "#8f8d87"


def panel_data(summary: pd.DataFrame, npz, focus: float) -> dict:
    """Everything the three panels plot.

    ``curves[metric][a] = (q, mean, ci_low, ci_high)`` arrays (q increasing) for panels
    (a) ``FDP`` and (b) ``power``; panel (c): pooled null / nonnull ``W`` at amplitude
    ``focus``, 40 equal bins on ``[-lim, lim]`` with ``lim = 1.02 max|W|``, and the
    histogram counts.
    """
    sig = summary[summary.model == "signal"]
    levels = sorted(sig.signal.unique())
    curves: dict[str, dict[float, tuple]] = {}
    for metric in ("FDP", "power"):
        curves[metric] = {}
        for a in levels:
            g = sig[sig.signal == a].sort_values("q")
            curves[metric][a] = (g.q.to_numpy(), g[f"{metric}_mean"].to_numpy(),
                                 g[f"{metric}_ci_low"].to_numpy(), g[f"{metric}_ci_high"].to_numpy())
    levels_arr = npz["signal_levels"]
    W = npz["W_signal"][np.isclose(levels_arr, focus)]
    sup = npz["supports"][np.isclose(levels_arr, focus)]
    if W.shape[0] == 0:
        raise SystemExit(f"no signal datasets at amplitude {focus}")
    mask = np.zeros_like(W, dtype=bool)
    for i, s in enumerate(sup):
        mask[i, np.asarray(s, int)] = True
    null_W = W[~mask]
    non_W = W[mask]
    lim = float(np.max(np.abs(W))) * 1.02
    bins = np.linspace(-lim, lim, 41)
    return {"levels": levels, "curves": curves, "focus": focus, "n_datasets": int(W.shape[0]),
            "null_W": null_W, "non_W": non_W, "bins": bins,
            "null_counts": np.histogram(null_W, bins=bins)[0], "non_counts": np.histogram(non_W, bins=bins)[0]}


def write_panel_csv(data: dict, path: Path) -> None:
    """``figureA1_panel_data.csv`` (floats as ``repr``, LF line ends, blank = not applicable)."""
    rows = []
    for panel, metric in (("a", "FDP"), ("b", "power")):
        for a in data["levels"]:
            for q, m, lo, hi in zip(*data["curves"][metric][a]):
                rows.append({"panel": panel, "series": metric, "a": repr(float(a)), "q": repr(float(q)),
                             "mean": repr(float(m)), "ci_low": repr(float(lo)), "ci_high": repr(float(hi))})
    edges = data["bins"]
    for series, counts in (("null", data["null_counts"]), ("nonnull", data["non_counts"])):
        for k, n in enumerate(counts):
            rows.append({"panel": "c", "series": series, "a": repr(float(data["focus"])),
                         "bin_left": repr(float(edges[k])), "bin_right": repr(float(edges[k + 1])),
                         "count": str(int(n))})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PANEL_FIELDS, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figure(data: dict, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })
    levels = data["levels"]
    ramp = RAMP6 if len(levels) > 4 else RAMP
    qs = np.array(QS)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), constrained_layout=True)

    for ax, metric, ylabel in zip(axes[:2], ["FDP", "power"], ["Mean FDP", "Mean power"]):
        if metric == "FDP":
            ax.plot(qs, qs, ls="--", lw=1.3, color=INK2, zorder=6, label="nominal $q$")
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        for color, a in zip(ramp, levels):
            _, m, lo, hi = data["curves"][metric][a]
            off = (levels.index(a) - (len(levels) - 1) / 2) * 0.0014   # small horizontal dodge
            ax.errorbar(qs + off, m, yerr=[m - lo, hi - m], color=color, lw=2.0, marker="o",
                        ms=4.5, mec="white", mew=0.8, capsize=0, elinewidth=1.0, label=f"{a:.2f}", zorder=3)
        ax.set_xticks(qs)
        ax.set_xlabel("Target level $q$")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title("(a) FDP versus $q$" if metric == "FDP" else "(b) Power versus $q$",
                     loc="left", fontsize=9, color=INK)
    axes[0].set_ylim(0, 0.2)
    axes[1].set_ylim(0, 1.04)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Amplitude $a$", fontsize=7.5, title_fontsize=7.5, frameon=False,
               loc="outside lower center", ncol=len(labels))

    ax = axes[2]
    bins = data["bins"]
    n_null, _, _ = ax.hist(data["null_W"], bins=bins, color=NULL_GRAY, alpha=0.85, label="null",
                           edgecolor="white", linewidth=0.4)
    n_non, _, _ = ax.hist(data["non_W"], bins=bins, color=ramp[3], alpha=0.85, label="nonnull",
                          edgecolor="white", linewidth=0.4)
    # The CSV holds exactly the plotted counts.
    assert np.array_equal(n_null, data["null_counts"]) and np.array_equal(n_non, data["non_counts"])
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("$W_j$")
    ax.set_ylabel(f"Coordinates ({data['n_datasets']} datasets)")
    ax.set_title(f"(c) $W_j$ at amplitude {data['focus']:.2f}", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "FigureA1_Gaussian.pdf", out / "FigureA1_Gaussian.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=200)
    plt.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS, help="root of the run directories (default results/exact_gaussian)")
    ap.add_argument("--from-reference", action="store_true", help="use the shipped records in reference/exact_gaussian")
    ap.add_argument("--focus", type=float, default=0.40, help="amplitude of panel (c)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default <results>/figures)")
    ap.add_argument("--no-plot", action="store_true", help="write only the panel-data CSV")
    ap.add_argument("--check", action="store_true",
                    help=f"compare the panel data with reference/exact_gaussian/{PANEL_CSV} byte for byte")
    a = ap.parse_args()
    root = REFERENCE if a.from_reference else a.results
    out = a.out or ((RESULTS / "figures_from_reference") if a.from_reference else (root / "figures"))
    summary = pd.read_csv(root / "eval_all" / "summary.csv", float_precision="round_trip")
    with np.load(root / "eval_all" / "W_per_seed.npz") as npz:
        data = panel_data(summary, npz, a.focus)
    csv_path = out / PANEL_CSV
    write_panel_csv(data, csv_path)
    print("wrote", display_path(csv_path))
    if not a.no_plot:
        for p in figure(data, out):
            print("wrote", display_path(p))
    if a.check:
        ref = REFERENCE / PANEL_CSV
        same = ref.exists() and ref.read_bytes() == csv_path.read_bytes()
        print(f"Figure A1 panel data: {'byte-identical to' if same else 'DIFFERS from'} "
              f"reference/exact_gaussian/{PANEL_CSV}")
        sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
