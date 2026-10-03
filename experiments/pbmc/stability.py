"""PBMC top-100 stability over the 20 matched bootstrap refits (paper Table A8, App. A.5).

Computed quantities (paper App. A.8 "Metric definitions"; universe p = 1,719, k = 100):

* **Run vs run** (every method): mean +- SD (ddof 1) over the 190 pairs of single-run top-100 sets of the
  set Jaccard index, the Kuncheva index ``(|A & B| - k^2/p) / (k - k^2/p)`` and the marker-only
  Jaccard index; mean MarkerHits / CoverageScore over the 20 sets.
* **Consensus vs members** (SSRK): coordinatewise median of ``W`` over the first K refits, top 100,
  mean index against each of its own K member sets (also mean-W and selection-frequency operators).
* **Consensus vs consensus** (SSRK): consensus sets built on disjoint groups of K refits; 2,000 random
  disjoint pairs (``default_rng(7)`` per operator) and the deterministic consecutive blocks.
  In Table A8 the Jaccard, Kuncheva and marker-Jaccard columns of these rows are the means (and SDs)
  over the 2,000 Monte Carlo pairs, while the Markers / coverage column is the mean over the
  deterministic consecutive blocks (20/K disjoint consensus sets: 6, 4 and 2 for K = 3, 5, 10;
  ``block_MarkerHits`` / ``block_Coverage``), as in the original table code. The paper check
  ``claim_consensus_sets_retain_same_markers`` (below) confirms that every consensus set, blocks and
  Monte Carlo draws alike, contains the same 14 markers, so both choices give 14.00 / 2.80.

SSRK sets are the top 100 of the 20 saved ``W`` arrays (``units/NNN_W.npy``); baseline sets are the
stored selections of the matched protocol (``reference/pbmc/baseline_selections.csv``; baselines are
not reimplemented in this package). The ARI/NMI of the aggregated sets (PCA + k-means on the full
matrix, seed 20000 + 41) are reported in ``stability.json`` as well.

Paper checks (``stability.json`` sections ``paper_claims`` and ``checks``, enforced by ``--strict``): the
statements of Sec. 5.3 and App. A.5 "PBMC: extended biological evidence" (paragraph "Stability"), quoted
verbatim in ``reference/pbmc/paper_stability_excerpts.tex``:

* every SSRK refit has marker hits 14 and coverage 2.80, from five groups, and the same 14 markers;
* SSRK has the highest whole-set Jaccard and Kuncheva indices of the six methods;
* SSRK Jaccard 0.772 against 0.756 for SSFS, marker Jaccard 1.000 (Sec. 5.3);
* every median-W consensus set (first K refits, blocks and Monte Carlo draws; K = 3, 5, 10) retains
  the same 14 markers;
* larger groups improve whole-set agreement (consensus-vs-consensus Jaccard and Kuncheva increase
  from K = 3 to 5 to 10).

Usage (from the package root)::

    python experiments/pbmc/stability.py --units-dir results/pbmc/units      # after run.py --units all
    python experiments/pbmc/stability.py --from-reference                    # shipped SSRK W arrays

Outputs in ``--out``: ``stability.json`` (sections ``stability_pairwise_recomputed``,
``stability_aggregation_recomputed``, ``consensus_vs_consensus``, ``marker_universe``, ``paper_claims``,
``checks``) and ``stability_body.tex`` (the ten rows of Table A8). Both are compared with the reference
values (``reference/pbmc/stability_body_paper.tex``, ``fill_nr_numbers_stability.json``); ``--strict``
turns a mismatch or a failed paper check into exit status 1. ``--paper-tex neurips_2026.tex``
additionally checks that the quoted excerpts occur verbatim in the paper source. The script imports no
torch.
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from experiments.common import (  # noqa: E402
    DEFAULT_DATA_DIR,
    REFERENCE_DIR,
    blas_threads,
    display_path,
    load_json,
    resolve_path,
    save_json,
    write_manifest,
)
from experiments.pbmc.data import PBMC_MARKER_MAP, load_pbmc, pbmc_marker_indices  # noqa: E402
from ssrk.ranking import deterministic_top_k  # noqa: E402

P_UNIVERSE = 1719
K_TOP = 100
N_UNITS = 20
BOOT_SEEDS = tuple(20_000 + 997 * i for i in range(N_UNITS))
#: Evaluation seed of the aggregated sets (first bootstrap seed + 41).
AGG_EVAL_SEED = BOOT_SEEDS[0] + 41
#: Monte Carlo draws of disjoint consensus pairs, and the seed of their generator.
MC_DRAWS = 2000
MC_SEED = 7
BASELINES = ("Variance", "STG", "CAE", "GAEFS", "SSFS")
AGGREGATIONS = ("mean_W", "median_W", "selection_frequency")


# ------------------------------------------------------------------------------------- indices
def kuncheva(a: set, b: set, p: int = P_UNIVERSE, k: int = K_TOP) -> float:
    e = k * k / p
    return (len(a & b) - e) / (k - e)


def jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def msd(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "sd_ddof1": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "sd_ddof0": float(x.std(ddof=0)), "min": float(x.min()), "max": float(x.max()), "n": int(x.size)}


def aggregate(mat: np.ndarray, how: str, k: int) -> list[int]:
    """Consensus top-k of several refits: mean-W, median-W, or selection frequency (mean-W tie-break)."""
    mat = mat.astype(np.float64)
    if not np.all(np.isfinite(mat)):
        raise ValueError("W arrays must be finite")
    if how == "mean_W":
        s = mat.mean(0)
    elif how == "median_W":
        s = np.median(mat, 0)
    elif how == "selection_frequency":
        freq = np.zeros(mat.shape[1])
        mean_score = mat.mean(0)
        for row in mat:
            freq[deterministic_top_k(row, k)] += 1.0
        freq /= mat.shape[0]
        scale = max(1.0, float(np.max(np.abs(mean_score))))
        s = freq * (scale * 10.0) + mean_score / scale
    else:
        raise ValueError(how)
    return deterministic_top_k(s, k)


class Markers:
    """Curated marker indices within the 1,719-gene universe and the marker metrics."""

    def __init__(self, gene_names):
        self.groups, self.all = pbmc_marker_indices(gene_names)
        self.gene_names = list(gene_names)

    def metrics(self, sel) -> tuple[int, float, int]:
        s = set(sel)
        hits = len(s & self.all)
        gh = sum(bool(s & v) for v in self.groups.values())
        return hits, hits / max(1, gh), gh

    def universe(self) -> dict:
        name_to_idx = {str(g): i for i, g in enumerate(self.gene_names)}
        return {
            "n_curated_genes": int(sum(len(v) for v in PBMC_MARKER_MAP.values())),
            "n_present_in_1719_hvgs": len(self.all),
            "present_by_group": {g: sorted(self.gene_names[i] for i in v) for g, v in self.groups.items()},
            "missing": sorted(g for gl in PBMC_MARKER_MAP.values() for g in gl if g not in name_to_idx),
        }


# -------------------------------------------------------------------------------------- inputs
def load_ssrk_W(units_dir: Path) -> tuple[list[np.ndarray], list[list[int]] | None]:
    """The 20 SSRK ``W`` arrays and, if present, the stored ``selected_indices`` of each unit."""
    W, stored = [], []
    for u in range(N_UNITS):
        f = units_dir / f"{u:03d}_W.npy"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: stability needs all {N_UNITS} PBMC units (run.py --units all)")
        W.append(np.load(f))
        rec = units_dir / f"{u:03d}.json"
        stored.append(load_json(rec)["selected_indices"] if rec.exists() else None)
    return W, (stored if all(s is not None for s in stored) else None)


def load_baseline_sets(path: Path) -> dict[str, list[set]]:
    """Stored baseline selections, per method, ordered by bootstrap id."""
    with Path(path).open(encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    out = {}
    for m in BASELINES:
        rs = sorted((r for r in rows if r["method"] == m), key=lambda r: int(r["bootstrap_id"]))
        if [int(r["bootstrap_id"]) for r in rs] != list(range(N_UNITS)):
            raise RuntimeError(f"{path}: method {m} does not have bootstrap ids 0..{N_UNITS - 1}")
        out[m] = [set(map(int, str(r["selected_indices"]).split(";"))) for r in rs]
    return out


# ---------------------------------------------------------------------------------- computation
def pairwise_block(sets: list[set], markers: Markers) -> dict:
    J = [jac(a, b) for a, b in itertools.combinations(sets, 2)]
    K = [kuncheva(a, b) for a, b in itertools.combinations(sets, 2)]
    MJ = [jac(a & markers.all, b & markers.all) for a, b in itertools.combinations(sets, 2)]
    mm = [markers.metrics(s) for s in sets]
    return {"Jaccard": msd(J), "Kuncheva": msd(K), "MarkerOnlyJaccard": msd(MJ),
            "MarkerHits_recomputed": msd([x[0] for x in mm]),
            "Coverage_recomputed": msd([x[1] for x in mm]),
            "groups_hit": msd([x[2] for x in mm]),
            "n_sets": len(sets)}


def consensus_stability(W, kr: int, how: str, markers: Markers, n_draws: int = MC_DRAWS, seed: int = MC_SEED,
                        collect: dict | None = None) -> dict:
    """Consensus-vs-consensus stability of operator ``how`` on disjoint groups of ``kr`` refits.

    If ``collect`` is a dict, the distinct consensus sets are stored in it as ``collect["blocks"]``
    (the deterministic consecutive blocks) and ``collect["mc"]`` (every set drawn by the Monte Carlo
    pairs); the returned statistics do not depend on it.
    """
    # (a) deterministic disjoint consecutive blocks
    blocks = [list(range(b * kr, (b + 1) * kr)) for b in range(N_UNITS // kr)]
    bsets = [set(aggregate(np.stack([W[i] for i in blk]), how, K_TOP)) for blk in blocks]
    Jb = [jac(a, b) for a, b in itertools.combinations(bsets, 2)]
    Kb = [kuncheva(a, b) for a, b in itertools.combinations(bsets, 2)]
    Mb = [jac(a & markers.all, b & markers.all) for a, b in itertools.combinations(bsets, 2)]
    hits_b = [markers.metrics(s)[0] for s in bsets]
    cov_b = [markers.metrics(s)[1] for s in bsets]
    # (b) Monte Carlo over random pairs of disjoint kr-subsets
    rng = np.random.default_rng(seed)
    Jm, Km, Mm = [], [], []
    cache = {}
    for _ in range(n_draws):
        perm = rng.permutation(N_UNITS)
        a_idx, b_idx = tuple(sorted(perm[:kr])), tuple(sorted(perm[kr:2 * kr]))
        for key in (a_idx, b_idx):
            if key not in cache:
                cache[key] = set(aggregate(np.stack([W[i] for i in key]), how, K_TOP))
        A, B = cache[a_idx], cache[b_idx]
        Jm.append(jac(A, B))
        Km.append(kuncheva(A, B))
        Mm.append(jac(A & markers.all, B & markers.all))
    if collect is not None:
        collect["blocks"] = list(bsets)
        collect["mc"] = list(cache.values())
    return {"K": kr, "aggregation": how, "blocks": len(blocks), "block_pairs": len(Jb),
            "block_J": msd(Jb), "block_Kuncheva": msd(Kb), "block_markerJ": msd(Mb),
            "block_MarkerHits": msd(hits_b), "block_Coverage": msd(cov_b),
            "mc_draws": n_draws, "mc_J": msd(Jm), "mc_Kuncheva": msd(Km), "mc_markerJ": msd(Mm)}


def compute_stability(W, baseline_sets, markers: Markers, X=None, labels=None,
                      eval_threads: int = 1, consensus_sets: dict | None = None) -> tuple[dict, list]:
    """All stability quantities; ARI/NMI of the aggregated sets only if ``X``/``labels`` are given.

    If ``consensus_sets`` is a dict, every consensus set is stored in it under ``(operator, K)`` as
    ``{"members": [set of the first K refits], "blocks": [...], "mc": [...]}`` (for the paper checks).
    """
    single_sets = [set(deterministic_top_k(w, K_TOP)) for w in W]
    out: dict = {}
    sets_by_method = dict(baseline_sets)
    sets_by_method["SSRK"] = single_sets
    out["stability_pairwise_recomputed"] = {m: pairwise_block(sets_by_method[m], markers)
                                            for m in sorted(sets_by_method)}
    out["marker_universe"] = markers.universe()

    if X is not None:
        from experiments.pbmc.run import cluster_metrics
    agg_rows = []
    for kr, how in [(1, "mean_W")] + [(k, h) for k in (3, 5, 10) for h in AGGREGATIONS]:
        sel = aggregate(np.stack(W[:kr]), how, K_TOP)
        s = set(sel)
        members = single_sets[:kr]
        hits, cov, gh = markers.metrics(sel)
        row = {
            "K": kr, "aggregation": "single_run" if kr == 1 else how,
            "mean_J_to_members": float(np.mean([jac(s, r) for r in members])),
            "mean_Kuncheva_to_members": float(np.mean([kuncheva(s, r) for r in members])),
            "mean_markerJ_to_members": float(np.mean([jac(s & markers.all, r & markers.all) for r in members])),
            "MarkerHits": hits, "Coverage": cov, "groups_hit": gh,
        }
        if X is not None:
            with blas_threads(eval_threads):
                row["ARI_recomputed"], row["NMI_recomputed"] = cluster_metrics(X, labels, sel, AGG_EVAL_SEED, 50)
        agg_rows.append(row)
        if consensus_sets is not None and kr > 1:
            consensus_sets.setdefault((how, kr), {})["members"] = [s]
    out["stability_aggregation_recomputed"] = agg_rows
    out["consensus_vs_consensus"] = [
        consensus_stability(W, kr, how, markers,
                            collect=None if consensus_sets is None else consensus_sets.setdefault((how, kr), {}))
        for kr in (3, 5, 10) for how in AGGREGATIONS]
    return out, single_sets


# ------------------------------------------------------------------------------- paper checks
#: Verbatim copies of the paper lines that state the PBMC stability results (Sec. 5.3, App. A.5).
PAPER_EXCERPTS = REFERENCE_DIR / "pbmc" / "paper_stability_excerpts.tex"


def _p3(x: float) -> str:
    """Three decimals with the leading zero, as printed in the running text (``0.772``)."""
    return f"{x:.3f}"


def paper_claims(d: dict, single_sets: list[set], consensus_sets: dict, markers: Markers) -> list[dict]:
    """The PBMC stability statements of Sec. 5.3 and App. A.5 ("Stability"), evaluated on the results.

    Each claim: ``id`` (also the check name), ``location``, ``text`` (verbatim fragment of
    ``reference/pbmc/paper_stability_excerpts.tex``), ``ok`` and the ``values`` it was decided on.
    """
    sp = d["stability_pairwise_recomputed"]
    methods = sorted(sp)
    baselines = [m for m in methods if m != "SSRK"]
    claims = []

    marker_sets = [frozenset(s & markers.all) for s in single_sets]
    refit = [markers.metrics(s) for s in single_sets]
    common = marker_sets[0]
    claims.append(dict(
        id="claim_refits_same_fourteen_markers", location="App. A.5, paragraph Stability",
        text="Every SSRK refit contains the same 14 markers from five lineage groups (marker hits 14 and "
             "coverage score 2.80 in all 20 refits)",
        ok=(len(single_sets) == N_UNITS and len(set(marker_sets)) == 1 and all(h == 14 for h, _, _ in refit)
            and all(f"{c:.2f}" == "2.80" for _, c, _ in refit) and all(g == 5 for _, _, g in refit)),
        values={"distinct_marker_sets": len(set(marker_sets)), "markers": sorted(markers.gene_names[i] for i in common),
                "hits": sorted({h for h, _, _ in refit}), "coverage": sorted({c for _, c, _ in refit}),
                "groups_hit": sorted({g for _, _, g in refit})}))

    J = {m: sp[m]["Jaccard"]["mean"] for m in methods}
    K = {m: sp[m]["Kuncheva"]["mean"] for m in methods}
    claims.append(dict(
        id="claim_ssrk_highest_jaccard_and_kuncheva", location="App. A.5, paragraph Stability",
        text="SSRK has the highest whole-set Jaccard and Kuncheva indices of the six methods",
        ok=len(methods) == 6 and all(J["SSRK"] > J[m] and K["SSRK"] > K[m] for m in baselines),
        values={"Jaccard": J, "Kuncheva": K}))

    MJ = sp["SSRK"]["MarkerOnlyJaccard"]["mean"]
    best_baseline = max(baselines, key=lambda m: J[m])
    claims.append(dict(
        id="claim_ssrk_most_stable_jaccard_against_ssfs", location="Sec. 5.3",
        text="its top-100 set is the most stable of the six methods (Jaccard index 0.772 against 0.756 for "
             "SSFS, marker Jaccard index 1.000",
        ok=(all(J["SSRK"] > J[m] for m in baselines) and best_baseline == "SSFS" and _p3(J["SSRK"]) == "0.772"
            and _p3(J["SSFS"]) == "0.756" and _p3(MJ) == "1.000"),
        values={"Jaccard_SSRK": J["SSRK"], "Jaccard_SSFS": J["SSFS"], "most_stable_baseline": best_baseline,
                "MarkerJaccard_SSRK": MJ}))

    per_k, others = {}, {}
    for (how, kr), groups in sorted(consensus_sets.items(), key=lambda t: (t[0][1], t[0][0])):
        sets = [s for kind in ("members", "blocks", "mc") for s in groups.get(kind, [])]
        same = all(frozenset(s & markers.all) == common for s in sets)
        distinct = len({frozenset(s) for s in sets})
        (per_k if how == "median_W" else others)[f"{how}_K{kr}"] = {"distinct_sets": distinct,
                                                                     "all_retain_same_markers": same}
    claims.append(dict(
        id="claim_consensus_sets_retain_same_markers", location="App. A.5, paragraph Stability",
        text="All consensus rankings retain the same 14 markers",
        ok=len(common) == 14 and len(per_k) == 3 and all(v["all_retain_same_markers"] for v in per_k.values()),
        values={"median_W": per_k, "other_operators (not claimed)": others}))

    cons = {r["K"]: r for r in d["consensus_vs_consensus"] if r["aggregation"] == "median_W"}
    Jc = [cons[k]["mc_J"]["mean"] for k in (3, 5, 10)]
    Kc = [cons[k]["mc_Kuncheva"]["mean"] for k in (3, 5, 10)]
    claims.append(dict(
        id="claim_larger_groups_improve_agreement", location="App. A.5, paragraph Stability",
        text="larger groups improve whole-set agreement",
        ok=Jc[0] < Jc[1] < Jc[2] and Kc[0] < Kc[1] < Kc[2],
        values={"consensus_vs_consensus_Jaccard_K3_K5_K10": Jc, "consensus_vs_consensus_Kuncheva_K3_K5_K10": Kc}))
    return claims


def paper_excerpts() -> list[str]:
    """The quoted paper lines (comment lines removed)."""
    text = PAPER_EXCERPTS.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln and not ln.startswith("%")]


# -------------------------------------------------------------------------------------- table
def _f(x: float) -> str:
    s = f"{x:.3f}"
    return s[1:] if s.startswith("0.") else s


def table_rows(d: dict) -> str:
    """Body of Table A8 (LaTeX label tab:pbmc_stability; ten rows).

    Consensus-vs-consensus rows: Jaccard / Kuncheva / marker Jaccard over the Monte Carlo pairs,
    Markers / coverage over the deterministic consecutive blocks (see the module docstring).
    """
    BS = "\\"
    EOL = BS + BS
    MID = EOL + BS + "midrule"
    PM = BS + "pm"
    sp = d["stability_pairwise_recomputed"]
    cons = {(r["K"], r["aggregation"]): r for r in d["consensus_vs_consensus"]}
    mem = {(r["K"], r["aggregation"]): r for r in d["stability_aggregation_recomputed"]}

    def pair(r, key):
        return f"${_f(r[key]['mean'])}{PM}{_f(r[key]['sd_ddof1'])}$"

    s = sp["SSRK"]
    rows = [f"SSRK, single run & run vs run & {pair(s, 'Jaccard')} & {pair(s, 'Kuncheva')} & "
            f"${_f(s['MarkerOnlyJaccard']['mean'])}$ & "
            f"{s['MarkerHits_recomputed']['mean']:.2f} / {s['Coverage_recomputed']['mean']:.2f}{EOL}"]
    m3 = mem[(3, "median_W")]
    rows.append(f"SSRK, median-$W$, $K=3$ & consensus vs members & ${_f(m3['mean_J_to_members'])}$ & "
                f"${_f(m3['mean_Kuncheva_to_members'])}$ & ${_f(m3['mean_markerJ_to_members'])}$ & "
                f"{m3['MarkerHits']:.2f} / {m3['Coverage']:.2f}{EOL}")
    for K in (3, 5, 10):
        c = cons[(K, "median_W")]
        end = MID if K == 10 else EOL
        rows.append(f"SSRK, median-$W$, $K={K}$ & consensus vs consensus & {pair(c, 'mc_J')} & "
                    f"{pair(c, 'mc_Kuncheva')} & ${_f(c['mc_markerJ']['mean'])}$ & "
                    f"{c['block_MarkerHits']['mean']:.2f} / {c['block_Coverage']['mean']:.2f}{end}")
    for m, lab in [("SSFS", "SSFS"), ("STG", "STG"), ("GAEFS", "GAEFS"), ("CAE", "CAE"),
                   ("Variance", "Variance")]:
        r = sp[m]
        rows.append(f"{lab}, single run & run vs run & {pair(r, 'Jaccard')} & {pair(r, 'Kuncheva')} & "
                    f"${_f(r['MarkerOnlyJaccard']['mean'])}$ & "
                    f"{r['MarkerHits_recomputed']['mean']:.2f} / {r['Coverage_recomputed']['mean']:.2f}{EOL}")
    return "\n".join(rows) + "\n"


def _diff_numbers(a, b, path="") -> list[str]:
    """Exact recursive comparison of two JSON-like structures; returns the differing paths."""
    if isinstance(a, dict) and isinstance(b, dict):
        diffs = [f"{path}/{k}: missing" for k in b if k not in a]
        for k in b:
            if k in a:
                diffs += _diff_numbers(a[k], b[k], f"{path}/{k}")
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff_numbers(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--units-dir", default=None, help="directory with NNN_W.npy (default results/pbmc/units)")
    src.add_argument("--from-reference", action="store_true", help="use the shipped W arrays in reference/pbmc/units")
    ap.add_argument("--baseline-selections", default=str(REFERENCE_DIR / "pbmc" / "baseline_selections.csv"),
                    help="baseline top-100 selections (default reference/pbmc/baseline_selections.csv)")
    ap.add_argument("--data-dir", default=None, help="data directory (default: <package>/data)")
    ap.add_argument("--skip-clustering", action="store_true",
                    help="do not recompute ARI/NMI of aggregated sets (not needed for the table; the data file is "
                         "still read for the gene names)")
    ap.add_argument("--eval-threads", type=int, default=1,
                    help="BLAS/OpenMP threads for PCA + k-means of the aggregated sets (0 = library default)")
    ap.add_argument("--out", default=None, help="output directory (default results/pbmc or results/pbmc_reference)")
    ap.add_argument("--strict", action="store_true",
                    help="exit with status 1 if any reference check or paper check fails")
    ap.add_argument("--paper-tex", default=None,
                    help="optional paper source (neurips_2026.tex): also check that the quoted excerpts of "
                         "reference/pbmc/paper_stability_excerpts.tex occur in it verbatim")
    a = ap.parse_args(argv)

    if a.from_reference:
        units_dir = REFERENCE_DIR / "pbmc" / "units"
        out = resolve_path(a.out, resolve_path("results/pbmc_reference", Path()))
    else:
        units_dir = resolve_path(a.units_dir or "results/pbmc/units", Path())
        out = resolve_path(a.out, units_dir.parent)
    t0 = time.perf_counter()
    W, stored = load_ssrk_W(units_dir)
    baseline_sets = load_baseline_sets(resolve_path(a.baseline_selections, Path()))
    X, labels, genes = load_pbmc(resolve_path(a.data_dir, DEFAULT_DATA_DIR))
    markers = Markers(genes)
    if a.skip_clustering:
        X = labels = None
    consensus_sets: dict = {}
    d, single_sets = compute_stability(W, baseline_sets, markers, X, labels, a.eval_threads, consensus_sets)

    checks = {}
    if stored is not None:
        checks["score_arrays_match_stored_sets"] = all(single_sets[i] == set(stored[i]) for i in range(N_UNITS))
    body = table_rows(d)
    paper_body = (REFERENCE_DIR / "pbmc" / "stability_body_paper.tex").read_text(encoding="utf-8")
    checks["table_body_equals_paper"] = body == paper_body
    ref = load_json(REFERENCE_DIR / "pbmc" / "fill_nr_numbers_stability.json")
    for key in ("stability_pairwise_recomputed", "stability_aggregation_recomputed", "consensus_vs_consensus",
                "marker_universe"):
        expected = ref[key]
        if a.skip_clustering and key == "stability_aggregation_recomputed":
            expected = [{k: v for k, v in r.items() if k not in ("ARI_recomputed", "NMI_recomputed")} for r in expected]
        diffs = _diff_numbers(d[key], expected)
        checks[f"{key}_equals_reference"] = not diffs
        if diffs:
            checks[f"{key}_differences"] = diffs[:20]

    # statements of Sec. 5.3 and App. A.5 ("Stability"), quoted verbatim in reference/pbmc/
    claims = paper_claims(d, single_sets, consensus_sets, markers)
    excerpts = paper_excerpts()
    for c in claims:
        c["text_in_excerpts"] = any(c["text"] in ln for ln in excerpts)
        checks[c["id"]] = bool(c["ok"] and c["text_in_excerpts"])
    if a.paper_tex:
        source = resolve_path(a.paper_tex, Path()).read_text(encoding="utf-8").splitlines()
        checks["paper_excerpts_verbatim_in_paper_tex"] = all(ln in source for ln in excerpts)
    d["paper_claims"] = claims
    d["checks"] = checks
    d["inputs"] = {"units_dir": display_path(units_dir), "baseline_selections": display_path(a.baseline_selections)}

    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "stability.json", d)
    (out / "stability_body.tex").write_text(body, encoding="utf-8")
    write_manifest(out, {"protocol": "pbmc_stability", "inputs": d["inputs"], "checks": checks,
                         "total_seconds": time.perf_counter() - t0}, name="stability_manifest.json")
    print(body, end="")
    print(json.dumps({k: v for k, v in checks.items() if not k.endswith("_differences")}, indent=1))
    if not all(v for k, v in checks.items() if not k.endswith("_differences")):
        print("STABILITY REFERENCE CHECK FAILED (see 'checks' in stability.json)", file=sys.stderr)
        if a.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
