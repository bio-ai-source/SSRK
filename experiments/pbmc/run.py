"""PBMC 3k matched protocol, SSRK side (paper Sec. 5.3 and App. A.3.3 "Matched comparison"; PBMC rows
of Table A3 and SSRK rows of Table A8).

Each of the 20 units bootstraps the 2,504 cells (``default_rng(seed)``), draws ONE independent
Gaussian-marginal reference for the bootstrap sample (seed = unit seed), fits the self-reconstruction
mode once (``ssrk.selfrecon``), keeps the top 100 genes of ``W`` (deterministic ranking), and evaluates

* ARI / NMI of k-means (k = number of Leiden clusters in the resample, 10 initializations) on the
  first 50 randomized-PCA components of the selected genes (evaluation seed = unit seed + 41;
  single-thread BLAS/OpenMP by default, see ``--eval-threads``);
* MarkerHits (curated markers in the top 100) and CoverageScore (hits per marker group hit).

Usage (from the package root)::

    python experiments/pbmc/run.py --units all            # 20 units, ~30-50 s each on an RTX 4090
    python experiments/pbmc/run.py --units 0,5 --out results/pbmc_check

Outputs in ``--out``: ``units/NNN.json`` (unit, seed, ARI, NMI, MarkerHits, CoverageScore,
selected_indices, meta), ``units/NNN_W.npy`` (float32 (1719,)), ``units/NNN_Wall.npy`` (float32 (1, 1719)),
``per_run.csv`` (all unit records present in ``units/``) and ``manifest.json``.

Everything after the fit (top-100 selection, clustering and marker metrics) is :func:`score_unit`, which
``experiments/pbmc/rescore.py`` applies to the shipped ``W`` on the CPU (Level A of ``verify.py``).
Importing this module does not import torch; :func:`main` imports it only for the fit.
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from ssrk.determinism import enforce_cublas_workspace  # noqa: E402  (torch-free)

# deterministic cuBLAS: ':4096:8' (the value of the paper records) must be in the environment before the
# first cuBLAS call; any other pre-set value is overridden with a warning
enforce_cublas_workspace()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from typing import Sequence  # noqa: E402

import numpy as np  # noqa: E402

from experiments.common import (  # noqa: E402
    DEFAULT_DATA_DIR,
    display_path,
    REFERENCE_DIR,
    blas_threads,
    compare_unit_with_reference,
    load_json,
    parse_units,
    resolve_path,
    write_csv,
    write_manifest,
)
from experiments.pbmc.data import bootstrap_resample, load_pbmc, pbmc_marker_indices  # noqa: E402

#: Bootstrap seeds of the 20 matched units (unit i uses 20000 + 997 i).
PBMC_BOOTSTRAP_SEEDS = tuple(20_000 + 997 * i for i in range(20))
#: Selection budget (top-k genes).
PBMC_K = 100
#: Evaluation seed offset (PCA + k-means random_state = unit seed + 41).
EVAL_SEED_OFFSET = 41
#: PCA components for the clustering evaluation.
PCA_COMPONENTS = 50
#: Reference used for PBMC (independent Gaussian marginal) and number of restarts.
REFERENCE = "independent"
RESTARTS = 1
#: BLAS/OpenMP threads of the clustering evaluation. The original runs used the library default (20
#: threads on the reference machine); the 20 SSRK evaluations and the stability aggregates give
#: bitwise-identical ARI/NMI with 1, 2, 4 and 20 threads, and 1 thread removes the core-count dependence.
EVAL_THREADS = 1


def pbmc_config():
    """Resolved SSRK configuration of the PBMC matched protocol (record set ``EVAL_pbmc_e200sym``).

    Resolved from the original configuration machinery (``configs/paper.yaml`` real_data.pbmc overrides,
    matched-protocol base settings, and the run overrides ``{"epochs": 200,
    "gate_parameterization": "symmetric"}``) and hard-coded here.
    """
    from ssrk.selfrecon import SelfReconConfig

    return SelfReconConfig(
        encoder_dims=(256, 128),
        latent_dim=32,
        decoder_dims=(128, 256),
        temperature=1.0,
        use_batchnorm=True,
        epochs=200,
        batch_size=128,
        lr=3e-4,
        lr_gate_factor=0.05,
        lambda_entropy=0.005,
        mask_prob=0.5,
        stage1_frac=0.5,
        freeze_gates_stage1=True,
        entropy_weighting="gap",
        gate_parameterization="symmetric",
    )


def cluster_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    selected: Sequence[int],
    seed: int,
    pca_components: int = PCA_COMPONENTS,
) -> tuple[float, float]:
    """ARI / NMI of k-means on randomized-PCA components of the selected columns (paper App. A.3.2, "Evaluation")."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    cols = np.asarray(sorted(selected), dtype=int)
    X_selected = X[:, cols]
    n_comp = min(int(pca_components), X_selected.shape[1], X_selected.shape[0] - 1)
    if n_comp > 0 and n_comp < X_selected.shape[1]:
        X_selected = PCA(
            n_components=n_comp,
            svd_solver="randomized",
            random_state=int(seed),
        ).fit_transform(X_selected)
    pred = KMeans(
        n_clusters=len(np.unique(labels)),
        n_init=10,
        random_state=int(seed),
    ).fit_predict(X_selected)
    return (
        float(adjusted_rand_score(labels, pred)),
        float(normalized_mutual_info_score(labels, pred)),
    )


def marker_metrics(selected: Sequence[int], marker_groups: dict, all_markers: set) -> tuple[int, float]:
    """MarkerHits = |selected & markers|; CoverageScore = hits / max(1, number of marker groups hit)."""
    selected_set = set(int(i) for i in selected)
    hits = len(selected_set & all_markers)
    groups_hit = sum(bool(selected_set & members) for members in marker_groups.values())
    return int(hits), float(hits / max(1, groups_hit))


def score_unit(unit: int, W: np.ndarray, Xb: np.ndarray, yb: np.ndarray, marker_groups, all_markers,
               eval_threads: int = EVAL_THREADS) -> dict:
    """Score the statistics ``W`` of one unit on its bootstrap resample ``(Xb, yb)`` (no torch).

    Top 100 of ``W`` (deterministic ranking), ARI/NMI of PCA + k-means on the selected genes (evaluation
    seed = unit seed + 41, ``eval_threads`` BLAS/OpenMP threads), MarkerHits and CoverageScore. Returns the
    unit record without ``meta``.
    """
    from ssrk.ranking import deterministic_top_k

    seed = PBMC_BOOTSTRAP_SEEDS[unit]
    sel = deterministic_top_k(W, PBMC_K)
    with blas_threads(eval_threads):
        ari, nmi = cluster_metrics(Xb, yb, sel, seed + EVAL_SEED_OFFSET, PCA_COMPONENTS)
    hits, cov = marker_metrics(sel, marker_groups, all_markers)
    return dict(unit=unit, seed=seed, ARI=ari, NMI=nmi, MarkerHits=hits, CoverageScore=cov, selected_indices=sel)


def run_unit(unit: int, X: np.ndarray, labels: np.ndarray, marker_groups, all_markers, config, device,
             eval_threads: int = EVAL_THREADS) -> tuple:
    """One matched-protocol unit; returns ``(record, W, Wall)``."""
    from ssrk.selfrecon import ssrk_scores

    seed = PBMC_BOOTSTRAP_SEEDS[unit]
    Xb, yb = bootstrap_resample(X, labels, seed)
    # SSRK fit seed = unit seed (method offset 0 of the matched protocol)
    W, Wall, meta = ssrk_scores(Xb, seed, REFERENCE, config, device, restarts=RESTARTS)
    rec = score_unit(unit, W, Xb, yb, marker_groups, all_markers, eval_threads)
    rec["meta"] = meta
    return rec, W, Wall


PER_RUN_FIELDS = ["unit", "seed", "ARI", "NMI", "MarkerHits", "CoverageScore", "train_seconds", "selected_indices"]


def write_per_run(units_dir: Path, out_csv: Path) -> list[dict]:
    """Collect every ``units/NNN.json`` into ``per_run.csv`` (selected indices ';'-joined)."""
    rows = []
    for f in sorted(units_dir.glob("[0-9][0-9][0-9].json")):
        rec = load_json(f)
        rows.append({
            "unit": rec["unit"], "seed": rec["seed"], "ARI": repr(float(rec["ARI"])), "NMI": repr(float(rec["NMI"])),
            "MarkerHits": rec["MarkerHits"], "CoverageScore": repr(float(rec["CoverageScore"])),
            "train_seconds": rec.get("meta", {}).get("seconds", ""),
            "selected_indices": ";".join(str(int(i)) for i in rec["selected_indices"]),
        })
    write_csv(out_csv, rows, PER_RUN_FIELDS)
    return rows


#: Record fields compared bitwise by ``--check``.
CHECK_KEYS = ("unit", "seed", "ARI", "NMI", "MarkerHits", "CoverageScore", "selected_indices")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units", default="all", help='"all", "0,5" or "0-4" (unit i uses bootstrap seed 20000+997i)')
    ap.add_argument("--data-dir", default=None, help="data directory (default: <package>/data)")
    ap.add_argument("--out", default="results/pbmc", help="output directory (relative to the package root)")
    ap.add_argument("--device", default="cuda", help="cuda (default, as in the paper) | cpu | auto")
    ap.add_argument("--skip-existing", action="store_true", help="reuse units/NNN.json already in --out")
    ap.add_argument("--check", action="store_true",
                    help="compare every unit bitwise with the shipped records in reference/pbmc/units")
    ap.add_argument("--eval-threads", type=int, default=EVAL_THREADS,
                    help="BLAS/OpenMP threads for PCA + k-means (default 1; 0 = library default, as in the "
                         "original runs)")
    a = ap.parse_args(argv)

    from ssrk.selfrecon import resolve_device

    data_dir = resolve_path(a.data_dir, DEFAULT_DATA_DIR)
    out = resolve_path(a.out, Path())
    units_dir = out / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    units = parse_units(a.units, len(PBMC_BOOTSTRAP_SEEDS))
    device = resolve_device(a.device)
    config = pbmc_config()

    t_all = time.perf_counter()
    X, labels, genes = load_pbmc(data_dir)
    marker_groups, all_markers = pbmc_marker_indices(genes)
    timings = {}
    for unit in units:
        rec_path = units_dir / f"{unit:03d}.json"
        if a.skip_existing and rec_path.exists():
            cached = load_json(rec_path)
            if int(cached["seed"]) != PBMC_BOOTSTRAP_SEEDS[unit]:
                raise RuntimeError(f"{rec_path}: seed {cached['seed']} != {PBMC_BOOTSTRAP_SEEDS[unit]}")
            continue
        t0 = time.perf_counter()
        rec, W, Wall = run_unit(unit, X, labels, marker_groups, all_markers, config, device, a.eval_threads)
        np.save(units_dir / f"{unit:03d}_W.npy", W)
        np.save(units_dir / f"{unit:03d}_Wall.npy", Wall)
        rec_path.write_text(json.dumps(rec), encoding="utf-8")
        timings[f"{unit:03d}"] = time.perf_counter() - t0
        shown = {k: round(rec[k], 4) for k in ("ARI", "NMI", "CoverageScore")}
        print(f"pbmc unit {unit:2d} seed {rec['seed']} {shown} MarkerHits={rec['MarkerHits']} "
              f"train {rec['meta']['seconds']:.1f}s total {timings[f'{unit:03d}']:.1f}s", flush=True)

    rows = write_per_run(units_dir, out / "per_run.csv")
    summary = {}
    if rows:
        for k in ("ARI", "NMI", "MarkerHits", "CoverageScore"):
            summary[k] = float(np.mean([float(r[k]) for r in rows]))
    checks = {}
    if a.check:
        ref_dir = REFERENCE_DIR / "pbmc" / "units"
        for unit in units:
            res = compare_unit_with_reference(units_dir, ref_dir, unit, CHECK_KEYS)
            checks[f"{unit:03d}"] = all(res.values())
            print(f"check unit {unit:2d}: {'IDENTICAL' if checks[f'{unit:03d}'] else 'DIFFERENT'} {res}")
        if len(rows) == len(PBMC_BOOTSTRAP_SEEDS):
            expected = load_json(REFERENCE_DIR / "pbmc" / "expected_ssrk_means.json")["means"]
            checks["means_equal_paper_records"] = all(summary[k] == expected[k] for k in expected)
            print("check 20-unit means vs paired records:", checks["means_equal_paper_records"], expected)
    write_manifest(out, {
        "protocol": "pbmc_matched",
        "units_run": units,
        "seeds": {f"{u:03d}": PBMC_BOOTSTRAP_SEEDS[u] for u in units},
        "config": config.to_dict(),
        "reference": REFERENCE,
        "restarts": RESTARTS,
        "k": PBMC_K,
        "eval_seed_offset": EVAL_SEED_OFFSET,
        "eval_threads": a.eval_threads,
        "device": str(device),
        "data_file": display_path(Path(data_dir) / "pbmc" / "pbmc3k_processed.h5ad"),
        "unit_seconds": timings,
        "total_seconds": time.perf_counter() - t_all,
        "n_units_in_dir": len(rows),
        "means_over_units_in_dir": summary,
        "checks": checks,
    })
    print("means over", len(rows), "units:", {k: round(v, 4) for k, v in summary.items()})
    if checks and not all(checks.values()):
        print("REFERENCE CHECK FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
