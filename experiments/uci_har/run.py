"""UCI HAR matched protocol, SSRK side (paper Sec. 5.5 and App. A.3.3 "Matched comparison"; UCI HAR rows
of Table A3).

Each of the 20 units appends 561 permuted diagnostic columns to the official training split
(``default_rng(seed + 7)``), draws ONE within-column permutation reference for the augmented matrix
(seed = unit seed), fits the self-reconstruction mode with 3 restarts on that single paired design
(restart seeds ``seed + 100003 r``), averages the signed statistics (``W`` = coordinatewise mean), and
reports

* the top 384 of the 561 real features (``selected_indices``) and the top 384 of all 1,122 columns
  (``calibration_topk_indices``);
* HNI = fraction of appended (known-null) columns in the all-column top 384;
* DomainBalance = time/frequency-domain selection-rate imbalance of the real top 384;
* single-seed MLP Accuracy / macro-F1 (evaluator seed = unit seed + 41, single-thread BLAS), as stored
  in the original records. The paper reports the five-seed means computed by ``evaluate.py``.

Usage (from the package root)::

    python experiments/uci_har/run.py --units all          # 20 units, ~75-140 s each on an RTX 4090
    python experiments/uci_har/evaluate.py --workers 4     # five-seed evaluator -> per_run.csv

Outputs in ``--out``: ``units/NNN.json`` (unit, seed, Accuracy, F1_macro, HNI, DomainBalance,
selected_indices, calibration_topk_indices, meta), ``units/NNN_W.npy`` (float32 (1122,)),
``units/NNN_Wall.npy`` (float32 (3, 1122)), ``manifest.json``.

The post-fit part that needs no data (both top-384 selections, HNI, DomainBalance) is
:func:`rank_unit`, which ``experiments/uci_har/rescore.py`` applies to the shipped ``W`` on the CPU
(Level A of ``verify.py``). Importing this module does not import torch; :func:`main` imports it only
for the fit.
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

import numpy as np  # noqa: E402

from experiments.common import (  # noqa: E402
    DEFAULT_DATA_DIR,
    display_path,
    REFERENCE_DIR,
    compare_unit_with_reference,
    load_json,
    parse_units,
    resolve_path,
    write_manifest,
)
from experiments.uci_har.data import (  # noqa: E402
    append_permuted_nulls,
    load_feature_names,
    load_uci_har,
    uci_domain_balance,
)
from experiments.uci_har.evaluate import EVAL_OFFSETS, REPEAT_SEEDS, evaluate_uci  # noqa: E402

#: Selection budget (top-k real features).
UCI_K = 384
#: Number of real features; appended diagnostic columns have indices 561..1121.
N_REAL = 561
#: Appended permuted columns per real column, and the seed offset of their generator.
APPENDED_NULL_RATIO = 1.0
APPENDED_SEED_OFFSET = 7
#: Reference (within-column permutation) and restarts on one reference draw.
REFERENCE = "permutation"
RESTARTS = 3


def uci_config():
    """Resolved SSRK configuration of the UCI HAR matched protocol (record set ``EVAL_uci_symR3``).

    Resolved from the original configuration machinery (``configs/paper.yaml`` real_data.uci_har
    overrides and the run overrides ``{"gate_parameterization": "symmetric", "restarts": 3}``).
    """
    from ssrk.selfrecon import SelfReconConfig

    return SelfReconConfig(
        encoder_dims=(512, 256, 128),
        latent_dim=64,
        decoder_dims=(128, 256, 512),
        temperature=0.82,
        use_batchnorm=True,
        epochs=100,
        batch_size=256,
        lr=3e-4,
        lr_gate_factor=0.07,
        lambda_entropy=0.0025,
        mask_prob=0.4,
        stage1_frac=0.48,
        freeze_gates_stage1=True,
        entropy_weighting="gap",
        gate_parameterization="symmetric",
    )


def rank_unit(W: np.ndarray, names, nulls: set) -> dict:
    """Data-free scoring of the statistics ``W`` (1,122 columns) of one unit (no torch).

    ``selected_indices`` = top 384 of the 561 real features (the evaluated ranking),
    ``calibration_topk_indices`` = top 384 of all 1,122 columns, ``HNI`` = share of the appended columns
    ``nulls`` in the latter, ``DomainBalance`` of the former (feature names ``names``).
    """
    from ssrk.ranking import deterministic_top_k

    sel = deterministic_top_k(W, UCI_K, np.arange(N_REAL))
    sel_all = deterministic_top_k(W, UCI_K)
    return dict(HNI=len(set(sel_all) & nulls) / UCI_K, DomainBalance=uci_domain_balance(sel, names),
                selected_indices=sel, calibration_topk_indices=sel_all)


def unit_record(unit: int, seed: int, accuracy: float, f1_macro: float, ranked: dict) -> dict:
    """Unit record in the field order of the original records (``meta`` is appended by the caller)."""
    return dict(unit=unit, seed=seed, Accuracy=accuracy, F1_macro=f1_macro, HNI=ranked["HNI"],
                DomainBalance=ranked["DomainBalance"], selected_indices=ranked["selected_indices"],
                calibration_topk_indices=ranked["calibration_topk_indices"])


def run_unit(unit: int, Xtr, ytr, Xte, yte, names, config, device) -> tuple:
    """One matched-protocol unit; returns ``(record, W, Wall)``."""
    from threadpoolctl import threadpool_limits

    from ssrk.selfrecon import ssrk_scores

    seed = REPEAT_SEEDS[unit]
    X_aug, nulls = append_permuted_nulls(Xtr.astype(np.float32), APPENDED_NULL_RATIO, seed + APPENDED_SEED_OFFSET)
    # SSRK fit seed = unit seed (method offset 0 of the matched protocol)
    W, Wall, meta = ssrk_scores(X_aug, seed, REFERENCE, config, device, restarts=RESTARTS)
    ranked = rank_unit(W, names, nulls)
    with threadpool_limits(limits=1):
        acc, f1 = evaluate_uci(Xtr, ytr, Xte, yte, ranked["selected_indices"], seed + EVAL_OFFSETS[0])
    rec = unit_record(unit, seed, acc, f1, ranked)
    rec["meta"] = meta
    return rec, W, Wall


#: Record fields compared bitwise by ``--check``.
CHECK_KEYS = ("unit", "seed", "Accuracy", "F1_macro", "HNI", "DomainBalance", "selected_indices",
              "calibration_topk_indices")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units", default="all", help='"all", "0,7" or "0-4" (unit i uses seed 31000+997i)')
    ap.add_argument("--data-dir", default=None, help="data directory (default: <package>/data)")
    ap.add_argument("--out", default="results/uci_har", help="output directory (relative to the package root)")
    ap.add_argument("--device", default="cuda", help="cuda (default, as in the paper) | cpu | auto")
    ap.add_argument("--skip-existing", action="store_true", help="reuse units/NNN.json already in --out")
    ap.add_argument("--check", action="store_true",
                    help="compare every unit bitwise with the shipped records in reference/uci_har/units")
    a = ap.parse_args(argv)

    from ssrk.selfrecon import resolve_device

    data_dir = resolve_path(a.data_dir, DEFAULT_DATA_DIR)
    out = resolve_path(a.out, Path())
    units_dir = out / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    units = parse_units(a.units, len(REPEAT_SEEDS))
    device = resolve_device(a.device)
    config = uci_config()

    t_all = time.perf_counter()
    Xtr, ytr, Xte, yte = load_uci_har(data_dir)
    names = load_feature_names(data_dir)
    timings = {}
    for unit in units:
        rec_path = units_dir / f"{unit:03d}.json"
        if a.skip_existing and rec_path.exists():
            cached = load_json(rec_path)
            if int(cached["seed"]) != REPEAT_SEEDS[unit]:
                raise RuntimeError(f"{rec_path}: seed {cached['seed']} != {REPEAT_SEEDS[unit]}")
            continue
        t0 = time.perf_counter()
        rec, W, Wall = run_unit(unit, Xtr, ytr, Xte, yte, names, config, device)
        np.save(units_dir / f"{unit:03d}_W.npy", W)
        np.save(units_dir / f"{unit:03d}_Wall.npy", Wall)
        rec_path.write_text(json.dumps(rec), encoding="utf-8")
        timings[f"{unit:03d}"] = time.perf_counter() - t0
        shown = {k: round(rec[k], 4) for k in ("Accuracy", "F1_macro", "HNI", "DomainBalance")}
        print(f"uci_har unit {unit:2d} seed {rec['seed']} {shown} train {rec['meta']['seconds']:.1f}s "
              f"total {timings[f'{unit:03d}']:.1f}s", flush=True)

    checks = {}
    if a.check:
        ref_dir = REFERENCE_DIR / "uci_har" / "units"
        for unit in units:
            res = compare_unit_with_reference(units_dir, ref_dir, unit, CHECK_KEYS)
            checks[f"{unit:03d}"] = all(res.values())
            print(f"check unit {unit:2d}: {'IDENTICAL' if checks[f'{unit:03d}'] else 'DIFFERENT'} {res}")

    write_manifest(out, {
        "protocol": "uci_har_matched",
        "units_run": units,
        "seeds": {f"{u:03d}": REPEAT_SEEDS[u] for u in units},
        "config": config.to_dict(),
        "reference": REFERENCE,
        "restarts": RESTARTS,
        "restart_seeds": "fit_seed + 100003 * r",
        "k": UCI_K,
        "appended_null_ratio": APPENDED_NULL_RATIO,
        "appended_seed_offset": APPENDED_SEED_OFFSET,
        "single_seed_eval_offset": EVAL_OFFSETS[0],
        "device": str(device),
        "data_dir": display_path(data_dir),
        "unit_seconds": timings,
        "total_seconds": time.perf_counter() - t_all,
        "note": "Accuracy/F1_macro in units/*.json are single-seed; run evaluate.py for the five-seed means.",
        "checks": checks,
    })
    if checks and not all(checks.values()):
        print("REFERENCE CHECK FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
