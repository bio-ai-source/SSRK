#!/usr/bin/env python3
"""Exact Gaussian fixed-target study (Sec. 5.2, App. A.4) -- experiment runner.

Run from the package root, e.g. ``python experiments/exact_gaussian/run.py main``.
Results go to ``results/exact_gaussian/<tag>/`` (``--out`` changes the root).

Subcommands
-----------
``main``      The paper's evaluation (Table 1, Table A5, Figure A1):
              ``eval_main`` (a = 0.30, 0.40, 0.50, 0.75 on 50 signal datasets + 100
              all-null datasets) and ``eval_low`` (a = 0.15, 0.20, no all-null),
              both with the locked configuration (R = 20), then merged into
              ``eval_all``.  ``--only main,low,merge`` runs a subset.
``ablation``  Component ablation (Table A6): R = 1, R = 5, no Stage-II
              sharpening, no entropy term, no visible-target context, at a = 0.20,
              0.30 on the evaluation seeds (``--only R1,R5,...``).
``tune``      Predeclared tuning grid (Table A4) on the disjoint tuning seeds 5000-5019 /
              6000-6019 at a = 0.20, 0.30; writes the row files,
              ``tuning_summary.csv`` and ``selection.json`` (selection rule applied).
``audit``     100 paired trajectories x {symmetric, naive} x {float32, float64}
              (App. A.4, "Swap checks and compute").
``eval``      Generic evaluation with any amplitudes / restarts / overrides.
``compare``   Bitwise comparison of a result directory with a reference directory.
``rescore``   CPU only, no torch: rebuild every record that follows from the stored
              statistics W (knockoff+ selections, FDP/power, bootstrap intervals,
              Clopper-Pearson bounds, the eval_all merge, the tuning summary and
              selection, the swap-audit summary) and compare it byte for byte with the
              source records (exit status 1 on any difference).

Every evaluation directory contains ``per_seed_q.csv`` (one row per dataset and
level), ``summary.csv`` (cells with bootstrap CIs), ``W_per_seed.npz`` (aggregate
and per-restart statistics), ``per_seed_info.json`` and ``manifest.json``
(configuration, seeds, chunk layout, swap checks, environment, timings).

Equivalences (the batched member layout -- units in order, at most 640 members per
call, audited units in their own chunks -- is fixed by the arguments, so these give
bitwise-identical records):

    main --only low     ==  eval --amplitudes 0.15,0.2 --restarts 20 --config-key T04_R20_lr2e-2_e150
                                 --tag eval_low --no-null
    ablation --only R1  ==  eval --amplitudes 0.2,0.3 --restarts 1 --config-key ABL_R1 --tag ablation_R1 --no-null

Check a run against the shipped records:

    python experiments/exact_gaussian/run.py compare results/exact_gaussian/eval_low reference/exact_gaussian/eval_low

Recompute everything that follows from the shipped W (Level A of verify.py), then the tables:

    python experiments/exact_gaussian/run.py rescore --src reference/exact_gaussian --dst results/rescored
    python experiments/exact_gaussian/make_tables.py --results results/rescored --check --compare-timings

The training subcommands need torch and, for bitwise agreement with the records,
CUDA (``--device cuda``, the default; the device is checked before any output
directory is created).  ``compare``, ``rescore`` and ``main --only merge`` import no
torch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from ssrk.determinism import enforce_cublas_workspace  # noqa: E402  (imports no torch)

# Deterministic cuBLAS workspace, set before torch / cuBLAS initialisation (torch is
# imported only by the training subcommands); a different pre-set value is overridden
# with a warning.  The effective value is recorded as env.cublas_workspace in every manifest.
enforce_cublas_workspace()

import numpy as np  # noqa: E402

from experiments.common import display_path  # noqa: E402  (torch-free)
from experiments.exact_gaussian.protocol import (  # noqa: E402
    ABLATION_AMPLITUDES, ABLATIONS, EVAL_ALGORITHM_BASE, EVAL_ALL_TAG, EVAL_LOW, EVAL_MAIN,
    EVAL_NULL_AUDIT_EVERY, EVAL_SIGNAL_AUDIT_EVERY, LOCKED_CONFIG_KEY, LOCKED_OVERRIDES, LOCKED_RESTARTS,
    PROTOCOL, SELECTION_RULE, TRAJECTORY_ALGORITHM_BASE, TUNE_ALGORITHM_BASE, TUNING_AMPLITUDES,
    TUNING_GRID, TUNING_NULL_SEEDS, TUNING_SIGNAL_SEEDS, algorithm_seeds, build_units, fdp_power,
    make_dataset, run_units, select_configuration, summarize, trajectory_inputs, unit_info, unit_rows,
    write_csv,
)
from ssrk.ft_config import FixedTargetConfig  # noqa: E402
from ssrk.knockoff_plus import knockoff_plus  # noqa: E402

DEFAULT_OUT = PKG / "results" / "exact_gaussian"
REFERENCE = PKG / "reference" / "exact_gaussian"
AUDIT_TAG = "swap_audit"
TUNING_TAG = "tuning"
AUDIT_COMBOS = [(par, dt) for par in ("symmetric", "naive") for dt in ("float32", "float64")]
# Evaluation and ablation directories whose records are rebuilt from W by ``rescore``.
RESCORE_EVAL_TAGS = [EVAL_MAIN["tag"], EVAL_LOW["tag"]] + [f"ablation_{k}" for k in ABLATIONS]


# ----------------------------------------------------------------------------
# Environment record and device check
# ----------------------------------------------------------------------------

def env_record() -> dict:
    """Software/hardware record stored in every manifest."""
    import pandas
    import scipy
    import torch
    return {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic": True, "tf32": False, "command": " ".join(sys.argv),
    }


def check_device(device: str) -> None:
    """Fail with a clear message (exit status 1) unless ``device`` can train.

    Called before any output directory is created.  ``device`` stays a string
    downstream (it is written into the manifests).
    """
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"error: the training subcommands need torch ({exc}); install requirements.txt "
                         "(rescore, compare and 'main --only merge' run without torch)")
    try:
        dev = torch.device(device)
    except RuntimeError as exc:
        raise SystemExit(f"error: invalid --device {device!r}: {exc}")
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"error: --device {device} requested but CUDA is not available (the records were "
                             "produced on CUDA; --device cpu runs but does not match them bitwise)")
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise SystemExit(f"error: --device {device}: only {torch.cuda.device_count()} CUDA device(s) visible")
    elif dev.type == "cpu":
        print("note: --device cpu; the records were produced on CUDA and are not reproduced bitwise on CPU",
              file=sys.stderr, flush=True)


def _write_json(path: Path, obj, indent: int | None = 2, newline: str | None = None) -> None:
    path.write_text(json.dumps(obj, indent=indent), encoding="utf-8", newline=newline)


def run_diagnostics(records: list[dict]) -> dict:
    """Swap-check totals and training diagnostics of a run, as stored in its manifest.

    ``records`` are the trained units (:func:`mode_eval`) or, equivalently, the
    per-dataset records of ``per_seed_info.json`` (:func:`rescore_eval`); both carry
    ``swap_audit`` (audited datasets only), ``cap_projections``,
    ``logit_clamp_active`` and ``max_frobenius``.
    """
    audits = [r["swap_audit"] for r in records if "swap_audit" in r]
    return {
        "swap_audits": {
            "count": len(audits),
            "subset_sizes": [a["subset_size"] for a in audits],
            "total_per_run_bitwise_mismatches": int(sum(a["per_run_bitwise_mismatches"] for a in audits)),
            "max_per_run_abs_error": max(a["per_run_max_abs_error"] for a in audits) if audits else None,
            "total_aggregate_bitwise_mismatches": int(sum(a["aggregate_bitwise_mismatches"] for a in audits)),
            "all_same_selection": all(a["same_selection_all_q"] for a in audits),
        },
        "cap_projections_total": int(sum(r["cap_projections"] for r in records)),
        "logit_clamp_active_total": int(sum(r["logit_clamp_active"] for r in records)),
        "max_frobenius_over_runs": np.max(np.array([r["max_frobenius"] for r in records]), axis=0).tolist(),
    }


# ----------------------------------------------------------------------------
# eval
# ----------------------------------------------------------------------------

def mode_eval(out_root: Path, device: str, signals: list[float], overrides: dict, R: int, tag: str,
              config_key: str, with_null: bool = True) -> dict:
    """Evaluate one configuration on the 50 evaluation signal datasets per amplitude
    (and the 100 all-null datasets unless ``with_null`` is False), with in-batch
    bitwise swap checks on every fifth signal / every tenth all-null dataset."""
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)
    cfg = FixedTargetConfig(**overrides)
    units = []
    for a in signals:
        units += build_units(PROTOCOL["signal_seeds"], False, a, R, EVAL_ALGORITHM_BASE, EVAL_SIGNAL_AUDIT_EVERY)
    if with_null:
        units += build_units(PROTOCOL["all_null_seeds"], True, None, R, EVAL_ALGORITHM_BASE, EVAL_NULL_AUDIT_EVERY)
    t0 = time.perf_counter()
    layout, secs = run_units(units, cfg, device)
    rows = [r for u in units for r in unit_rows(u)]
    infos = [unit_info(u) for u in units]
    write_csv(out / "per_seed_q.csv", rows)
    summary = summarize(rows)
    write_csv(out / "summary.csv", summary)
    sigu = [u for u in units if not u["null"]]
    nullu = [u for u in units if u["null"]]
    np.savez_compressed(
        out / "W_per_seed.npz",
        W_signal=np.array([u["W"] for u in sigu]),
        W_signal_runs=np.array([u["W_runs"] for u in sigu]),
        signal_seeds=np.array([u["seed"] for u in sigu]),
        signal_levels=np.array([u["signal"] for u in sigu]),
        supports=np.array([u["support"] for u in sigu]),
        W_all_null=np.array([u["W"] for u in nullu]) if nullu else np.zeros((0, PROTOCOL["p"])),
        W_all_null_runs=np.array([u["W_runs"] for u in nullu]) if nullu else np.zeros((0, R, PROTOCOL["p"])),
        all_null_seeds=np.array([u["seed"] for u in nullu]),
    )
    (out / "per_seed_info.json").write_text(json.dumps(infos), encoding="utf-8")
    diag = run_diagnostics(units)
    manifest = {
        "purpose": "exact Gaussian fixed-target controlled-discovery study with the SSRK gate network (GPU)",
        "protocol": dict(PROTOCOL), "amplitudes": signals, "config_key": config_key,
        "config": cfg.to_dict(), "restarts": R,
        "algorithm_seeds": algorithm_seeds(EVAL_ALGORITHM_BASE, R),
        "aggregation": "coordinatewise mean of W over restarts on the same paired design (odd rule)",
        "chunk_layout": layout, "gpu_seconds": secs, "wall_seconds": time.perf_counter() - t0,
        "swap_audits": diag["swap_audits"],
        "cap_projections_total": diag["cap_projections_total"],
        "logit_clamp_active_total": diag["logit_clamp_active_total"],
        "max_frobenius_over_runs": diag["max_frobenius_over_runs"],
        "run": {"tag": tag, "amplitudes": signals, "restarts": R, "overrides": overrides,
                "with_null": with_null, "device": device},
        "env": env_record(),
    }
    _write_json(out / "manifest.json", manifest)
    for rec in summary:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
    print(f"[{tag}] swap checks:", json.dumps({k: v for k, v in manifest["swap_audits"].items()
                                               if k != "subset_sizes"}))
    print(f"[{tag}] gpu {secs:.0f}s, wall {manifest['wall_seconds']:.0f}s -> {display_path(out)}", flush=True)
    return manifest


# ----------------------------------------------------------------------------
# merge (eval_main + eval_low -> eval_all)
# ----------------------------------------------------------------------------

def merge_evals(out_root: Path, sources: list[str], target: str) -> None:
    """Merge evaluation runs that share the locked configuration.

    Rows are concatenated and sorted by (model, signal, seed, q); the summary is
    recomputed from the merged rows (pandas parsing of the per-run CSVs, as in the
    records used by the paper); W arrays and per-dataset records are concatenated
    in source order.

    Byte identity with the stored ``eval_all`` records depends on pandas' default C
    float parser (``pd.read_csv`` without ``float_precision``), which is *not*
    round-trip: e.g. ``'0.14285714285714285'`` parses to ``0.1428571428571428``.
    The stored ``eval_all/per_seed_q.csv`` and ``summary.csv`` were produced with that
    parser, so it is kept (``float_precision='round_trip'`` would change 72 summary
    cells by at most 1 ulp; the printed tables are unaffected).
    ``tests/test_fixed_target.py`` checks the parser behaviour.
    """
    import pandas as pd

    out = out_root / target
    out.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((out_root / s / "manifest.json").read_text(encoding="utf-8")) for s in sources]
    cfgs = [json.dumps(m["config"], sort_keys=True) for m in manifests]
    assert len(set(cfgs)) == 1, "configurations differ"
    assert len({m["restarts"] for m in manifests}) == 1
    rows = pd.concat([pd.read_csv(out_root / s / "per_seed_q.csv") for s in sources], ignore_index=True)
    rows = rows.sort_values(["model", "signal", "seed", "q"]).reset_index(drop=True)
    rec = rows.to_dict("records")
    write_csv(out / "per_seed_q.csv", rec)
    write_csv(out / "summary.csv", summarize(rec))
    infos = []
    for s in sources:
        infos += json.loads((out_root / s / "per_seed_info.json").read_text(encoding="utf-8"))
    (out / "per_seed_info.json").write_text(json.dumps(infos), encoding="utf-8")
    npzs = [np.load(out_root / s / "W_per_seed.npz") for s in sources]
    cat = {}
    for key in ["W_signal", "W_signal_runs", "signal_seeds", "signal_levels", "supports"]:
        cat[key] = np.concatenate([z[key] for z in npzs if z[key].size])
    for key in ["W_all_null", "W_all_null_runs", "all_null_seeds"]:
        cat[key] = np.concatenate([z[key] for z in npzs if z[key].size])
    np.savez_compressed(out / "W_per_seed.npz", **cat)
    audits = [m["swap_audits"] for m in manifests]
    merged = dict(manifests[0])
    merged.pop("run", None)
    merged["amplitudes"] = sorted({a for m in manifests for a in m["amplitudes"]})
    merged["sources"] = {s: m["amplitudes"] for s, m in zip(sources, manifests)}
    merged["swap_audits"] = {
        "count": sum(a["count"] for a in audits),
        "subset_sizes": [x for a in audits for x in a["subset_sizes"]],
        "total_per_run_bitwise_mismatches": sum(a["total_per_run_bitwise_mismatches"] for a in audits),
        "max_per_run_abs_error": max(a["max_per_run_abs_error"] for a in audits),
        "total_aggregate_bitwise_mismatches": sum(a["total_aggregate_bitwise_mismatches"] for a in audits),
        "all_same_selection": all(a["all_same_selection"] for a in audits),
    }
    merged["cap_projections_total"] = sum(m["cap_projections_total"] for m in manifests)
    merged["logit_clamp_active_total"] = sum(m["logit_clamp_active_total"] for m in manifests)
    merged["max_frobenius_over_runs"] = np.max(np.array([m["max_frobenius_over_runs"] for m in manifests]),
                                               axis=0).tolist()
    merged["gpu_seconds"] = sum(m["gpu_seconds"] for m in manifests)
    merged["wall_seconds"] = sum(m["wall_seconds"] for m in manifests)
    merged["chunk_layout"] = {s: m["chunk_layout"] for s, m in zip(sources, manifests)}
    merged["env"] = {s: m["env"] for s, m in zip(sources, manifests)}
    _write_json(out / "manifest.json", merged)
    print(f"[{target}] merged {len(rec)} rows; amplitudes {merged['amplitudes']}; "
          f"swap checks {merged['swap_audits']['count']}", flush=True)


def mode_main(out_root: Path, device: str, only: list[str]) -> None:
    """The two evaluation invocations of the paper, then the merge."""
    for part, spec in (("main", EVAL_MAIN), ("low", EVAL_LOW)):
        if part in only:
            mode_eval(out_root, device, spec["amplitudes"], dict(LOCKED_OVERRIDES), LOCKED_RESTARTS,
                      spec["tag"], LOCKED_CONFIG_KEY, spec["with_null"])
    if "merge" in only:
        merge_evals(out_root, [EVAL_MAIN["tag"], EVAL_LOW["tag"]], EVAL_ALL_TAG)


def mode_ablation(out_root: Path, device: str, only: list[str]) -> None:
    """Component ablation: one component of the locked configuration changed at a time."""
    for name, (config_key, overrides, R) in ABLATIONS.items():
        if name in only:
            mode_eval(out_root, device, list(ABLATION_AMPLITUDES), dict(overrides), R, f"ablation_{name}",
                      config_key, with_null=False)


# ----------------------------------------------------------------------------
# tune
# ----------------------------------------------------------------------------

def tuning_summary_from_rows(out: Path, keys: list[str], amplitudes: list[float]) -> list[dict]:
    """Summary of the tuning grid computed from the written row files.

    The row files are read back with ``pandas.read_csv`` (default float parser),
    exactly as the stored ``tuning_summary.csv`` was produced, so that the
    summary file is byte-identical to the record.  That parser is not round-trip
    (``'0.14285714285714285'`` parses to ``0.1428571428571428``), so the byte
    identity depends on it; see :func:`merge_evals`.
    """
    import pandas as pd

    summary = []
    for key in keys:
        R = TUNING_GRID[key][1]
        for a in amplitudes:
            rows = pd.read_csv(out / f"{key}_a{a}_rows.csv").to_dict("records")
            s = summarize(rows)
            for rec in s:
                rec.update({"config_key": key, "restarts": R, "tuning_amplitude": a})
            summary.extend(s)
    return summary


def mode_tune(out_root: Path, device: str, amplitudes: list[float], keys: list[str] | None) -> None:
    """Predeclared grid on the tuning seeds (disjoint from evaluation), then the selection rule.

    ``keys`` restricts the fits to part of the grid (the grid can be run in several
    invocations into the same directory).  The summary and the selection are always
    computed over every grid configuration whose row files for all ``amplitudes``
    are present in the output directory; ``complete_grid`` in the manifest says
    whether that is the whole predeclared grid.
    """
    out = out_root / TUNING_TAG
    out.mkdir(parents=True, exist_ok=True)
    run_keys = [k for k in TUNING_GRID if not keys or k in keys]
    manifest_path = out / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    timings = dict(previous.get("timings", {}))
    t_start = time.perf_counter()
    for key in run_keys:
        override, R = TUNING_GRID[key]
        cfg = FixedTargetConfig(**override)
        for a in amplitudes:
            units = build_units(TUNING_SIGNAL_SEEDS, False, a, R, TUNE_ALGORITHM_BASE, 0)
            units += build_units(TUNING_NULL_SEEDS, True, None, R, TUNE_ALGORITHM_BASE, 0)
            layout, secs = run_units(units, cfg, device)
            rows = [r for u in units for r in unit_rows(u)]
            write_csv(out / f"{key}_a{a}_rows.csv", rows)
            timings[f"{key}_a{a}"] = {"gpu_seconds": secs,
                                      "chunks": [[c["members"], c["audit_chunk"]] for c in layout]}
            s = summarize(rows)
            r10 = [x for x in s if x["model"] == "signal" and abs(x["q"] - 0.10) < 1e-9][0]
            n10 = [x for x in s if x["model"] == "all_null" and abs(x["q"] - 0.10) < 1e-9][0]
            print(f"{key} a={a}: power@.05/.10/.15 = " + "/".join(
                f"{[x for x in s if x['model'] == 'signal' and abs(x['q'] - q) < 1e-9][0]['power_mean']:.3f}"
                for q in (0.05, 0.10, 0.15))
                + f"  FDP@.10={r10['FDP_mean']:.3f}  null nonempty@.10={n10['nonempty_count']}  ({secs:.0f}s)",
                flush=True)
    summary_keys = [k for k in TUNING_GRID if all((out / f"{k}_a{a}_rows.csv").exists() for a in amplitudes)]
    summary = tuning_summary_from_rows(out, summary_keys, amplitudes)
    write_csv(out / "tuning_summary.csv", summary)
    selection = select_configuration(summary, summary_keys)
    _write_json(out / "selection.json", selection)
    complete = summary_keys == list(TUNING_GRID)
    _write_json(manifest_path, {
        "tuning_signal_seeds": TUNING_SIGNAL_SEEDS, "tuning_null_seeds": TUNING_NULL_SEEDS,
        "algorithm_base": TUNE_ALGORITHM_BASE, "grid": {k: [v[0], v[1]] for k, v in TUNING_GRID.items()},
        "amplitudes": amplitudes, "selection_rule": SELECTION_RULE, "selection": selection,
        "keys_in_summary": summary_keys, "complete_grid": complete,
        "invocations": previous.get("invocations", []) + [{"keys_run": run_keys,
                                                           "seconds": time.perf_counter() - t_start,
                                                           "env": env_record()}],
        "timings": timings,
    })
    print("selection:", json.dumps(selection), flush=True)
    if not complete:
        print(f"note: summary/selection cover {len(summary_keys)}/{len(TUNING_GRID)} grid configurations "
              f"({', '.join(summary_keys)}); run the remaining keys into the same directory")


# ----------------------------------------------------------------------------
# audit
# ----------------------------------------------------------------------------

def audit_summary(rows: list[dict], combos: list[tuple[str, str]], epochs: int) -> list[dict]:
    """``swap_audit/summary.csv``: per (parameterization, dtype), the distribution of the
    per-trajectory max |W(swap_S D) - T_S W(D)| and the selection changes at q = 0.10."""
    summ = []
    for par, dt in combos:
        g = [r for r in rows if r["parameterization"] == par and r["dtype"] == dt]
        e = np.array([r["max_abs_error"] for r in g])
        summ.append({"parameterization": par, "dtype": dt, "trajectories": len(g), "epochs": epochs,
                     "max_abs_error_max": float(e.max()), "max_abs_error_median": float(np.median(e)),
                     "max_abs_error_q90": float(np.quantile(e, 0.9)),
                     "exact_trajectories": int(np.sum(e == 0)),
                     "selection_changed_q10": int(sum(r["selection_changed_q10"] for r in g))})
    return summ


def mode_audit(out_root: Path, device: str, trajectories: int, epochs: int,
               combos: list[tuple[str, str]]) -> None:
    """Paired trajectories: each with a fresh dataset (a = 0.75), a swap subset of
    uniform size in {1,...,80} and a single fit; the swapped copy is trained in a
    second batched call with the identical member layout."""
    from ssrk.fixed_target import fit_fixed_target_batched

    out = out_root / AUDIT_TAG
    out.mkdir(parents=True, exist_ok=True)
    data = [trajectory_inputs(k) for k in range(trajectories)]
    seeds = [TRAJECTORY_ALGORITHM_BASE + k for k in range(trajectories)]
    rows, timings = [], {}
    for par, dt in combos:
        cfg = FixedTargetConfig(epochs=epochs, parameterization=par, dtype=dt)
        Vs, Vts, V2s, Vt2s, Ys = [], [], [], [], []
        for V, Vt, Y, S in data:
            V2, Vt2 = V.copy(), Vt.copy()
            V2[:, S], Vt2[:, S] = Vt[:, S], V[:, S]
            Vs.append(V); Vts.append(Vt); V2s.append(V2); Vt2s.append(Vt2); Ys.append(Y)
        fit = fit_fixed_target_batched(np.stack(Vs), np.stack(Vts), np.stack(Ys), cfg, seeds, device)
        fit2 = fit_fixed_target_batched(np.stack(V2s), np.stack(Vt2s), np.stack(Ys), cfg, seeds, device)
        timings[f"{par}_{dt}"] = {"gpu_seconds_original": fit.seconds, "gpu_seconds_swapped": fit2.seconds}
        for k, (_, _, _, S) in enumerate(data):
            W, W2 = fit.W[k], fit2.W[k]
            mask = np.zeros(PROTOCOL["p"], bool)
            mask[S] = True
            expected = np.where(mask, -W, W)
            err = np.abs(W2 - expected)
            rows.append({
                "trajectory": k, "parameterization": par, "dtype": dt, "epochs": epochs,
                "subset_size": int(mask.sum()), "max_abs_error": float(err.max()),
                "swapped_max_abs_error": float(err[mask].max()),
                "nonswapped_max_abs_error": float(err[~mask].max()) if (~mask).any() else 0.0,
                "bitwise_mismatches": int(np.sum(W2 != expected)),
                "selection_changed_q10": int(knockoff_plus(W2, 0.10)[1] != knockoff_plus(expected, 0.10)[1]),
            })
        g = [r for r in rows if r["parameterization"] == par and r["dtype"] == dt]
        e = np.array([r["max_abs_error"] for r in g])
        print(json.dumps({"parameterization": par, "dtype": dt, "max": float(e.max()),
                          "median": float(np.median(e)), "exact": int(np.sum(e == 0)),
                          "selection_changed_q10": int(sum(r["selection_changed_q10"] for r in g)),
                          "gpu_seconds": round(fit.seconds + fit2.seconds, 1)}), flush=True)
    write_csv(out / "paired_trajectories.csv", rows)
    write_csv(out / "summary.csv", audit_summary(rows, combos, epochs))
    _write_json(out / "manifest.json", {"trajectories": trajectories, "epochs": epochs,
                                        "combos": [f"{p}:{d}" for p, d in combos],
                                        "timings": timings, "env": env_record()})


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------

MANIFEST_FIELDS = ["amplitudes", "config_key", "config", "restarts", "algorithm_seeds", "swap_audits",
                   "cap_projections_total", "logit_clamp_active_total", "max_frobenius_over_runs", "protocol"]


def _layout_no_time(layout):
    """Chunk layout without timings; a merged layout (dict keyed by source run) is
    compared in source order, independent of the source-directory names."""
    if isinstance(layout, dict):
        return [_layout_no_time(v) for v in layout.values()]
    return [{k: v for k, v in c.items() if k != "seconds"} for c in layout]


PROTOCOL_DOC_FIELDS = ("tuning_seeds", "tuning_null_seeds", "tuning_note")


def _protocol(p: dict | None) -> dict | None:
    """Protocol parameters used by an evaluation run.

    The tuning-seed lists inside an evaluation manifest are documentary (the
    stored ``eval_main``/``eval_low`` records list the earlier CPU-screen seeds
    5000-5009 / 6000-6009 there, the merged record the GPU tuning seeds
    5000-5019 / 6000-6019 that were actually used); they do not enter any
    evaluation computation and are excluded from the comparison.
    """
    return None if p is None else {k: v for k, v in p.items() if k not in PROTOCOL_DOC_FIELDS}


def compare_dirs(run: Path, ref: Path) -> dict:
    """Bitwise comparison of a result directory with a reference directory.

    Evaluation dirs: every array of ``W_per_seed.npz`` (dtype, shape, values),
    ``per_seed_q.csv`` and ``summary.csv`` byte for byte, ``per_seed_info.json``
    as parsed JSON, and the deterministic manifest fields (timings, environment
    and command are not compared).  Tuning dirs: every row file present in
    ``run`` and ``tuning_summary.csv`` byte for byte, ``selection.json`` parsed.
    Audit dirs: the rows of ``paired_trajectories.csv`` / ``summary.csv`` present
    in ``run``.
    """
    rep: dict = {"run": str(run), "reference": str(ref), "checks": {}}
    c = rep["checks"]

    def same_bytes(name):
        c[name] = (run / name).read_bytes() == (ref / name).read_bytes()

    if (ref / "W_per_seed.npz").exists():
        a, b = np.load(run / "W_per_seed.npz"), np.load(ref / "W_per_seed.npz")
        c["npz_keys"] = sorted(a.files) == sorted(b.files)
        for k in b.files:
            c[f"npz:{k}"] = bool(k in a.files and a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                                 and np.array_equal(a[k], b[k]))
        for name in ("per_seed_q.csv", "summary.csv"):
            same_bytes(name)
        c["per_seed_info.json"] = (json.loads((run / "per_seed_info.json").read_text(encoding="utf-8"))
                                   == json.loads((ref / "per_seed_info.json").read_text(encoding="utf-8")))
        ma = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        mb = json.loads((ref / "manifest.json").read_text(encoding="utf-8"))
        for f in MANIFEST_FIELDS:
            if f == "protocol" and f in mb:
                c["manifest:protocol"] = _protocol(ma.get(f)) == _protocol(mb[f])
            elif f in mb:
                c[f"manifest:{f}"] = ma.get(f) == mb[f]
        if "sources" in mb:
            c["manifest:sources(amplitudes)"] = list(ma.get("sources", {}).values()) == list(mb["sources"].values())
        c["manifest:chunk_layout"] = _layout_no_time(ma["chunk_layout"]) == _layout_no_time(mb["chunk_layout"])
    elif (ref / "tuning_summary.csv").exists():
        for f in sorted(run.glob("*_rows.csv")):
            same_bytes(f.name)
        if all(f"{k}_a{a}_rows.csv" in {p.name for p in run.glob('*_rows.csv')}
               for k in TUNING_GRID for a in TUNING_AMPLITUDES):
            same_bytes("tuning_summary.csv")
            c["selection.json"] = (json.loads((run / "selection.json").read_text(encoding="utf-8"))
                                   == json.loads((ref / "selection.json").read_text(encoding="utf-8")))
    elif (ref / "paired_trajectories.csv").exists():
        ra = list(csv.DictReader((run / "paired_trajectories.csv").open(encoding="utf-8")))
        rb = list(csv.DictReader((ref / "paired_trajectories.csv").open(encoding="utf-8")))
        combos = {(r["parameterization"], r["dtype"]) for r in ra}
        rb = [r for r in rb if (r["parameterization"], r["dtype"]) in combos]
        c["paired_trajectories.csv(rows present)"] = ra == rb
        sa = list(csv.DictReader((run / "summary.csv").open(encoding="utf-8")))
        sb = [r for r in csv.DictReader((ref / "summary.csv").open(encoding="utf-8"))
              if (r["parameterization"], r["dtype"]) in combos]
        c["summary.csv(rows present)"] = sa == sb
        if len(combos) == len(AUDIT_COMBOS):
            same_bytes("paired_trajectories.csv")
            same_bytes("summary.csv")
    rep["all_identical"] = bool(c) and all(bool(v) for v in c.values())
    return rep


# ----------------------------------------------------------------------------
# rescore (CPU, no torch): everything that follows from the stored W
# ----------------------------------------------------------------------------

def _same_bytes(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()


def _float32_valued(x: np.ndarray) -> bool:
    """True if every entry of the float64 array ``x`` is a float32 number."""
    return bool(np.array_equal(x.astype(np.float32).astype(np.float64), x))


_SUPPORTS: dict[int, list[int]] = {}


def protocol_support(seed: int) -> list[int]:
    """Support of signal dataset ``seed`` regenerated by :func:`make_dataset` (amplitude-free)."""
    if seed not in _SUPPORTS:
        _SUPPORTS[seed] = make_dataset(seed, null_model=False)[4]
    return _SUPPORTS[seed]


def rescore_eval(src: Path, dst: Path) -> dict[str, bool]:
    """Rebuild one evaluation / ablation record directory from its ``W_per_seed.npz``.

    The per-restart statistics ``W_runs`` are aggregated exactly as in
    :func:`run_units` (float64 ``mean`` over restarts of the trainer's float32
    statistics stored as float64; checked bitwise against the stored aggregate),
    then knockoff+ (eq. (8)) at q = 0.05, 0.10, 0.15, FDP and power against the
    stored supports (checked against the protocol's ``make_dataset``), and the
    cell summaries (20,000-draw percentile bootstrap, Clopper-Pearson bound) are
    recomputed and written to ``dst``.  ``W_per_seed.npz``, ``per_seed_info.json``
    and ``manifest.json`` are copied; the manifest's swap-check totals and
    diagnostics are re-derived from the per-dataset records.
    """
    c: dict[str, bool] = {}
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("W_per_seed.npz", "per_seed_info.json", "manifest.json"):
        shutil.copyfile(src / name, dst / name)
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    infos = json.loads((src / "per_seed_info.json").read_text(encoding="utf-8"))
    with np.load(src / "W_per_seed.npz") as npz:
        z = {k: npz[k] for k in npz.files}
    R = int(man["restarts"])
    units = []
    for i in range(len(z["signal_seeds"])):
        units.append({"seed": int(z["signal_seeds"][i]), "null": False, "signal": float(z["signal_levels"][i]),
                      "W_runs": z["W_signal_runs"][i], "W_stored": z["W_signal"][i],
                      "support": [int(j) for j in z["supports"][i]]})
    for i in range(len(z["all_null_seeds"])):
        units.append({"seed": int(z["all_null_seeds"][i]), "null": True, "signal": 0.0,
                      "W_runs": z["W_all_null_runs"][i], "W_stored": z["W_all_null"][i], "support": []})
    # Same aggregation as run_units: u["W"] = W_runs.mean(axis=0) on the (R, p) float64 array.
    for u in units:
        u["W"] = u["W_runs"].mean(axis=0)

    with_null = bool(len(z["all_null_seeds"]))
    expected_layout = ([(float(a), s, False) for a in man["amplitudes"] for s in PROTOCOL["signal_seeds"]]
                       + ([(0.0, s, True) for s in PROTOCOL["all_null_seeds"]] if with_null else []))
    c["datasets: amplitudes x seeds in protocol order"] = (
        [(u["signal"], u["seed"], u["null"]) for u in units] == expected_layout)
    c[f"W_runs: {R} restarts per dataset"] = all(u["W_runs"].shape == (R, PROTOCOL["p"]) for u in units)
    if man["config"]["dtype"] == "float32":
        c["W_runs: float32 trainer values stored as float64"] = all(
            u["W_runs"].dtype == np.float64 and _float32_valued(u["W_runs"]) for u in units)
    c["W aggregate == float64 mean of W_runs over restarts (bitwise)"] = all(
        u["W"].dtype == u["W_stored"].dtype and np.array_equal(u["W"], u["W_stored"]) for u in units)
    c["supports == make_dataset(seed) supports"] = all(
        u["support"] == protocol_support(u["seed"]) for u in units if not u["null"])
    c["per_seed_info.json agrees with npz (seed, model, signal, support, W)"] = len(infos) == len(units) and all(
        i["seed"] == u["seed"] and i["model"] == ("all_null" if u["null"] else "signal")
        and i["signal"] == u["signal"] and i["support"] == u["support"] and i["W"] == u["W"].tolist()
        for i, u in zip(infos, units))
    diag = run_diagnostics(infos)
    c["manifest swap checks and diagnostics == recomputed from per_seed_info.json"] = all(
        man.get(k) == v for k, v in diag.items())

    rows = [r for u in units for r in unit_rows(u)]
    write_csv(dst / "per_seed_q.csv", rows)
    write_csv(dst / "summary.csv", summarize(rows))
    for name in ("per_seed_q.csv", "summary.csv"):
        c[f"{name} rebuilt: byte-identical"] = _same_bytes(dst / name, src / name)
    for name in ("W_per_seed.npz", "per_seed_info.json", "manifest.json"):
        c[f"{name} copied: byte-identical"] = _same_bytes(dst / name, src / name)
    return c


def rescore_merge(src: Path, dst: Path) -> dict[str, bool]:
    """Rebuild ``eval_all`` from the rebuilt ``eval_main`` / ``eval_low`` by :func:`merge_evals`
    (the operation of ``main --only merge``) and compare it with ``src/eval_all``.

    The data files are compared byte for byte.  The merged ``manifest.json`` is
    compared on its deterministic fields (as :func:`compare_dirs`) and on the
    summed GPU / wall times (used for the compute times of App. A.4); its bytes
    differ from the stored record only in documentary fields (tuning-seed lists,
    source-directory names; see :func:`_protocol`).
    """
    merge_evals(dst, [EVAL_MAIN["tag"], EVAL_LOW["tag"]], EVAL_ALL_TAG)
    s, d = src / EVAL_ALL_TAG, dst / EVAL_ALL_TAG
    c: dict[str, bool] = {}
    for name in ("per_seed_q.csv", "summary.csv", "per_seed_info.json", "W_per_seed.npz"):
        c[f"{name} merged: byte-identical"] = _same_bytes(d / name, s / name)
    rep = compare_dirs(d, s)
    c["manifest.json merged: deterministic fields (compare_dirs)"] = all(
        v for k, v in rep["checks"].items() if k.startswith("manifest:"))
    ma = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((s / "manifest.json").read_text(encoding="utf-8"))
    c["manifest.json merged: gpu_seconds, wall_seconds"] = all(ma[k] == mb[k] for k in ("gpu_seconds", "wall_seconds"))
    return c


def _typed_rows(path: Path, ints=(), floats=()) -> list[dict]:
    """Rows of a csv-module CSV with the named columns converted (``float(repr)`` is exact)."""
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for r in rows:
        for k in ints:
            r[k] = int(r[k])
        for k in floats:
            r[k] = float(r[k])
    return rows


def rescore_tuning(src: Path, dst: Path) -> dict[str, bool]:
    """Rebuild ``tuning_summary.csv`` and ``selection.json`` (Table A4) from the stored
    per-dataset row files (the tuning statistics W were not stored).

    Also checks that, in every row file, the selection sizes, true/false positives,
    FDP, power and the empty flag follow from the stored selections and the
    supports regenerated by :func:`make_dataset`.
    """
    c: dict[str, bool] = {}
    dst.mkdir(parents=True, exist_ok=True)
    keys, amps = list(TUNING_GRID), list(TUNING_AMPLITUDES)
    names = [f"{k}_a{a}_rows.csv" for k in keys for a in amps]
    for name in names + ["manifest.json"]:
        shutil.copyfile(src / name, dst / name)
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    c["manifest: tuning seeds, amplitudes and grid == protocol"] = (
        man["tuning_signal_seeds"] == TUNING_SIGNAL_SEEDS and man["tuning_null_seeds"] == TUNING_NULL_SEEDS
        and man["amplitudes"] == amps and man["grid"] == {k: [v[0], v[1]] for k, v in TUNING_GRID.items()})
    consistent = True
    for name in names:
        for r in csv.DictReader((src / name).open(encoding="utf-8", newline="")):
            sel = [int(j) for j in r["selected"].split()]
            sup = [] if r["model"] == "all_null" else protocol_support(int(r["seed"]))
            fdp, power = fdp_power(sel, sup)
            consistent &= (r["selection_size"] == str(len(sel)) and r["true_positives"] == str(len(set(sel) & set(sup)))
                           and r["false_positives"] == str(len(set(sel) - set(sup))) and r["FDP"] == str(fdp)
                           and r["power"] == str(power) and r["empty"] == str(int(len(sel) == 0)))
    c["row files: sizes, TP, FP, FDP, power, empty follow from selections and supports"] = consistent
    summary = tuning_summary_from_rows(dst, keys, amps)
    write_csv(dst / "tuning_summary.csv", summary)
    selection = select_configuration(summary, keys)
    # The stored JSON records were written in text mode on Windows (CRLF line ends).
    _write_json(dst / "selection.json", selection, newline="\r\n")
    c["tuning_summary.csv rebuilt: byte-identical"] = _same_bytes(dst / "tuning_summary.csv", src / "tuning_summary.csv")
    c["selection.json rebuilt: byte-identical"] = _same_bytes(dst / "selection.json", src / "selection.json")
    c["manifest selection == rebuilt selection"] = man["selection"] == selection
    for name in names + ["manifest.json"]:
        c[f"{name} copied: byte-identical"] = _same_bytes(dst / name, src / name)
    return c


def rescore_audit(src: Path, dst: Path) -> dict[str, bool]:
    """Rebuild ``swap_audit/summary.csv`` from ``paired_trajectories.csv`` (the audit W were not stored)."""
    c: dict[str, bool] = {}
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("paired_trajectories.csv", "manifest.json"):
        shutil.copyfile(src / name, dst / name)
    rows = _typed_rows(src / "paired_trajectories.csv",
                       ints=("trajectory", "epochs", "subset_size", "bitwise_mismatches", "selection_changed_q10"),
                       floats=("max_abs_error", "swapped_max_abs_error", "nonswapped_max_abs_error"))
    combos = list(dict.fromkeys((r["parameterization"], r["dtype"]) for r in rows))
    epochs = sorted({r["epochs"] for r in rows})
    c["one epoch count"] = len(epochs) == 1
    write_csv(dst / "summary.csv", audit_summary(rows, combos, epochs[0]))
    c["summary.csv rebuilt: byte-identical"] = _same_bytes(dst / "summary.csv", src / "summary.csv")
    for name in ("paired_trajectories.csv", "manifest.json"):
        c[f"{name} copied: byte-identical"] = _same_bytes(dst / name, src / name)
    return c


def mode_rescore(src: Path, dst: Path) -> bool:
    """Rebuild every record of the exact Gaussian study that follows from the stored W
    into ``dst`` and compare it with ``src``; returns True if every check passes.

    ``dst`` then holds a complete record set (``make_tables.py --results dst``).
    """
    if dst.resolve() == src.resolve():
        raise SystemExit("error: --dst must differ from --src")
    t0 = time.perf_counter()
    report: dict[str, dict[str, bool]] = {}
    for tag in RESCORE_EVAL_TAGS:
        report[tag] = rescore_eval(src / tag, dst / tag)
    report[EVAL_ALL_TAG] = rescore_merge(src, dst)
    report[TUNING_TAG] = rescore_tuning(src / TUNING_TAG, dst / TUNING_TAG)
    report[AUDIT_TAG] = rescore_audit(src / AUDIT_TAG, dst / AUDIT_TAG)
    n_ok = n_all = 0
    for tag, checks in report.items():
        bad = [k for k, v in checks.items() if not v]
        n_all += len(checks)
        n_ok += len(checks) - len(bad)
        print(f"[{'PASS' if not bad else 'FAIL'}] {tag}: {len(checks) - len(bad)}/{len(checks)} checks"
              + (f"; failed: {'; '.join(bad)}" if bad else ""), flush=True)
    ok = n_ok == n_all
    try:
        src_label = src.resolve().relative_to(PKG).as_posix()
    except ValueError:
        src_label = str(src)
    _write_json(dst / "rescore_report.json", {"source": src_label, "all_identical": ok, "checks": report})
    print(f"rescore: {n_ok}/{n_all} checks passed in {time.perf_counter() - t0:.0f}s "
          f"({'every rebuilt record byte-identical to the source' if ok else 'DIFFERENCES FOUND'}) -> {display_path(dst)}",
          flush=True)
    return ok


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def _names(s: str, allowed) -> list[str]:
    names = [x for x in s.split(",") if x]
    bad = [x for x in names if x not in allowed]
    if bad:
        raise SystemExit(f"unknown names {bad}; choose from {list(allowed)}")
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results root (default results/exact_gaussian)")
    ap.add_argument("--device", default="cuda", help="torch device (the records were produced on CUDA)")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("main", help="paper evaluation: eval_main + eval_low + merge -> eval_all")
    p.add_argument("--only", default="main,low,merge", help="subset of main,low,merge")

    p = sub.add_parser("ablation", help="component ablation (Table A6)")
    p.add_argument("--only", default=",".join(ABLATIONS), help=f"subset of {','.join(ABLATIONS)}")

    p = sub.add_parser("tune", help="predeclared tuning grid + selection rule (Table A4)")
    p.add_argument("--amplitudes", default=",".join(str(a) for a in TUNING_AMPLITUDES))
    p.add_argument("--keys", default="", help="comma-separated subset of the grid (default: all)")

    p = sub.add_parser("audit", help="100 paired trajectories x {symmetric,naive} x {float32,float64}")
    p.add_argument("--trajectories", type=int, default=100)
    p.add_argument("--audit-epochs", type=int, default=150)
    p.add_argument("--combos", default=",".join(f"{a}:{b}" for a, b in AUDIT_COMBOS),
                   help="subset of parameterization:dtype pairs")

    p = sub.add_parser("eval", help="generic evaluation")
    p.add_argument("--amplitudes", default="0.2,0.3")
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--config-json", default="{}", help="FixedTargetConfig overrides as JSON")
    p.add_argument("--config-key", default="locked")
    p.add_argument("--tag", default="eval_custom")
    p.add_argument("--no-null", action="store_true", help="skip the 100 all-null datasets")

    p = sub.add_parser("compare", help="bitwise comparison of a result dir with a reference dir")
    p.add_argument("run_dir", type=Path)
    p.add_argument("ref_dir", type=Path)

    p = sub.add_parser("rescore", help="CPU, no torch: rebuild every record that follows from the stored W and "
                                       "compare it byte for byte with the source")
    p.add_argument("--src", type=Path, default=REFERENCE, help="record root (default reference/exact_gaussian)")
    p.add_argument("--dst", type=Path, required=True, help="output root for the rebuilt records")

    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.mode == "compare":
        rep = compare_dirs(a.run_dir, a.ref_dir)
        print(json.dumps(rep, indent=2))
        sys.exit(0 if rep["all_identical"] else 1)
    if a.mode == "rescore":
        sys.exit(0 if mode_rescore(a.src, a.dst) else 1)

    # Training subcommands: validate the arguments and the device before any output directory exists.
    only = combos = None
    if a.mode == "main":
        only = _names(a.only, ["main", "low", "merge"])
    elif a.mode == "ablation":
        only = _names(a.only, ABLATIONS)
    elif a.mode == "audit":
        combos = [tuple(x.split(":")) for x in a.combos.split(",") if x]
        bad = [c for c in combos if c not in AUDIT_COMBOS]
        if bad:
            raise SystemExit(f"unknown combos {bad}")
    keys = _names(a.keys, TUNING_GRID) if a.mode == "tune" else None
    if not (a.mode == "main" and set(only) <= {"merge"}):
        check_device(a.device)

    if a.mode == "main":
        mode_main(a.out, a.device, only)
    elif a.mode == "ablation":
        mode_ablation(a.out, a.device, only)
    elif a.mode == "tune":
        mode_tune(a.out, a.device, _floats(a.amplitudes), keys or None)
    elif a.mode == "audit":
        mode_audit(a.out, a.device, a.trajectories, a.audit_epochs, combos)
    elif a.mode == "eval":
        mode_eval(a.out, a.device, _floats(a.amplitudes), json.loads(a.config_json), a.restarts, a.tag,
                  a.config_key, not a.no_null)
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
