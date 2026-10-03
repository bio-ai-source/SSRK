"""Tests of the fixed-target mode and the exact Gaussian protocol.

* exact oddness of the gate nonlinearity (Remark 1, "Centered gate");
* bitwise flip-sign of the batched trainer (Lemma 2, eq. (18)): retraining
  on swapped data with the same member layout reproduces T_S W exactly, on CPU and
  CUDA, in float32 and float64;
* the naive (uncentered) form is the same function (to rounding);
* chunk packing reproduces the layouts stored in the reference manifests;
* the data-generating process (support, amplitudes, common random numbers);
* set_seed seeds every global generator like the research code; the cuBLAS
  workspace setting overrides other values; the training subcommands check the
  device before creating output directories;
* record-level regressions, CPU only and without torch, from the shipped reference
  files: the tuning summary / selection rule, the eval merge, the rescoring of a
  record directory from its W, the Figure A1 panel data, the pandas float parser
  that the byte identity of the merged summaries depends on, and the torch-free
  imports of the scripts used by ``verify.py --from-reference``.

The tests that train need torch; the others also run with ``requirements-levelA.txt``.
"""

from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # the records' value, set before torch is imported

import io  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

try:
    import torch
except ImportError:  # Level A installation (requirements-levelA.txt)
    torch = None

from experiments.exact_gaussian import protocol as P  # noqa: E402
from ssrk.ft_config import FixedTargetConfig  # noqa: E402

if torch is not None:
    from ssrk.determinism import set_seed
    from ssrk.fixed_target import fit_fixed_target_batched
    from ssrk.gates import odd_tanh

REF = PKG / "reference" / "exact_gaussian"
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="CUDA not available"))]
needs_reference = pytest.mark.skipif(not (REF / "eval_all" / "W_per_seed.npz").exists(),
                                     reason="reference records not available")


def _members(n=96, p=12, r=4, seeds=(11, 12, 13)):
    Vs, Vts, Ys = [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        V = rng.normal(size=(n, p))
        Vt = rng.normal(size=(n, p))
        B = np.zeros((p, r))
        B[:3, 0], B[3:5, 1] = 0.9, -0.7
        Vs.append(V)
        Vts.append(Vt)
        Ys.append(V @ B + rng.normal(size=(n, r)))
    return np.stack(Vs), np.stack(Vts), np.stack(Ys)


def _swap(V, Vt, subset):
    Vs, Vts = V.copy(), Vt.copy()
    Vs[:, :, subset], Vts[:, :, subset] = Vt[:, :, subset], V[:, :, subset]
    return Vs, Vts


# ----------------------------------------------------------------------------
# gate nonlinearity
# ----------------------------------------------------------------------------

@needs_torch
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_odd_tanh_is_exactly_odd_with_even_derivative(device, dtype):
    dtype = getattr(torch, dtype)
    x = torch.linspace(-20, 20, 10_001, dtype=dtype, device=device)
    x = torch.cat([x, torch.tensor([0.0, 1e-30, -1e-30, 3e-8], dtype=dtype, device=device)])
    assert torch.equal(odd_tanh(-x), -odd_tanh(x))
    xp = x.clone().requires_grad_(True)
    xn = (-x).clone().requires_grad_(True)
    odd_tanh(xp).sum().backward()
    odd_tanh(xn).sum().backward()
    assert torch.equal(xp.grad, xn.grad)


# ----------------------------------------------------------------------------
# bitwise flip-sign of the batched trainer
# ----------------------------------------------------------------------------

@needs_torch
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_batched_retraining_on_swapped_data_flips_W_bitwise(device, dtype):
    V, Vt, Y = _members()
    # Small Frobenius caps so that norm projections are active during the test.
    cfg = FixedTargetConfig(hidden=(32, 16, 32), epochs=8, batch_size=32,
                            frobenius_caps=(3.0, 3.0, 3.0, 3.0), dtype=dtype)
    seeds = [101, 202, 303]
    fit = fit_fixed_target_batched(V, Vt, Y, cfg, seeds, device)
    subset = np.array([0, 3, 7, 11])
    Vs, Vts = _swap(V, Vt, subset)
    fit_s = fit_fixed_target_batched(Vs, Vts, Y, cfg, seeds, device)
    sign = np.ones(V.shape[2])
    sign[subset] = -1
    assert np.any(fit.W != 0)
    assert np.all(fit.cap_projections > 0)
    assert np.array_equal(fit_s.W, fit.W * sign)
    assert np.array_equal(fit_s.logits, fit.logits * sign)
    assert np.array_equal(fit_s.final_loss, fit.final_loss)
    assert np.array_equal(fit_s.cap_projections, fit.cap_projections)
    assert np.array_equal(fit_s.max_frobenius, fit.max_frobenius)


@needs_torch
@pytest.mark.parametrize("device", DEVICES)
def test_flip_sign_without_target_context_and_without_entropy(device):
    V, Vt, Y = _members(seeds=(21, 22))
    for override in ({"use_target_context": False, "target_mask_prob": 1.0},
                     {"lambda_stage1": 0.0, "lambda_stage2": 0.0}):
        cfg = FixedTargetConfig(hidden=(16, 8, 16), epochs=5, batch_size=32,
                                frobenius_caps=(30.0,) * 4, **override)
        seeds = [7, 8]
        fit = fit_fixed_target_batched(V, Vt, Y, cfg, seeds, device)
        subset = np.array([1, 2, 10])
        Vs, Vts = _swap(V, Vt, subset)
        fit_s = fit_fixed_target_batched(Vs, Vts, Y, cfg, seeds, device)
        sign = np.ones(V.shape[2])
        sign[subset] = -1
        assert np.any(fit.W != 0)
        assert np.array_equal(fit_s.W, fit.W * sign), override


@needs_torch
def test_full_swap_and_empty_swap():
    V, Vt, Y = _members(seeds=(31, 32))
    cfg = FixedTargetConfig(hidden=(16, 8, 16), epochs=4, batch_size=32, frobenius_caps=(30.0,) * 4)
    fit = fit_fixed_target_batched(V, Vt, Y, cfg, [1, 2], "cpu")
    fit_all = fit_fixed_target_batched(Vt, V, Y, cfg, [1, 2], "cpu")
    fit_none = fit_fixed_target_batched(V.copy(), Vt.copy(), Y, cfg, [1, 2], "cpu")
    assert np.array_equal(fit_all.W, -fit.W)
    assert np.array_equal(fit_none.W, fit.W)


@needs_torch
def test_naive_parameterization_is_the_same_function():
    V, Vt, Y = _members(seeds=(41,))
    kw = dict(hidden=(32, 16, 32), epochs=1, batch_size=32, frobenius_caps=None, dtype="float64")
    a = fit_fixed_target_batched(V, Vt, Y, FixedTargetConfig(**kw), [1], "cpu")
    b = fit_fixed_target_batched(V, Vt, Y, FixedTargetConfig(parameterization="naive", **kw), [1], "cpu")
    np.testing.assert_allclose(a.W, b.W, rtol=0, atol=1e-10)
    assert np.any(a.W != 0)


def test_default_config_is_the_locked_configuration():
    d = FixedTargetConfig().to_dict()
    ref = REF / "eval_all" / "manifest.json"
    if not ref.exists():
        pytest.skip("reference records not available")
    assert d == json.loads(ref.read_text(encoding="utf-8"))["config"]


@needs_torch
def test_fixed_target_reexports_the_torch_free_config():
    import ssrk.fixed_target
    import ssrk.ft_config

    assert ssrk.fixed_target.FixedTargetConfig is ssrk.ft_config.FixedTargetConfig


# ----------------------------------------------------------------------------
# computational protocol: chunk packing
# ----------------------------------------------------------------------------

def _dummy_units(seed_groups, R):
    units = []
    for seeds, every in seed_groups:
        for s in seeds:
            units.append({"seed": s, "algo": P.algorithm_seeds(P.EVAL_ALGORITHM_BASE, R),
                          "audit_subset": (np.array([0]) if every > 0 and s % every == 0 else None)})
    return units


@pytest.mark.parametrize("run, n_amp, with_null, R", [
    ("eval_main", 4, True, 20), ("eval_low", 2, False, 20), ("ablation_R1", 2, False, 1),
    ("ablation_R5", 2, False, 5), ("ablation_noctx", 2, False, 20)])
def test_chunk_layout_matches_reference_manifest(run, n_amp, with_null, R):
    ref = REF / run / "manifest.json"
    if not ref.exists():
        pytest.skip("reference records not available")
    groups = [(P.PROTOCOL["signal_seeds"], P.EVAL_SIGNAL_AUDIT_EVERY)] * n_amp
    if with_null:
        groups.append((P.PROTOCOL["all_null_seeds"], P.EVAL_NULL_AUDIT_EVERY))
    plan = P.chunk_plan(_dummy_units(groups, R))
    got = [{"chunk": i, "audit_chunk": a, "members": sum(len(u["algo"]) for u in c), "units": [u["seed"] for u in c]}
           for i, (c, a) in enumerate(plan)]
    want = [{k: v for k, v in c.items() if k != "seconds"}
            for c in json.loads(ref.read_text(encoding="utf-8"))["chunk_layout"]]
    assert got == want
    assert all(g["members"] <= P.MAX_MEMBERS for g in got)


# ----------------------------------------------------------------------------
# data-generating process
# ----------------------------------------------------------------------------

def test_make_dataset_design():
    V, Vt, Y, B, support, s = P.make_dataset(3, False, 0.3)
    assert V.shape == Vt.shape == (800, 80) and Y.shape == (800, 12)
    assert len(support) == 30 and support == sorted(set(support))
    nz = np.flatnonzero(np.abs(B).sum(axis=1) > 0)
    assert nz.tolist() == support
    assert np.allclose(np.linalg.norm(B[support], axis=1), 0.3)
    assert np.all((B[support] != 0).sum(axis=1) == 3)
    assert s == 0.99
    # common random numbers across amplitudes: only B (and hence Y) changes
    V2, Vt2, Y2, B2, support2, _ = P.make_dataset(3, False, 0.75)
    assert np.array_equal(V, V2) and np.array_equal(Vt, Vt2) and support == support2
    assert np.array_equal(B2 != 0, B != 0) and np.allclose(B2 * 0.3, B * 0.75)
    assert not np.array_equal(Y, Y2)
    # all-null model
    V0, Vt0, Y0, B0, support0, _ = P.make_dataset(1000, True)
    assert support0 == [] and not B0.any()


def test_audit_subset_law():
    sizes = [len(P.audit_subset(s)) for s in range(0, 1100, 5)]
    assert min(sizes) >= 1 and max(sizes) <= 40
    S = P.audit_subset(0)
    assert np.all(np.diff(S) > 0)


# ----------------------------------------------------------------------------
# seeding, cuBLAS workspace, device check
# ----------------------------------------------------------------------------

@needs_torch
def test_set_seed_matches_direct_seeding():
    set_seed(1234)
    a = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    b = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert a == b
    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_cublas_workspace_other_value_is_overridden_with_a_warning(monkeypatch, capsys):
    from ssrk.determinism import enforce_cublas_workspace

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    assert enforce_cublas_workspace() == ":16:8"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "overridden" in capsys.readouterr().err
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    assert enforce_cublas_workspace() is None
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert capsys.readouterr().err == ""


@needs_torch
def test_training_subcommands_check_the_device_before_creating_outputs(tmp_path, monkeypatch):
    from experiments.exact_gaussian import run as RUN

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for args in (["ablation", "--only", "R1"], ["main", "--only", "low"], ["audit"], ["tune"], ["eval"]):
        monkeypatch.setattr(sys, "argv", ["run.py", "--out", str(tmp_path / "out"), "--device", "cuda", *args])
        with pytest.raises(SystemExit) as exc:
            RUN.main()
        assert exc.value.code not in (0, None) and "CUDA is not available" in str(exc.value.code)
        assert not (tmp_path / "out").exists()


def test_level_a_scripts_import_no_torch_and_override_the_workspace(tmp_path):
    """The scripts run by ``verify.py --from-reference`` must work without torch."""
    stub = tmp_path / "stub" / "torch"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("raise ImportError('torch blocked by the test')\n", encoding="utf-8")
    code = ("import os, sys\n"
            "import experiments.exact_gaussian.make_tables, experiments.exact_gaussian.make_figure\n"
            "import experiments.exact_gaussian.run\n"
            "assert 'torch' not in sys.modules\n"
            "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "stub"), PYTHONDONTWRITEBYTECODE="1",
               CUBLAS_WORKSPACE_CONFIG=":16:8")
    proc = subprocess.run([sys.executable, "-c", code], cwd=PKG, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ":4096:8"
    assert "overridden" in proc.stderr


# ----------------------------------------------------------------------------
# record-level regressions (CPU, no torch, from the shipped reference files)
# ----------------------------------------------------------------------------

PARSER_MESSAGE = ("pandas' default CSV float parser differs from the one that produced the reference records "
                  "(pandas==3.0.6, default C parser, which is not round-trip); the merged eval_all and the tuning "
                  "summaries rebuilt from the per-run CSVs will not be byte-identical to the records")


def test_pandas_default_float_parser_is_the_records_parser():
    x = pd.read_csv(io.StringIO("x\n0.14285714285714285\n")).x[0]
    assert x == 0.1428571428571428, f"{PARSER_MESSAGE} (parsed {x!r})"
    assert pd.read_csv(io.StringIO("x\n0.14285714285714285\n"), float_precision="round_trip").x[0] == 1 / 7


@needs_reference
def test_tuning_summary_and_selection_reproduce_records(tmp_path):
    from experiments.exact_gaussian.run import tuning_summary_from_rows

    summary = tuning_summary_from_rows(REF / "tuning", list(P.TUNING_GRID), P.TUNING_AMPLITUDES)
    P.write_csv(tmp_path / "tuning_summary.csv", summary)
    assert (tmp_path / "tuning_summary.csv").read_bytes() == (REF / "tuning" / "tuning_summary.csv").read_bytes()
    sel = P.select_configuration(summary)
    assert sel == json.loads((REF / "tuning" / "selection.json").read_text(encoding="utf-8"))
    assert sel["chosen"] == P.LOCKED_CONFIG_KEY and sel["restarts"] == P.LOCKED_RESTARTS


@needs_reference
def test_merge_reproduces_eval_all(tmp_path):
    from experiments.exact_gaussian.run import compare_dirs, merge_evals

    for tag in ("eval_main", "eval_low"):
        shutil.copytree(REF / tag, tmp_path / tag)
    merge_evals(tmp_path, ["eval_main", "eval_low"], "eval_all")
    rep = compare_dirs(tmp_path / "eval_all", REF / "eval_all")
    assert rep["all_identical"], {k: v for k, v in rep["checks"].items() if not v}


@needs_reference
def test_rescore_rebuilds_a_record_directory_from_W_and_detects_changes(tmp_path):
    from experiments.exact_gaussian.run import rescore_eval

    checks = rescore_eval(REF / "ablation_R1", tmp_path / "rebuilt")
    assert all(checks.values()), [k for k, v in checks.items() if not v]
    assert checks["per_seed_q.csv rebuilt: byte-identical"] and checks["summary.csv rebuilt: byte-identical"]

    # A one-ulp change of one restart statistic breaks the aggregate check ...
    src = tmp_path / "src"
    shutil.copytree(REF / "ablation_R1", src)
    with np.load(src / "W_per_seed.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["W_signal_runs"][0, 0, 5] = np.nextafter(arrays["W_signal_runs"][0, 0, 5], 2.0)
    np.savez_compressed(src / "W_per_seed.npz", **arrays)
    checks = rescore_eval(src, tmp_path / "ulp")
    assert not checks["W aggregate == float64 mean of W_runs over restarts (bitwise)"]
    assert not checks["W_runs: float32 trainer values stored as float64"]
    # ... and a consistent change of the statistics of one dataset changes the rebuilt rows.
    arrays["W_signal_runs"][0] = np.abs(arrays["W_signal_runs"][0])
    arrays["W_signal"][0] = arrays["W_signal_runs"][0].mean(axis=0)
    np.savez_compressed(src / "W_per_seed.npz", **arrays)
    checks = rescore_eval(src, tmp_path / "changed")
    assert checks["W aggregate == float64 mean of W_runs over restarts (bitwise)"]
    assert not checks["per_seed_q.csv rebuilt: byte-identical"]
    assert not checks["per_seed_info.json agrees with npz (seed, model, signal, support, W)"]


@needs_reference
def test_figure_panel_data_match_the_shipped_csv(tmp_path):
    from experiments.exact_gaussian.make_figure import PANEL_CSV, panel_data, write_panel_csv

    summary = pd.read_csv(REF / "eval_all" / "summary.csv", float_precision="round_trip")
    with np.load(REF / "eval_all" / "W_per_seed.npz") as npz:
        data = panel_data(summary, npz, 0.40)
    write_panel_csv(data, tmp_path / PANEL_CSV)
    assert (tmp_path / PANEL_CSV).read_bytes() == (REF / PANEL_CSV).read_bytes()
    assert int(data["null_counts"].sum()) == 50 * 50 and int(data["non_counts"].sum()) == 50 * 30
