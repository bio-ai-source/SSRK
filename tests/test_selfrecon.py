"""Tests of the self-reconstruction (ranking) mode, its references and the deterministic ranking.

The central test is the bitwise flip-sign property (Lemma 2, eq. (18)): retraining on
``swap_S(D)`` -- columns ``S`` of ``X`` and ``X_tilde`` exchanged -- with the same seed (same network
initialization, minibatch order and masks) must give ``W(swap_S D) = T_S W(D)`` *exactly*: negated on
``S`` and unchanged elsewhere. It holds in floating point because the gate is evaluated in the centered
form ``H = (X + X~)/2 + t (X - X~)/2`` with an exactly odd ``t`` (Remark 1 "Centered gate"), the loss uses
the exactly exchanged weights ``a, b = (1 +- t)/2``, the entropy and its gap weights are exactly even,
and Adam / global-norm clipping map a negated gate state to a negated state.

Run from the package root:  python -m pytest tests/test_selfrecon.py -q
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest

try:
    import torch
except ImportError:  # Level A environments (requirements-levelA.txt) have no torch
    torch = None

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from experiments.pbmc.run import pbmc_config  # noqa: E402
from experiments.uci_har.data import append_permuted_nulls  # noqa: E402
from ssrk.ranking import deterministic_top_k  # noqa: E402
from ssrk.references import generate_reference, independent_gaussian_reference, permutation_reference  # noqa: E402

if torch is not None:
    from ssrk.selfrecon import KnockoffGatedLayer, fit_ssrk_with_knockoffs  # noqa: E402

#: Tests that train or evaluate the network are skipped (and reported) when torch is not installed.
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")
#: CUDA cases are reported as skipped (not silently dropped) when no GPU is visible.
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(torch is None or not torch.cuda.is_available(),
                                                                reason="CUDA not available"))]


def _data(seed: int = 0, n: int = 256, p: int = 24):
    """Correlated candidates and an independent Gaussian-marginal reference (float32)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 4))
    A = rng.normal(size=(4, p))
    X = (Z @ A + 0.5 * rng.normal(size=(n, p))).astype(np.float32)
    X = (X - X.mean(0)) / X.std(0)
    Xt = (rng.normal(size=(n, p)) * X.std(0) + X.mean(0)).astype(np.float32)
    return X.astype(np.float32), Xt


def _cfg(freeze_gates_stage1: bool):
    """The PBMC configuration, shrunk (6 epochs, small MLP) and with a large gate learning rate."""
    return replace(pbmc_config(), epochs=6, batch_size=64, encoder_dims=(32, 16), decoder_dims=(16, 32),
                   latent_dim=8, lr_gate_factor=10.0, freeze_gates_stage1=freeze_gates_stage1)


@needs_torch
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("freeze", [False, True])
def test_selfrecon_bitwise_flip_sign(device, freeze):
    torch.use_deterministic_algorithms(True)
    X, Xt = _data()
    cfg = _cfg(freeze)
    dev = torch.device(device)
    seeds = (1234, 1234 + 100_003)  # two restarts on the same paired design
    Ws = [fit_ssrk_with_knockoffs(X, Xt, s, cfg, dev)[0] for s in seeds]
    W, W_bar = Ws[0], np.stack([w.astype(np.float64) for w in Ws]).mean(0)
    assert np.abs(W).max() > 1e-3  # the gates actually moved
    rng = np.random.default_rng(7)
    for _ in range(3):
        S = np.sort(rng.choice(X.shape[1], size=int(rng.integers(1, X.shape[1])), replace=False))
        X2, Xt2 = X.copy(), Xt.copy()
        X2[:, S], Xt2[:, S] = Xt[:, S], X[:, S]
        W2s = [fit_ssrk_with_knockoffs(X2, Xt2, s, cfg, dev)[0] for s in seeds]
        expected = W.copy()
        expected[S] = -expected[S]
        assert np.array_equal(W2s[0], expected), float(np.abs(W2s[0] - expected).max())
        # the restart mean (an odd aggregation, Table A1, step 7) flips exactly as well
        expected_bar = W_bar.copy()
        expected_bar[S] = -expected_bar[S]
        W2_bar = np.stack([w.astype(np.float64) for w in W2s]).mean(0)
        assert np.array_equal(W2_bar, expected_bar)


@needs_torch
def test_refit_is_deterministic_cpu():
    X, Xt = _data(1, n=128, p=10)
    cfg = _cfg(True)
    a = fit_ssrk_with_knockoffs(X, Xt, 5, cfg, torch.device("cpu"))[0]
    b = fit_ssrk_with_knockoffs(X, Xt, 5, cfg, torch.device("cpu"))[0]
    assert a.dtype == np.float32 and np.array_equal(a, b)


@needs_torch
def test_centered_gate_equals_sigmoid_form():
    """Centered form == pi X + (1 - pi) X~ with pi = sigmoid(u / tau), to rounding (eq. (4))."""
    torch.manual_seed(0)
    tau = 0.82
    layer = KnockoffGatedLayer(10, tau)
    with torch.no_grad():
        layer.gate_logits.copy_(torch.randn(10) * 3)
    u = layer.gate_logits.detach()
    pi = torch.sigmoid(u / tau)
    X, Xt = torch.randn(5, 10), torch.randn(5, 10)
    assert torch.allclose(layer(X, Xt), pi * X + (1 - pi) * Xt, atol=1e-6)
    a, b = layer.get_gate_weights()
    assert torch.allclose(a, pi, atol=1e-6) and torch.allclose(b, 1 - pi, atol=1e-6)
    ent = -(pi * torch.log(pi) + (1 - pi) * torch.log(1 - pi))
    assert torch.allclose(layer.compute_entropy_regularization(reduction="none"), ent, atol=1e-5)
    assert np.allclose(layer.compute_statistics(), (2 * pi - 1).numpy(), atol=1e-6)
    # exact oddness of t and exact exchange of (a, b) under u -> -u
    neg = KnockoffGatedLayer(10, tau)
    with torch.no_grad():
        neg.gate_logits.copy_(-u)
    assert torch.equal(neg.gate_t(), -layer.gate_t())
    a2, b2 = neg.get_gate_weights()
    assert torch.equal(a2, b) and torch.equal(b2, a)


def test_deterministic_top_k():
    s = np.array([0.5, 0.9, 0.5, -1.0, 0.9, np.nan], dtype=np.float32)
    assert deterministic_top_k(s, 3) == [1, 4, 0]  # ties broken by the smaller index
    assert deterministic_top_k(s, 6)[-1] == 5  # NaN ranks last
    assert deterministic_top_k(s, 2, eligible=np.array([0, 2, 3])) == [0, 2]
    with pytest.raises(ValueError):
        deterministic_top_k(s, 4, eligible=[0, 1])


@needs_torch
def test_references():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 6)).astype(np.float32)
    P = permutation_reference(X, seed=11)
    assert P.dtype == np.float32
    assert all(np.array_equal(np.sort(P[:, j]), np.sort(X[:, j])) for j in range(6))
    G = independent_gaussian_reference(X, seed=11)
    assert G.shape == X.shape and G.dtype == np.float32
    for kind in ("independent", "permutation"):
        a, _ = generate_reference(kind, X, 99)
        b, _ = generate_reference(kind, X, 99)
        assert a.dtype == np.float32 and np.array_equal(a, b)
    with pytest.raises(ValueError):
        generate_reference("gaussian", X, 0)


@needs_torch
@pytest.mark.parametrize("area", ["pbmc", "uci_har"])
def test_hard_coded_configs_match_original_resolution(area):
    """The hard-coded protocol presets equal the configuration resolved by the original code."""
    from experiments.uci_har.run import RESTARTS as UCI_RESTARTS, uci_config
    from experiments.pbmc.run import RESTARTS as PBMC_RESTARTS

    ref = json.loads((PKG / "reference" / area / "orig_resolved_config.json").read_text(encoding="utf-8"))
    cfg = (pbmc_config() if area == "pbmc" else uci_config()).to_dict()
    cfg["model_latent_dim"] = cfg.pop("latent_dim")
    assert {k: cfg[k] for k in ref["config"]} == ref["config"]
    assert ref["restarts"] == (PBMC_RESTARTS if area == "pbmc" else UCI_RESTARTS)
    assert ref["settings"] == {}


def test_append_permuted_nulls():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 7)).astype(np.float32)
    X_aug, nulls = append_permuted_nulls(X, 1.0, 13)
    assert X_aug.shape == (40, 14) and X_aug.dtype == np.float32 and nulls == set(range(7, 14))
    assert np.array_equal(X_aug[:, :7], X)
    sorted_real = [np.sort(X[:, j]) for j in range(7)]
    for j in range(7, 14):
        assert any(np.array_equal(np.sort(X_aug[:, j]), c) for c in sorted_real)


# ------------------------------------------------------------------- Level A rescoring (no torch)
#: Prefix of a subprocess that makes ``import torch`` fail, as on a machine without torch.
_BLOCK_TORCH = (
    "import importlib.abc, sys\n"
    "class _BlockTorch(importlib.abc.MetaPathFinder):\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'torch' or name.startswith('torch.'):\n"
    "            raise ImportError('torch is blocked in this test')\n"
    "sys.meta_path.insert(0, _BlockTorch())\n"
)


def _run_without_torch(code: str):
    import subprocess

    return subprocess.run([sys.executable, "-c", _BLOCK_TORCH + code], cwd=PKG, capture_output=True, text=True,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))


def test_level_a_scripts_import_without_torch():
    """The scripts run by ``verify.py --from-reference`` for PBMC / UCI HAR import no torch."""
    proc = _run_without_torch(
        f"sys.path.insert(0, {str(PKG)!r})\n"
        "import experiments.pbmc.rescore, experiments.uci_har.rescore, experiments.pbmc.stability\n"
        "assert 'torch' not in sys.modules\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_uci_rescore_reproduces_shipped_records_without_torch(tmp_path):
    """W -> selections, HNI, DomainBalance of all 20 UCI HAR units, byte-identical records (CPU, < 2 s)."""
    script = PKG / "experiments" / "uci_har" / "rescore.py"
    proc = _run_without_torch(
        f"import runpy\nsys.argv = [{str(script)!r}, '--out', {str(tmp_path)!r}]\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    report = json.loads((tmp_path / "rescore.json").read_text(encoding="utf-8"))
    assert all(report["checks"].values()) and len(report["units"]) == 20
    assert (tmp_path / "eval5.json").read_bytes() == (PKG / "reference" / "uci_har" / "eval5.json").read_bytes()
