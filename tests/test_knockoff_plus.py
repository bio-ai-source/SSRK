"""Edge cases of the knockoff+ rule, eq. (8).

T = min{t in {|W_j| : |W_j| > 0} : (1 + #{W_j <= -t}) / max(1, #{W_j >= t}) <= q},
S = {j : W_j >= T},  min(empty) = +inf.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from ssrk.knockoff_plus import knockoff_plus  # noqa: E402


def brute_force(W, q):
    """Definition of eq. (8) with exact rational arithmetic.

    ``q`` is the decimal level (``Fraction("0.1")``, not the binary value of the
    float 0.1): e.g. 3/10 <= q must hold at q = 0.3.  The float implementation
    agrees because a correctly rounded ``a / b`` equals the float literal ``q``
    whenever ``a / b`` equals the decimal ``q``.
    """
    W = [float(w) for w in W]
    qf = Fraction(repr(q))
    ts = sorted({abs(w) for w in W if w != 0})
    for t in ts:
        neg = sum(w <= -t for w in W)
        pos = sum(w >= t for w in W)
        if Fraction(1 + neg, max(1, pos)) <= qf:
            return t, [j for j, w in enumerate(W) if w >= t]
    return math.inf, []


def test_empty_selection_when_no_threshold_qualifies():
    assert knockoff_plus(np.array([-0.5, -0.2, 0.0, -0.1]), 0.1) == (math.inf, [])
    assert knockoff_plus(np.zeros(20), 0.2) == (math.inf, [])          # empty threshold set
    assert knockoff_plus(np.array([]), 0.1) == (math.inf, [])


def test_offset_one_requires_at_least_one_over_q_selections():
    # 9 positives, no negatives: (1 + 0) / 9 > 0.1 -> empty;  10 positives: 1/10 <= 0.1 -> all 10.
    W9 = np.r_[np.linspace(0.1, 0.9, 9), np.zeros(5)]
    assert knockoff_plus(W9, 0.10) == (math.inf, [])
    W10 = np.r_[np.linspace(0.1, 1.0, 10), np.zeros(5)]
    T, sel = knockoff_plus(W10, 0.10)
    assert sel == list(range(10)) and T == pytest.approx(0.1)
    # q = 0.05 needs at least ceil(1/q) = 20 selections
    W19 = np.linspace(0.05, 0.95, 19)
    assert knockoff_plus(W19, 0.05) == (math.inf, [])
    assert len(knockoff_plus(np.linspace(0.05, 1.0, 20), 0.05)[1]) == 20


def test_threshold_set_excludes_zeros():
    W = np.r_[np.full(30, 0.0), np.linspace(0.2, 0.9, 12)]
    T, sel = knockoff_plus(W, 0.10)
    assert T > 0 and all(W[j] > 0 for j in sel)
    assert sel == list(range(30, 42))
    # zeros are never selected and never counted as negatives
    T2, sel2 = knockoff_plus(np.r_[W, np.zeros(50)], 0.10)
    assert (T2, sel2) == (T, sel)


def test_ties_at_the_threshold():
    # 20 ties at +0.5 and one tie at -0.5: at t = 0.5, (1 + 1)/20 = 0.1 <= 0.1 -> all 20 selected.
    W = np.r_[np.full(20, 0.5), [-0.5]]
    T, sel = knockoff_plus(W, 0.10)
    assert T == 0.5 and sel == list(range(20))
    # two negative ties: (1 + 2)/20 = 0.15 > 0.1 -> empty at q = 0.1, all 20 at q = 0.15
    W2 = np.r_[np.full(20, 0.5), [-0.5, -0.5]]
    assert knockoff_plus(W2, 0.10) == (math.inf, [])
    assert knockoff_plus(W2, 0.15) == (0.5, list(range(20)))


def test_smallest_qualifying_threshold_is_returned():
    # Thresholds 0.1 (ratio (1+1)/11 > 0.1) fails; 0.2 qualifies with 10 selections.
    W = np.r_[np.linspace(0.2, 1.0, 10), [0.1, -0.1]]
    T, sel = knockoff_plus(W, 0.10)
    assert T == pytest.approx(0.2) and sel == list(range(10))


def test_matches_definition_on_random_inputs_with_ties_and_zeros():
    rng = np.random.default_rng(0)
    for _ in range(400):
        p = int(rng.integers(1, 120))
        W = np.round(rng.normal(0.3, 0.6, size=p), int(rng.integers(0, 3)))
        W[rng.random(p) < 0.15] = 0.0
        for q in (0.05, 0.10, 0.15, 0.3):
            T, sel = knockoff_plus(W, q)
            Tb, selb = brute_force(W, q)
            assert sel == selb and (T == Tb or (math.isinf(T) and math.isinf(Tb)))
            assert isinstance(T, float) and all(isinstance(j, int) for j in sel)


def test_selection_is_nested_in_q():
    rng = np.random.default_rng(1)
    for _ in range(200):
        W = rng.normal(0.4, 0.5, size=80)
        sets = [set(knockoff_plus(W, q)[1]) for q in (0.05, 0.10, 0.15, 0.2)]
        assert all(a <= b for a, b in zip(sets, sets[1:]))


def test_threshold_is_an_attained_magnitude_and_selection_is_the_upper_set():
    rng = np.random.default_rng(2)
    for _ in range(100):
        W = np.round(rng.normal(0.3, 0.5, size=60), 2)
        T, sel = knockoff_plus(W.tolist(), 0.2)          # lists are accepted
        if not sel:
            assert math.isinf(T)
            continue
        assert T in set(np.abs(W[W != 0]).tolist())
        assert sel == np.flatnonzero(W >= T).tolist()
    assert knockoff_plus(-np.abs(rng.normal(size=40)), 0.2) == (math.inf, [])
