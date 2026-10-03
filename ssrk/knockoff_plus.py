"""Knockoff+ selection rule -- eq. (8).

For a target level ``q`` in (0, 1) and the threshold set
``T_set = {|W_j| : |W_j| > 0}`` (zeros are excluded),

    T = min{ t in T_set : (1 + #{j : W_j <= -t}) / max(1, #{j : W_j >= t}) <= q },
    S_hat = {j : W_j >= T},        min(empty set) := +inf  (empty selection).

Ties are handled by the definition: all coordinates with ``W_j >= T`` are
selected and all with ``W_j <= -T`` are counted in the numerator.  For a fixed
``W`` the selected set is nondecreasing in ``q``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["knockoff_plus"]


def knockoff_plus(W: np.ndarray, q: float) -> tuple[float, list[int]]:
    """Return ``(T, selected)`` for knockoff+ at level ``q``.

    ``T`` is ``math.inf`` and ``selected`` is empty when no threshold in the
    threshold set satisfies the knockoff+ inequality.  ``selected`` lists
    coordinate indices in increasing order.
    """
    W = np.asarray(W, dtype=float)
    candidates = np.sort(np.unique(np.abs(W[W != 0])))
    for t in candidates:
        if (1.0 + np.sum(W <= -t)) / max(1.0, float(np.sum(W >= t))) <= q:
            return float(t), [int(j) for j in np.flatnonzero(W >= t)]
    return math.inf, []
