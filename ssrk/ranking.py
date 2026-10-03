"""Deterministic fixed-budget ranking of feature statistics (self-reconstruction mode outputs).

In the real-data protocols SSRK reports the fixed-budget ranking of ``W`` (Table A1, step 8):
the ``k`` largest statistics, with ties broken by the smaller column index so that the selected set
is a deterministic function of ``W``. This module imports no torch: the Level A rescoring scripts
(``experiments/pbmc/rescore.py``, ``experiments/uci_har/rescore.py``) use it on the shipped ``W``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def deterministic_top_k(scores: np.ndarray, k: int, eligible: Sequence[int] | np.ndarray | None = None) -> list[int]:
    """Indices of the ``k`` largest ``scores`` among ``eligible`` columns.

    Order: decreasing score, ties by increasing column index (``np.lexsort``). NaN counts as ``-inf``.
    Scores are compared in float64. Returns a Python list of ints in rank order.

    Args:
        scores: feature statistics, any shape (flattened).
        k: selection budget.
        eligible: optional subset of column indices (e.g. the 561 real UCI HAR features, excluding the
            appended permuted diagnostic columns); default all columns.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if eligible is None:
        eligible = np.arange(values.size, dtype=int)
    else:
        eligible = np.asarray(eligible, dtype=int)
    if eligible.size < k:
        raise ValueError(f"Only {eligible.size} eligible features for k={k}")
    eligible_scores = values[eligible]
    eligible_scores = np.nan_to_num(eligible_scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    order = np.lexsort((eligible, -eligible_scores))
    return eligible[order[:k]].astype(int).tolist()
