"""SSRK statistics and knockoff+ filtering for the synthetic demo."""

from __future__ import annotations

from typing import Optional, Set, Tuple

import numpy as np


def _normalize_original_in_first_mask(
    pi: np.ndarray,
    original_in_first_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    if original_in_first_mask is None:
        return np.ones_like(pi, dtype=bool)

    mask = np.asarray(original_in_first_mask, dtype=bool)
    if mask.ndim == 0:
        return np.full_like(pi, bool(mask), dtype=bool)
    if mask.shape != pi.shape:
        raise ValueError(f"original_in_first_mask must have shape {pi.shape}, got {mask.shape}")
    return mask


def compute_effective_original_weight(
    pi: np.ndarray,
    original_in_first_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the slot-aware effective weight on the original feature."""
    pi = np.asarray(pi, dtype=float)
    mask = _normalize_original_in_first_mask(pi, original_in_first_mask)
    return np.where(mask, pi, 1.0 - pi)


def compute_W_statistics(
    pi: np.ndarray,
    original_in_first_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute slot-aware knockoff statistics W_j = 2 alpha_j - 1."""
    alpha = compute_effective_original_weight(pi, original_in_first_mask)
    return 2.0 * alpha - 1.0


def knockoff_plus_filter(W: np.ndarray, q: float = 0.1) -> Tuple[float, Set[int]]:
    """Apply the knockoff+ threshold to a signed statistic vector."""
    t_values = np.sort(np.unique(np.abs(W[W != 0])))
    threshold = float("inf")
    selected: Set[int] = set()

    for t in t_values:
        numerator = 1.0 + np.sum(W <= -t)
        denominator = max(1.0, np.sum(W >= t))
        if numerator / denominator <= q:
            threshold = float(t)
            break

    if threshold != float("inf"):
        selected = set(np.where(W >= threshold)[0])

    return threshold, selected


def compute_fdp_power(
    selected: Set[int],
    true_support: Set[int],
    p_features: int | None = None,
) -> Tuple[float, float]:
    """Compute false discovery proportion and power."""
    if len(selected) == 0:
        return 0.0, 0.0

    false_discoveries = len(selected - true_support)
    fdp = false_discoveries / len(selected)
    power = 1.0 if len(true_support) == 0 else len(selected & true_support) / len(true_support)
    return fdp, power
