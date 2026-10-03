"""Matched-marginal references for the self-reconstruction (ranking) mode.

The real-data studies (paper App. A.3.1 "Reference constructions and self-reconstruction training") use

* PBMC 3k: an *independent Gaussian-marginal* reference, which draws each gene from
  ``N(mu_j, sigma_j^2)`` (empirical mean and standard deviation of column ``j``) independently of all
  other genes and of the observed row;
* UCI HAR (and the image benchmarks): a *within-column permutation* reference, which permutes every
  column independently and therefore preserves each marginal exactly while breaking dependence.

Both samplers use numpy's legacy global stream (``np.random.seed``), column by column, exactly as the
original research code did; the draws therefore depend on the seed and on the column order only.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

from .determinism import set_seed

REFERENCE_KINDS = ("independent", "permutation")


def independent_gaussian_reference(X: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Independent Gaussian-marginal reference: ``X_tilde[:, j] ~ N(mean(X[:, j]), std(X[:, j]))``.

    The column mean/std are computed in the dtype of ``X`` (float32 in the protocols); the normal draws
    are float64 and are cast on assignment into an array of ``X``'s dtype.
    """
    if seed is not None:
        np.random.seed(seed)

    n, p = X.shape
    X_tilde = np.zeros_like(X)

    for j in range(p):
        mu = np.mean(X[:, j])
        std = np.std(X[:, j])
        X_tilde[:, j] = np.random.normal(mu, std, n)

    return X_tilde


def permutation_reference(X: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Within-column permutation reference: every column is permuted independently."""
    if seed is not None:
        np.random.seed(seed)

    n, p = X.shape
    X_tilde = np.zeros_like(X)

    for j in range(p):
        X_tilde[:, j] = np.random.permutation(X[:, j])

    return X_tilde


def generate_reference(kind: str, X: np.ndarray, seed: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """Draw the reference matrix ``X_tilde`` for the paired design ``D = (X, X_tilde)``.

    Mirrors the original ``generate_knockoffs``: global seeding (``set_seed(seed)``, which also seeds
    torch and Python's ``random``), the sampler (which re-seeds numpy's legacy stream with ``seed``),
    and a final cast to float32.

    Returns:
        ``(X_tilde float32, {"generator_time_sec": ...})``
    """
    set_seed(seed, deterministic=True)
    start = time.time()
    if kind == "independent":
        X_tilde = independent_gaussian_reference(X, seed=seed)
    elif kind == "permutation":
        X_tilde = permutation_reference(X, seed=seed)
    else:
        raise ValueError(f"Unknown reference kind {kind!r}; expected one of {REFERENCE_KINDS}")
    return X_tilde.astype(np.float32), {"generator_time_sec": float(time.time() - start)}
