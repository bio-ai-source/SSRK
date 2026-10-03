"""Exact Gaussian (Model-X) knockoffs with known covariance -- eq. (3).

For candidates ``V ~ N(mu, Sigma)`` with ``Sigma > 0`` the exact conditional
sampler is, in column-vector notation,

    V~ | V ~ N( mu + (I - Lambda Sigma^{-1})(V - mu),  2 Lambda - Lambda Sigma^{-1} Lambda ),
    Lambda = diag(s),  0 <= Lambda <= 2 Sigma.

The exact Gaussian study uses ``mu = 0`` and the equicorrelated choice
``s = 0.99 min(1, 2 lambda_min(Sigma))`` (App. A.4, "Exact Gaussian fixed-target
study": here ``s = 0.99``).  Knockoffs are generated from ``V`` and independent
generation randomness only, so ``V~`` is independent of the target ``Y`` given
``V`` and pairwise exchangeability holds exactly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["equicorrelated_s", "exact_gaussian_knockoffs"]


def equicorrelated_s(Sigma: np.ndarray) -> float:
    """Equicorrelated diagonal ``s = 0.99 * min(1, 2 * lambda_min(Sigma))``."""
    return 0.99 * min(1.0, 2.0 * float(np.linalg.eigvalsh(Sigma).min()))


def exact_gaussian_knockoffs(V: np.ndarray, Sigma: np.ndarray, rng: np.random.Generator):
    """Draw one knockoff matrix for the rows of ``V`` (mean zero, known ``Sigma``).

    Args:
        V: candidate matrix, shape (n, p); rows are i.i.d. ``N(0, Sigma)``.
        Sigma: known candidate covariance, shape (p, p).
        rng: NumPy ``Generator`` reserved for knockoff generation (independent of
            the draws of ``V`` and ``Y``); consumes one ``rng.normal(size=V.shape)``.

    Returns:
        ``(V_tilde, s_value)``.

    Row form: ``V~_i = V_i (I - Sigma^{-1} S) + z_i C^T`` with ``S = s I`` and
    ``C C^T = 2S - S Sigma^{-1} S`` (``C`` from a clipped eigendecomposition).
    """
    p = Sigma.shape[0]
    s_value = equicorrelated_s(Sigma)
    S = s_value * np.eye(p)
    Sigma_inv = np.linalg.inv(Sigma)
    mean_map = np.eye(p) - Sigma_inv @ S          # = (I - S Sigma^{-1})^T: V @ mean_map is the row form
    cond_cov = 2.0 * S - S @ Sigma_inv @ S
    cond_cov = 0.5 * (cond_cov + cond_cov.T)
    evals, evecs = np.linalg.eigh(cond_cov)
    root = evecs @ np.diag(np.sqrt(np.clip(evals, 0.0, None)))
    return V @ mean_map + rng.normal(size=V.shape) @ root.T, s_value
