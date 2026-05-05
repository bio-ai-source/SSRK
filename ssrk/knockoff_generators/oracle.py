"""Oracle synthetic knockoff generator for the SSRK synthetic demo."""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np


def generate_oracle_knockoffs_latent_factor(
    n_samples: int,
    p_features: int,
    s_signals: int,
    latent_dim: int = 3,
    signal_strength: float = 3.0,
    noise_std: float = 0.5,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Set[int]]:
    """Generate synthetic data and oracle knockoffs from a latent-factor model."""
    if seed is not None:
        np.random.seed(seed)

    A = np.random.randn(s_signals, latent_dim)
    A = A / np.linalg.norm(A, axis=1, keepdims=True) * signal_strength

    Z1 = np.random.randn(n_samples, latent_dim)
    Z2 = np.random.randn(n_samples, latent_dim)

    X = np.zeros((n_samples, p_features))
    X_tilde = np.zeros((n_samples, p_features))

    if s_signals > 0:
        X[:, :s_signals] = Z1 @ A.T + np.random.randn(n_samples, s_signals) * noise_std
        X_tilde[:, :s_signals] = Z2 @ A.T + np.random.randn(n_samples, s_signals) * noise_std

    if p_features > s_signals:
        n_nulls = p_features - s_signals
        X[:, s_signals:] = np.random.randn(n_samples, n_nulls) * noise_std
        X_tilde[:, s_signals:] = np.random.randn(n_samples, n_nulls) * noise_std

    true_support = set(range(s_signals))
    return X, X_tilde, true_support
