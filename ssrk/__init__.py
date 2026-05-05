"""
SSRK - Self-Supervised Reconstruction Knockoffs.
Minimal public package for the synthetic oracle demo.
"""

from .model import KnockoffGatedLayer, SSRKModel
from .statistics import (
    compute_effective_original_weight,
    compute_W_statistics,
    compute_fdp_power,
    knockoff_plus_filter,
)
from .training import set_seed

__version__ = "2.0.0"

__all__ = [
    "KnockoffGatedLayer",
    "SSRKModel",
    "set_seed",
    "compute_effective_original_weight",
    "compute_W_statistics",
    "compute_fdp_power",
    "knockoff_plus_filter",
]
