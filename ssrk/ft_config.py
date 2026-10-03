"""Configuration of the fixed-target mode (paper Table A1; App. A.4, "Gate network and training").

This module holds only the :class:`FixedTargetConfig` dataclass and imports no torch, so that the
CPU-only recomputation from the shipped records (``verify.py --from-reference``: the tables, in-text
numbers and rescoring of the exact Gaussian study) runs without a torch installation.  The trainer
itself is :func:`ssrk.fixed_target.fit_fixed_target_batched`; ``ssrk.fixed_target`` re-exports this
class, so ``from ssrk.fixed_target import FixedTargetConfig`` keeps working.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

__all__ = ["FixedTargetConfig"]


@dataclass
class FixedTargetConfig:
    """Predeclared fixed-target SSRK training configuration.

    The defaults are the selected ("locked") configuration of the exact Gaussian
    study (App. A.4, "Gate network and training"); the tuning grid (Table A4) and
    the component ablation (Table A6) override individual fields.
    """

    hidden: Sequence[int] = (256, 128, 32, 128, 256)
    tau: float = 1.0
    epochs: int = 150
    batch_size: int = 128
    lr: float = 1e-3
    lr_gate: float = 2e-2
    weight_decay: float = 1e-4
    gate_weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 5.0
    # Masking (Table A1, step 4): candidate-mixture entries and target
    # entries are masked by data-independent Bernoulli masks shared by the two
    # pair members (the mask multiplies the mixture H).
    candidate_mask_prob: float = 0.2
    target_mask_prob: float = 0.5
    use_target_context: bool = True
    # Two-stage entropy schedule lambda(t) * sum_j h(pi_j), h = negative entropy
    # (eq. (6)): Stage I lambda > 0 (exploratory gates), Stage II lambda < 0
    # (sharpening).
    stage1_frac: float = 0.3
    lambda_stage1: float = 1e-3
    lambda_stage2: float = -1e-4
    # Bounded class: symmetric logit clamp |u_j| <= u_max (step 5) and layerwise
    # caps on the augmented Frobenius norm ||[A_l b_l; 0 1]||_F.
    logit_cap: float = 8.0
    frobenius_caps: Sequence[float] | None = (30.0, 30.0, 30.0, 30.0, 30.0, 30.0)
    input_clip: float = 6.0     # c_V, one coordinatewise clip shared by both pair members
    target_clip: float = 6.0    # c_Y, applied to the standardised target view
    output_clip: float = 10.0
    dtype: str = "float32"
    # "symmetric": centered form (bitwise sign-symmetric);
    # "naive": H = sigmoid(u/tau) V + (1 - sigmoid(u/tau)) V~ (the same function,
    # equivariant only up to rounding).  "naive" is used only by the swap audit.
    parameterization: str = "symmetric"

    def to_dict(self) -> dict:
        """JSON-serialisable dict (lists instead of tuples), as stored in manifests."""
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        d["betas"] = list(self.betas)
        if self.frobenius_caps is not None:
            d["frobenius_caps"] = list(self.frobenius_caps)
        return d
