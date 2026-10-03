"""Exactly odd gate nonlinearity shared by both SSRK modes.

The fixed-target mode evaluates the gate of eq. (4) in the centered form of
Remark 1 ("Centered gate"; paper, App. A.1.9, "Proof of Lemma 2: optimizer equivariance"):

    H_j = (V_j + V~_j)/2 + t_j (V_j - V~_j)/2,     t_j = 2 pi_j - 1 = tanh(u_j / (2 tau)).

Bitwise flip-sign (Lemma 2, eq. (18)) needs ``t(-u) == -t(u)`` *exactly*
in floating point, and an *even* derivative.  ``torch.tanh`` is odd in exact
arithmetic, but nothing guarantees that a vectorised kernel returns bitwise
negated values for negated inputs.  ``OddTanh`` therefore evaluates
``copysign(tanh(|x|), x)``: the magnitude is computed from ``|x|`` (identical for
``x`` and ``-x``) and the sign is attached afterwards, which makes the map odd by
construction.  The backward pass uses ``1 - t^2``, which depends on ``t`` only
through ``t * t`` and is therefore exactly even.
"""

from __future__ import annotations

import torch


class OddTanh(torch.autograd.Function):
    """tanh with an explicitly odd forward pass and an exactly even derivative."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        t = torch.copysign(torch.tanh(torch.abs(x)), x)
        ctx.save_for_backward(t)
        return t

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        (t,) = ctx.saved_tensors
        return grad_output * (1.0 - t * t)


def odd_tanh(x: torch.Tensor) -> torch.Tensor:
    """Apply :class:`OddTanh` (``copysign(tanh|x|, x)``, derivative ``1 - t^2``)."""
    return OddTanh.apply(x)
