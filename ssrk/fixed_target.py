"""Fixed-target mode of SSRK (controlled discovery; paper Table A1, App. A.2).

Each observation is split, before reference generation, into candidate
coordinates ``V`` (knockoffs ``V~`` are generated for these only) and a
predeclared, held-out target view ``Y``.  One vector of gate logits ``u`` mixes
every candidate pair (eq. (4)),

    H_j = pi_j V_j + (1 - pi_j) V~_j,          pi_j = sigmoid(u_j / tau),

and a ReLU MLP predicts the masked entries of the standardised target view from
the masked mixture and, as swap-invariant context, the visible target entries
and the target mask.  The reversed input ordering uses the tied logit ``-u`` and
yields the same mixture (eq. (5)), so a single pass suffices.  The signed
statistic is (eq. (7))

    W_j = 2 pi_j - 1 = tanh(u_j / (2 tau)).

Centered (sign-symmetric) floating-point form -- Remark 1 ("Centered gate", App. A.1.9)
---------------------------------------------------------------------------------------
``H = (V + V~)/2 + t (V - V~)/2`` with ``t = copysign(tanh(|u|/(2 tau)), u)``
(:func:`ssrk.gates.odd_tanh`).  Under IEEE-754 round-to-nearest, ``V + V~`` is
commutative, ``V - V~ = -(V~ - V)``, multiplication by 1/2 is exact and
``(-t)(-d) = t d``; hence after swapping a subset ``S`` of pairs and negating
``u_S`` the mixture is *bitwise* identical.  The entropy term, computed from
``(1 + t)/2`` and ``(1 - t)/2``, is exactly even; AdamW, decoupled weight decay,
per-member global-norm clipping, the symmetric logit clamp and the Frobenius
projections (which act on network weights only) map a negated gate state to a
negated gate state.  With deterministic kernels, retraining on ``swap_S(D)``
therefore reproduces ``T_S W(D)`` exactly (Lemma 2, eq. (18)).

Batched training
----------------
``K`` independent gate networks (datasets x restarts, plus swapped copies for
the swap checks) are trained simultaneously with batched matrix products
(``torch.baddbmm``).  Every member has its own data, its own algorithm seed
(initialisation, minibatch order and masks come from per-member generators),
its own global-norm clipping, AdamW state, Frobenius caps and logit clamp, so
member ``k``'s values never mix with other members'.  Batched BLAS kernels may
round differently by batch slot, so the *member layout of a call is part of the
computational protocol*: a swap check retrains the swapped data in a second call
with an identical layout (member ``k`` of the second call is the swapped copy of
member ``k`` of the first), which is exactly "retrain on swap_S(D)".

The code is a faithful port of the research implementation that produced the
paper's exact Gaussian study records: same operations, dtypes, operation order and RNG
consumption order.  Do not "simplify" any arithmetic here -- results are
compared bitwise with the stored records.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .determinism import configure_determinism
from .ft_config import FixedTargetConfig  # noqa: F401  (re-export; torch-free module)
from .gates import odd_tanh

__all__ = ["FixedTargetConfig", "BatchedFit", "fit_fixed_target_batched"]


@dataclass
class BatchedFit:
    """Outputs of one batched call with ``K`` members."""

    W: np.ndarray                 # (K, p) statistics tanh(u / (2 tau)), float64 copy
    logits: np.ndarray            # (K, p) final gate logits
    final_loss: np.ndarray        # (K,) reconstruction loss of the last minibatch
    max_frobenius: np.ndarray     # (K, L) max augmented Frobenius norm reached per layer
    cap_projections: np.ndarray   # (K,) number of Frobenius-cap projections
    logit_clamp_active: np.ndarray  # (K,) number of logit entries that hit the clamp
    seconds: float                # wall time of the call (including synchronisation)


def _preprocess(V, Vt, Y, config: FixedTargetConfig):
    """Swap-equivariant preprocessing (Table A1, step 3).

    The same fixed coordinatewise clip ``c_V`` is applied to both pair members;
    the target view is standardised with its own column statistics (per member)
    and clipped at ``c_Y`` -- a function of ``Y`` alone, hence swap-invariant.
    Arrays have shapes (K, n, p), (K, n, p), (K, n, r).
    """
    c = config.input_clip
    if c:
        V = np.clip(V, -c, c)
        Vt = np.clip(Vt, -c, c)
    y_mean = Y.mean(axis=1, keepdims=True)
    y_sd = Y.std(axis=1, keepdims=True)
    y_sd = np.where(y_sd > 0, y_sd, 1.0)
    Yn = (Y - y_mean) / y_sd
    if config.target_clip:
        Yn = np.clip(Yn, -config.target_clip, config.target_clip)
    return V, Vt, Yn


def fit_fixed_target_batched(
    V: np.ndarray,
    V_tilde: np.ndarray,
    Y: np.ndarray,
    config: FixedTargetConfig,
    algorithm_seeds: Sequence[int],
    device: str | torch.device = "cuda",
) -> BatchedFit:
    """Train ``K`` fixed-target gate networks in one batched call.

    Args:
        V, V_tilde: candidates and their knockoffs, shape (K, n, p).
        Y: target views, shape (K, n, r).
        config: the predeclared configuration (shared by all members).
        algorithm_seeds: one data-independent seed per member.  Member ``k``'s
            network initialisation uses a CPU ``torch.Generator`` seeded with
            ``seed_k``; its minibatch permutations and masks use a generator on
            ``device`` seeded with ``seed_k + 7919``.
        device: torch device of the batched call.

    Returns:
        :class:`BatchedFit` with ``W = tanh(u/(2 tau))`` per member (eq. (7)).

    Implements Table A1 steps 3-6 for every member: zero gate logits
    and optimizer moments, He-uniform network weights, the centered mixture,
    masked-target reconstruction loss plus the two-stage entropy penalty,
    AdamW with a cosine schedule, per-member global-norm clipping, symmetric
    logit truncation and Frobenius projections of the network weights.
    """
    configure_determinism()
    t_start = time.perf_counter()
    dev = torch.device(device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    V = np.asarray(V, float)
    V_tilde = np.asarray(V_tilde, float)
    Y = np.asarray(Y, float)
    K, n, p = V.shape
    r = Y.shape[2]
    if len(algorithm_seeds) != K:
        raise ValueError("one algorithm seed per member is required")
    Vp, Vtp, Yp = _preprocess(V, V_tilde, Y, config)
    Vd = torch.tensor(Vp, dtype=dtype, device=dev)
    Vtd = torch.tensor(Vtp, dtype=dtype, device=dev)
    Yd = torch.tensor(Yp, dtype=dtype, device=dev)
    # Swap-invariant midpoint and swap-odd half difference of the centered form.
    mid_all = 0.5 * (Vd + Vtd)
    half_all = 0.5 * (Vd - Vtd)
    naive = config.parameterization == "naive"

    # ---------------- parameters (per member; step 3) ----------------
    use_ctx = config.use_target_context
    dims = [p + (2 * r if use_ctx else 0), *list(config.hidden), r]
    weights, biases = [], []
    init_gens = [torch.Generator().manual_seed(int(s)) for s in algorithm_seeds]
    for li in range(len(dims) - 1):
        fan_in, fan_out = dims[li], dims[li + 1]
        wb = math.sqrt(6.0 / fan_in)     # He-uniform bound for ReLU layers
        bb = 1.0 / math.sqrt(fan_in)     # small uniform biases
        Wl = torch.stack([(torch.rand((fan_in, fan_out), generator=g, dtype=torch.float64) * 2 - 1) * wb
                          for g in init_gens])
        bl = torch.stack([(torch.rand((1, fan_out), generator=g, dtype=torch.float64) * 2 - 1) * bb
                          for g in init_gens])
        weights.append(Wl.to(dev, dtype).requires_grad_(True))
        biases.append(bl.to(dev, dtype).requires_grad_(True))
    # Gate logits and optimizer moments start at zero.
    U = torch.zeros((K, p), dtype=dtype, device=dev, requires_grad=True)
    net_params = weights + biases
    optimizer = torch.optim.AdamW(
        [
            {"params": net_params, "lr": config.lr, "weight_decay": config.weight_decay},
            {"params": [U], "lr": config.lr_gate, "weight_decay": config.gate_weight_decay},
        ],
        betas=tuple(config.betas), eps=config.eps, foreach=False,
    )
    base_lrs = [config.lr, config.lr_gate]
    steps_per_epoch = int(math.ceil(n / config.batch_size))
    total_steps = config.epochs * steps_per_epoch
    stage1_epochs = int(round(config.stage1_frac * config.epochs))
    caps = None if config.frobenius_caps is None else torch.tensor(
        list(config.frobenius_caps), dtype=dtype, device=dev)
    if caps is not None and caps.numel() != len(weights):
        raise ValueError(f"need {len(weights)} Frobenius caps")

    data_gens = [torch.Generator(device=dev).manual_seed(int(s) + 7919) for s in algorithm_seeds]
    arangeK = torch.arange(K, device=dev)[:, None]
    projections = torch.zeros(K, dtype=torch.long, device=dev)
    clamp_active = torch.zeros(K, dtype=torch.long, device=dev)
    max_frob = torch.zeros((K, len(weights)), dtype=dtype, device=dev)
    last_loss = torch.full((K,), float("nan"), dtype=dtype, device=dev)
    step = 0
    for epoch in range(config.epochs):
        lam = config.lambda_stage1 if epoch < stage1_epochs else config.lambda_stage2
        # Per-member randomness of this epoch, drawn in a fixed order:
        # minibatch permutation, then one uniform array for both masks.
        perms = torch.stack([torch.randperm(n, generator=g, device=dev) for g in data_gens])
        umask = torch.stack([torch.rand((n, p + r), generator=g, device=dev, dtype=dtype) for g in data_gens])
        cand_all = (umask[:, :, :p] < config.candidate_mask_prob).to(dtype)
        targ_all = (umask[:, :, p:] < config.target_mask_prob).to(dtype)
        # Guarantee one predicted target entry per row (deterministic: the
        # coordinate with the smallest uniform draw).
        empty = targ_all.sum(dim=2, keepdim=True) == 0
        forced = torch.nn.functional.one_hot(umask[:, :, p:].argmin(dim=2), r).to(dtype)
        targ_all = torch.where(empty, forced, targ_all)
        for start in range(0, n, config.batch_size):
            idx = perms[:, start:start + config.batch_size]          # (K, B)
            cm = cand_all[:, start:start + config.batch_size]        # (K, B, p)
            tm = targ_all[:, start:start + config.batch_size]        # (K, B, r)
            yb = Yd[arangeK, idx]
            # Step 4: mixture-only pair access (centered form unless auditing "naive").
            if naive:
                pi = torch.sigmoid(U / config.tau)[:, None, :]
                H = pi * Vd[arangeK, idx] + (1.0 - pi) * Vtd[arangeK, idx]
            else:
                t = odd_tanh(U / (2.0 * config.tau))[:, None, :]
                H = mid_all[arangeK, idx] + t * half_all[arangeK, idx]
            H = H * (1.0 - cm)
            x = torch.cat([H, yb * (1.0 - tm), tm], dim=2) if use_ctx else H
            for li in range(len(weights)):
                x = torch.baddbmm(biases[li], x, weights[li])
                if li < len(weights) - 1:
                    x = torch.relu(x)
            if config.output_clip:
                x = torch.clamp(x, -config.output_clip, config.output_clip)
            # Step 5: masked-target reconstruction loss + symmetric entropy penalty.
            sq = (x - yb) ** 2 * tm
            loss_k = sq.sum(dim=(1, 2)) / tm.sum(dim=(1, 2))
            if naive:
                a = torch.sigmoid(U / config.tau)
                ent = torch.sum(a * torch.log(a) + (1.0 - a) * torch.log(1.0 - a), dim=1)
            else:
                tt = odd_tanh(U / (2.0 * config.tau))
                a = 0.5 * (1.0 + tt)
                b = 0.5 * (1.0 - tt)
                ent = torch.sum(a * torch.log(a) + b * torch.log(b), dim=1)
            total = (loss_k + lam * ent).sum()
            # Predeclared, data-independent cosine learning-rate schedule.
            factor = 0.5 * (1.0 + math.cos(math.pi * step / max(1, total_steps)))
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * factor
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if config.grad_clip and config.grad_clip > 0:
                # Global-norm clipping, computed separately for every member.
                with torch.no_grad():
                    sq_norm = U.grad.pow(2).sum(dim=1)
                    for prm in net_params:
                        sq_norm = sq_norm + prm.grad.pow(2).flatten(1).sum(dim=1)
                    coef = torch.clamp(config.grad_clip / (torch.sqrt(sq_norm) + 1e-6), max=1.0)
                    U.grad.mul_(coef[:, None])
                    for prm in net_params:
                        prm.grad.mul_(coef.view(K, *([1] * (prm.dim() - 1))))
            optimizer.step()
            with torch.no_grad():
                # Symmetric logit truncation |u_j| <= u_max (commutes with T_S).
                if config.logit_cap:
                    clamp_active += (U.abs() >= config.logit_cap).sum(dim=1)
                    U.clamp_(-config.logit_cap, config.logit_cap)
                # Projection of the augmented affine matrix [A b; 0 1] onto the
                # Frobenius ball (acts on network weights only).
                if caps is not None:
                    for li, (Wl, bl) in enumerate(zip(weights, biases)):
                        sqn = Wl.pow(2).sum(dim=(1, 2)) + bl.pow(2).sum(dim=(1, 2))
                        norm_aug = torch.sqrt(sqn + 1.0)
                        over = norm_aug > caps[li]
                        if bool(over.any()):
                            target = torch.sqrt(torch.clamp(caps[li] ** 2 - 1.0, min=1e-12))
                            scale = torch.where(over, target / torch.sqrt(sqn), torch.ones_like(sqn))
                            Wl.mul_(scale[:, None, None])
                            bl.mul_(scale[:, None, None])
                            projections += over.long()
                            sqn = Wl.pow(2).sum(dim=(1, 2)) + bl.pow(2).sum(dim=(1, 2))
                            norm_aug = torch.sqrt(sqn + 1.0)
                        max_frob[:, li] = torch.maximum(max_frob[:, li], norm_aug)
            last_loss = loss_k.detach()
            step += 1
    # Step 6: the signed statistic W = 2 pi - 1 = tanh(u / (2 tau)).
    with torch.no_grad():
        if naive:
            W = 2.0 * torch.sigmoid(U / config.tau) - 1.0
        else:
            W = odd_tanh(U / (2.0 * config.tau))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return BatchedFit(
        W=W.detach().double().cpu().numpy(),
        logits=U.detach().double().cpu().numpy(),
        final_loss=last_loss.double().cpu().numpy(),
        max_frobenius=max_frob.double().cpu().numpy(),
        cap_projections=projections.cpu().numpy(),
        logit_clamp_active=clamp_active.cpu().numpy(),
        seconds=time.perf_counter() - t_start,
    )
