"""Self-reconstruction (ranking) mode of SSRK, used by the real-data studies (PBMC 3k, UCI HAR).

A multilayer-perceptron autoencoder with batch normalization reconstructs the candidate vector
itself (paper App. A.3.1 "Reference constructions and self-reconstruction training"):

* **Gate** (eq. (4), evaluated in the centered form of Remark 1 "Centered gate")::

      H_j = (X_j + X~_j)/2 + t_j (X_j - X~_j)/2,     t_j = OddTanh(u_j / (2 tau)) = 2 pi_j - 1,

  with paired weights ``a_j = (1 + t_j)/2 = pi_j`` and ``b_j = (1 - t_j)/2 = 1 - pi_j``. Gate logits
  start at zero (``pi = 1/2``).
* **Masks**: data-independent Bernoulli(``mask_prob``) masks ``M`` (1 = hidden), shared by the pair;
  the encoder receives ``H * (1 - M)``.
* **Loss** over the masked entries: ``a_j ||X^_j - X_j||^2 + b_j ||X^_j - X~_j||^2`` (normalized by the
  number of masked entries) plus the two-stage entropy term. Each gate's binary entropy is weighted
  by its normalized reconstruction gap (a swap-invariant weight; exact formula in
  :func:`ssrk_loss_function`): ``w_j = g_j / (p^-1 sum_k g_k + 1e-9)`` with
  ``g_j = n_b^-1 sum_i |e_ij - e~_ij|``, ``e_ij = (X^_ij - X_ij)^2``, ``e~_ij = (X^_ij - X~_ij)^2``. The
  mean runs over all ``n_b`` rows of the minibatch, masked and visible entries alike, and ``w`` is not
  detached, so the entropy term also back-propagates into the network.
  Stage I (the first ``stage1_frac`` of the epochs) freezes the gates at ``pi = 1/2`` and uses
  ``lambda_ent (H_max - sum_j w_j H(pi_j))``. The freeze sets the learning rate of the gate parameter
  group of the shared Adam optimizer to 0: the gate gradients are still computed, count toward the
  global clip norm, and update the gate's Adam moments and step count, but the logits do not move.
  Stage II trains the gates with their own learning rate ``lr * lr_gate_factor`` and penalizes
  ``lambda_ent sum_j w_j H(pi_j)``, i.e. coefficient ``-lambda_ent`` on the negative entropy ``h`` of
  eq. (6).
* **Optimization**: Adam with two parameter groups. The network group follows a cosine learning-rate
  decay (``CosineAnnealingLR`` with ``T_max = epochs``, stepped once per epoch). The gate group's
  learning rate is reset at the start of every epoch (0 in Stage I, ``lr * lr_gate_factor`` in
  Stage II), so it stays constant within each stage and does not follow the cosine schedule. Gradients
  are clipped to global norm 1.0 over all parameters, and minibatches come from a shuffling
  ``DataLoader``.
* **Statistic** (eq. (7)): ``W_j = 2 pi_j - 1 = t_j``.

Every floating-point operation, its order, and every random draw (torch CPU generator for the network
initialization and the ``DataLoader`` permutations, the device generator for the masks) follow the
original research implementation, so that the published records are reproduced bitwise on the
reference machine. Because the centered form only uses sign-symmetric IEEE operations, retraining on
``swap_S(D)`` with the same randomness gives ``W(swap_S D) = T_S W(D)`` exactly (Lemma 2, eq. (18));
``tests/test_selfrecon.py`` checks this bitwise.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .determinism import set_seed
from .gates import odd_tanh
from .references import generate_reference

#: Offset between the seeds of consecutive restarts on one paired design (Table A1, step 7).
RESTART_SEED_STRIDE = 100_003


@dataclass(frozen=True)
class SelfReconConfig:
    """Resolved self-reconstruction hyperparameters (see the protocol presets in ``experiments/``)."""

    encoder_dims: Sequence[int]
    latent_dim: int
    decoder_dims: Sequence[int]
    temperature: float
    use_batchnorm: bool
    epochs: int
    batch_size: int
    lr: float
    lr_gate_factor: float
    lambda_entropy: float
    mask_prob: float
    stage1_frac: float
    freeze_gates_stage1: bool = True
    entropy_weighting: str = "gap"
    gate_parameterization: str = "symmetric"
    mask_mode: str = "bernoulli"

    def __post_init__(self):
        if self.entropy_weighting != "gap":
            raise ValueError("only the gap-weighted entropy schedule of the paper is implemented")
        if self.gate_parameterization != "symmetric":
            raise ValueError("only the sign-symmetric (centered) gate of the paper is implemented")
        if self.mask_mode != "bernoulli":
            raise ValueError("only Bernoulli masks are used by the PBMC / UCI HAR protocols")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["encoder_dims"] = list(self.encoder_dims)
        d["decoder_dims"] = list(self.decoder_dims)
        return d


# ----------------------------------------------------------------------------------------- model
class KnockoffGatedLayer(nn.Module):
    """Knockoff gate (eq. (4)) in the centered, sign-symmetric form (Remark 1 "Centered gate").

    One logit ``u_j`` per candidate pair, initialized at zero. ``t = OddTanh(u / (2 tau))`` is exactly
    odd in ``u``, so swapping a pair and negating its logit leaves the mixture ``H`` bitwise unchanged
    and exchanges the paired weights ``a = (1 + t)/2`` and ``b = (1 - t)/2`` exactly.
    """

    def __init__(self, p_features: int, temperature: float = 1.0):
        super().__init__()
        self.p_features = p_features
        self.temperature = temperature
        # zero (symmetric) initialization: pi_j = 1/2, no preference for X_j or X~_j
        self.gate_logits = nn.Parameter(torch.zeros(p_features))

    def initialize_symmetrically(self) -> None:
        nn.init.zeros_(self.gate_logits)

    def gate_t(self) -> torch.Tensor:
        """``t = 2 pi - 1 = tanh(u / (2 tau))``, exactly odd in ``u``."""
        return odd_tanh(self.gate_logits / (2.0 * self.temperature))

    def get_gate_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Paired weights ``(a, b) = ((1 + t)/2, (1 - t)/2) = (pi, 1 - pi)``."""
        t = self.gate_t()
        return 0.5 * (1.0 + t), 0.5 * (1.0 - t)

    def forward(self, X: torch.Tensor, X_tilde: torch.Tensor) -> torch.Tensor:
        """Mixture ``H = (X + X~)/2 + t (X - X~)/2`` (batch, p)."""
        t = self.gate_t().unsqueeze(0)
        return 0.5 * (X + X_tilde) + t * (0.5 * (X - X_tilde))

    def compute_entropy_regularization(self, reduction: str = "sum") -> torch.Tensor:
        """Binary entropy ``H(pi_j) = -a log a - b log b`` (``= -h(pi_j)`` of eq. (6)).

        Computed from ``a`` and ``b`` (never from ``1 - a``), so its value is exactly even in ``t``.
        """
        eps = 1e-7
        a, b = self.get_gate_weights()
        entropy = -a * torch.log(a + eps) - b * torch.log(b + eps)
        if reduction == "none":
            return entropy
        if reduction == "sum":
            return entropy.sum()
        raise ValueError(f"Unknown reduction: {reduction}")

    def compute_statistics(self) -> np.ndarray:
        """Signed statistic ``W = 2 pi - 1 = t`` (eq. (7)), float32."""
        return self.gate_t().detach().cpu().numpy()


class SSRKModel(nn.Module):
    """Knockoff-gated masked autoencoder: gate -> mask -> MLP encoder/decoder with BatchNorm."""

    def __init__(
        self,
        p_features: int,
        encoder_dims: Sequence[int],
        latent_dim: int,
        decoder_dims: Sequence[int],
        temperature: float = 1.0,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.p_features = p_features

        # module construction order fixes the parameter initialization order (torch CPU generator)
        self.gating_layer = KnockoffGatedLayer(p_features, temperature)

        enc_layers = []
        input_dim = p_features
        for dim in encoder_dims:
            enc_layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm:
                enc_layers.append(nn.BatchNorm1d(dim))
            enc_layers.append(nn.ReLU())
            input_dim = dim
        enc_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        input_dim = latent_dim
        for dim in decoder_dims:
            dec_layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm:
                dec_layers.append(nn.BatchNorm1d(dim))
            dec_layers.append(nn.ReLU())
            input_dim = dim
        dec_layers.append(nn.Linear(input_dim, p_features))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, X: torch.Tensor, X_tilde: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Reconstruct the candidates from the masked mixture ``H * (1 - M)`` (``M = 1``: hidden)."""
        H = self.gating_layer(X, X_tilde)
        H_masked = H * (1.0 - M)
        latent = self.encoder(H_masked)
        return self.decoder(latent)

    def get_W_statistics(self) -> np.ndarray:
        return self.gating_layer.compute_statistics()


# ------------------------------------------------------------------------------------------ loss
def ssrk_loss_function(
    X: torch.Tensor,
    X_tilde: torch.Tensor,
    X_recon: torch.Tensor,
    M: torch.Tensor,
    gating_layer: KnockoffGatedLayer,
    lambda_entropy: float,
    stage: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric masked self-reconstruction loss with the gap-weighted two-stage entropy term.

    ``loss_recon = sum(M * (a mse(X^, X) + b mse(X^, X~))) / (sum(M) + 1e-9)``;
    ``w_j = gap_j / (mean(gap) + 1e-9)`` with ``gap_j = mean_i |mse_ij(X) - mse_ij(X~)|``. The mean over
    ``i`` runs over all minibatch entries of column ``j``, masked and visible (``M`` does not enter the
    gap), and ``w`` is not detached, so the entropy term also back-propagates into the network;
    Stage I: ``loss_reg = log(2) sum_j w_j - sum_j w_j H(pi_j)``; Stage II: ``loss_reg = sum_j w_j H(pi_j)``.

    Returns ``(total_loss, loss_recon, loss_reg)`` with ``total = loss_recon + lambda_entropy * loss_reg``.
    """
    # 1) symmetric masked reconstruction loss (gate-weighted competition between X and X~)
    a, b = gating_layer.get_gate_weights()
    a, b = a.unsqueeze(0), b.unsqueeze(0)  # (1, p)
    mse_x = F.mse_loss(X_recon, X, reduction="none")
    mse_x_tilde = F.mse_loss(X_recon, X_tilde, reduction="none")
    sym_mse = a * mse_x + b * mse_x_tilde
    masked_sym_mse = sym_mse * M
    loss_recon = torch.sum(masked_sym_mse) / (torch.sum(M) + 1e-9)

    # 2) two-stage entropy regularization, each gate weighted by its normalized reconstruction gap
    entropy_vec = gating_layer.compute_entropy_regularization(reduction="none")
    gap = torch.mean(torch.abs(mse_x - mse_x_tilde), dim=0)
    gap_weight = gap / (gap.mean() + 1e-9)
    entropy = torch.sum(entropy_vec * gap_weight)
    stage1_entropy_max = torch.sum(gap_weight) * math.log(2.0)

    if stage == 1:
        loss_reg = stage1_entropy_max - entropy  # pulls pi toward 1/2
    elif stage == 2:
        loss_reg = entropy  # sharpens the gates
    else:
        raise ValueError(f"Unknown training stage {stage}, expected 1 or 2.")

    total_loss = loss_recon + lambda_entropy * loss_reg
    return total_loss, loss_recon, loss_reg


# --------------------------------------------------------------------------------------- trainer
class SSRKTrainer:
    """Two-stage trainer: Adam with a separate gate learning rate, cosine decay, clipping at 1.0.

    One ``torch.optim.Adam`` holds two parameter groups: the network (base ``lr``, cosine-annealed by
    ``CosineAnnealingLR(T_max=epochs)``, stepped once per epoch) and the gate logits. The gate group's
    learning rate is set at the start of every epoch: 0 during a frozen Stage I and the constant
    ``lr * lr_gate_factor`` in Stage II (the scheduler's per-epoch update of this group is overwritten,
    so it never follows the cosine schedule). With learning rate 0 the gate logits stay exactly at 0,
    but their gradients still enter the global gradient norm that is clipped to 1.0, and Adam still
    accumulates their first/second moments and step count, which carry over into Stage II.
    """

    def __init__(
        self,
        model: SSRKModel,
        lr: float,
        lr_gate_factor: float,
        lambda_entropy: float,
        mask_prob: float,
        device: torch.device,
        stage1_frac: float,
        freeze_gates_stage1: bool = True,
    ):
        self.model = model.to(device)
        self.lambda_entropy = lambda_entropy
        self.mask_prob = mask_prob
        self.device = device
        self.stage1_frac = max(0.0, min(1.0, stage1_frac))
        self.freeze_gates_stage1 = freeze_gates_stage1

        # zero gate initialization (fixed by every sign flip T_S)
        if not torch.allclose(
            self.model.gating_layer.gate_logits,
            torch.zeros_like(self.model.gating_layer.gate_logits),
        ):
            logging.warning("Gate logits not at zero; re-initializing symmetrically.")
            self.model.gating_layer.initialize_symmetrically()

        # separate parameter groups: network (base lr, cosine decay) and gate logits (lr * factor)
        gate_params = [self.model.gating_layer.gate_logits]
        network_params = [p for n, p in self.model.named_parameters() if "gate_logits" not in n]
        self.gate_lr = lr * lr_gate_factor
        self.optimizer = optim.Adam([
            {"params": network_params, "lr": lr},
            {"params": gate_params, "lr": self.gate_lr},
        ])
        self._gate_param_group = self.optimizer.param_groups[1]
        self.scheduler = None

    def _generate_mask(self, batch_shape: torch.Size) -> torch.Tensor:
        """Data-independent Bernoulli mask (1 = hidden entry), drawn from the device generator."""
        return torch.bernoulli(torch.full(batch_shape, self.mask_prob, device=self.device))

    def train(self, X: np.ndarray, X_tilde: np.ndarray, epochs: int, batch_size: int) -> Dict[str, list]:
        """Train on the paired design ``(X, X_tilde)``; returns per-epoch mean losses."""
        # Stage-I duration (gates frozen at pi = 1/2); the remaining epochs are Stage II
        stage1_epochs = max(1, int(epochs * self.stage1_frac)) if epochs > 1 else 1

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(X_tilde, dtype=torch.float32),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.model.train()
        history: Dict[str, list] = {"loss": [], "loss_recon": [], "loss_reg": []}
        for epoch in range(epochs):
            epoch_loss = epoch_recon = epoch_reg = 0.0
            n_batches = 0

            stage = 1 if epoch < stage1_epochs else 2
            if stage == 1 and self.freeze_gates_stage1:
                self._gate_param_group["lr"] = 0.0
            else:
                self._gate_param_group["lr"] = self.gate_lr

            for X_batch, Xk_batch in dataloader:
                X_batch = X_batch.to(self.device)
                Xk_batch = Xk_batch.to(self.device)
                M = self._generate_mask(X_batch.shape)

                self.optimizer.zero_grad()
                X_recon = self.model(X_batch, Xk_batch, M)
                loss, loss_recon, loss_reg = ssrk_loss_function(
                    X_batch, Xk_batch, X_recon, M,
                    self.model.gating_layer,
                    self.lambda_entropy,
                    stage=stage,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += loss_recon.item()
                epoch_reg += loss_reg.item()
                n_batches += 1

            self.scheduler.step()
            history["loss"].append(epoch_loss / n_batches)
            history["loss_recon"].append(epoch_recon / n_batches)
            history["loss_reg"].append(epoch_reg / n_batches)
        return history

    def get_W_statistics(self) -> np.ndarray:
        return self.model.get_W_statistics()


# ----------------------------------------------------------------------------------- entry points
def fit_ssrk_with_knockoffs(
    X: np.ndarray,
    X_tilde: np.ndarray,
    seed: int,
    config: SelfReconConfig,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """One self-reconstruction fit on the paired design ``(X, X_tilde)``; returns ``(W float32, meta)``.

    ``set_seed(seed)`` fixes the network initialization, the minibatch order and the masks.
    """
    set_seed(seed, deterministic=True)
    model = SSRKModel(
        p_features=X.shape[1],
        encoder_dims=config.encoder_dims,
        latent_dim=config.latent_dim,
        decoder_dims=config.decoder_dims,
        temperature=config.temperature,
        use_batchnorm=config.use_batchnorm,
    )
    trainer = SSRKTrainer(
        model,
        lr=config.lr,
        lr_gate_factor=config.lr_gate_factor,
        lambda_entropy=config.lambda_entropy,
        mask_prob=config.mask_prob,
        device=device,
        stage1_frac=config.stage1_frac,
        freeze_gates_stage1=config.freeze_gates_stage1,
    )
    start = time.time()
    history = trainer.train(X, X_tilde, epochs=config.epochs, batch_size=config.batch_size)
    W = trainer.get_W_statistics()
    return W.astype(np.float32), {"train_time_sec": float(time.time() - start),
                                  "final_loss": float(history["loss"][-1])}


def ssrk_scores(
    X_fit: np.ndarray,
    fit_seed: int,
    reference: str,
    config: SelfReconConfig,
    device: torch.device,
    restarts: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """SSRK ranking statistics for one unit of a matched protocol.

    1. Draw ONE reference ``X_tilde`` for ``X_fit`` (``reference`` = ``"independent"`` or
       ``"permutation"``, seed ``fit_seed``); the paired design is then fixed for all restarts.
    2. Fit ``restarts`` models with predeclared seeds ``fit_seed + 100003 r`` (restart 0 uses ``fit_seed``).
    3. Aggregate by the coordinatewise mean (an odd rule, Table A1, step 7), in float64.

    Returns:
        ``(W_bar float32 (p,), W_all float32 (restarts, p), meta)``.
    """
    set_seed(fit_seed, deterministic=True)
    X_tilde, gmeta = generate_reference(reference, X_fit, fit_seed)
    Ws = []
    t0 = time.perf_counter()
    for r in range(int(restarts)):
        rs = fit_seed + RESTART_SEED_STRIDE * r
        W, _ = fit_ssrk_with_knockoffs(X_fit, X_tilde, rs, config, device)
        Ws.append(np.asarray(W, dtype=np.float64))
    Wall = np.stack(Ws)
    Wbar = Wall.mean(axis=0)
    meta = dict(fit_seed=int(fit_seed), restarts=int(restarts), reference=reference,
                seconds=time.perf_counter() - t0, generator_seconds=gmeta.get("generator_time_sec"))
    return Wbar.astype(np.float32), Wall.astype(np.float32), meta


def resolve_device(requested: str = "auto") -> torch.device:
    """``"cuda"`` (error if unavailable), ``"cpu"``, or ``"auto"`` (cuda if available).

    The published records were produced on CUDA (RTX 4090 Laptop GPU); CPU runs are valid but are not
    expected to match them bitwise.
    """
    choice = (requested or "auto").strip().lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available (use --device cpu)")
        return torch.device("cuda")
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logging.warning("CUDA not available: running on CPU (results will not match the GPU records bitwise)")
        return torch.device("cpu")
    raise ValueError(f"Unknown device {requested!r}")
