"""
SSRK Training utilities with determinism and two-stage entropy schedule.
"""

import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple
import logging

from .model import SSRKModel


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic CUDA operations (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For newer PyTorch versions
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass  # Some operations don't support deterministic mode


def ssrk_loss_function(
    X: torch.Tensor,
    X_tilde: torch.Tensor,
    X_recon: torch.Tensor,
    M: torch.Tensor,
    gating_layer: nn.Module,
    lambda_entropy: float,
    stage: int = 2,
    stage1_entropy_max: Optional[float] = None,
    entropy_weighting: str = "uniform",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the SSRK loss: symmetric masked reconstruction + entropy regularization.

    L_SSRK = ||X_M - X^_M||^2 + lambda * R(gamma)

    where R(gamma) = -sum(pi_j * log(pi_j) + pitilde_j * log(pitilde_j))
    is the entropy regularizer that encourages deterministic gating at low entropy.

    Args:
        X: Original features (batch, p)
        X_tilde: Knockoff features (batch, p)
        X_recon: Reconstructed features (batch, p)
        M: Mask (1 = masked positions to predict)
        gating_layer: KnockoffGatedLayer for entropy computation
        lambda_entropy: Regularization weight
        stage: 1 = high-entropy phase, 2 = low-entropy phase
        stage1_entropy_max: H_max used in Stage-1 (defaults to p * log(2))

    Returns:
        total_loss, reconstruction_loss, regularization_loss
    """
    # 1) Symmetric masked reconstruction loss (gate-weighted competition between X / X_tilde)
    pi = gating_layer.get_gate_probabilities().unsqueeze(0)  # (1, p)
    mse_x = F.mse_loss(X_recon, X, reduction='none')
    mse_x_tilde = F.mse_loss(X_recon, X_tilde, reduction='none')
    sym_mse = pi * mse_x + (1.0 - pi) * mse_x_tilde
    masked_sym_mse = sym_mse * M
    loss_recon = torch.sum(masked_sym_mse) / (torch.sum(M) + 1e-9)
    
    # 2) Two-stage entropy regularization (Section 3.4.3)
    if entropy_weighting == "gap":
        entropy_vec = gating_layer.compute_entropy_regularization(reduction="none")
        gap = torch.mean(torch.abs(mse_x - mse_x_tilde), dim=0)
        gap_weight = gap / (gap.mean() + 1e-9)
        entropy = torch.sum(entropy_vec * gap_weight)
        if stage1_entropy_max is None:
            stage1_entropy_max = torch.sum(gap_weight) * math.log(2.0)
    else:
        entropy = gating_layer.compute_entropy_regularization()
        if stage1_entropy_max is None:
            stage1_entropy_max = gating_layer.p_features * math.log(2.0)
    
    if stage == 1:
        # Encourage high entropy: minimize (H_max - ΣH)
        loss_reg = stage1_entropy_max - entropy
    elif stage == 2:
        # Encourage low entropy: minimize ΣH
        loss_reg = entropy
    else:
        raise ValueError(f"Unknown training stage {stage}, expected 1 or 2.")

    total_loss = loss_recon + lambda_entropy * loss_reg
    
    return total_loss, loss_recon, loss_reg


class SSRKTrainer:
    """Trainer for the SSRK model with separate learning rates for gate and network parameters."""
    def __init__(
        self,
        model: SSRKModel,
        lr: float = 1e-3,
        lr_gate_factor: float = 0.1,
        lambda_entropy: float = 0.01,
        mask_prob: float = 0.5,
        device: torch.device = None,
        stage1_frac: float = 1/3,
        freeze_gates_stage1: bool = True,
        entropy_weighting: str = "uniform",
        mask_mode: str = "bernoulli",
        mask_shape: Optional[Tuple[int, int]] = None,
        mask_patch_size: int = 4,
        mask_extra_features: int = 0,
        mask_extra_prob: Optional[float] = None,
        gate_smoothness: float = 0.0,
        gate_smoothness_shape: Optional[Tuple[int, int]] = None,
        gate_balance: float = 0.0,
    ):
        """
        Args:
            model: SSRKModel instance
            lr: Learning rate for network parameters
            lr_gate_factor: Factor for gate parameter learning rate (lr * factor)
            lambda_entropy: Entropy regularization weight
            mask_prob: Probability of masking each feature
            device: Torch device
            stage1_frac: Fraction of epochs to keep high-entropy regularization
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(device)
        self.lambda_entropy = lambda_entropy
        self.mask_prob = mask_prob
        self.device = device
        self.stage1_frac = max(0.0, min(1.0, stage1_frac))
        self.freeze_gates_stage1 = freeze_gates_stage1
        self.entropy_weighting = entropy_weighting
        self.mask_mode = mask_mode
        self.mask_shape = mask_shape
        self.mask_patch_size = mask_patch_size
        self.mask_extra_features = max(0, int(mask_extra_features))
        self.mask_extra_prob = mask_extra_prob
        self.gate_smoothness = gate_smoothness
        self.gate_smoothness_shape = gate_smoothness_shape
        self.gate_balance = gate_balance
        
        # Verify symmetric initialization (A.2)
        if not torch.allclose(
            self.model.gating_layer.gate_logits, 
            torch.zeros_like(self.model.gating_layer.gate_logits)
        ):
            logging.warning("A.2 Check Failed: Gate logits not symmetric. Re-initializing.")
            self.model.gating_layer.initialize_symmetrically()
        
        # Separate gate and network parameters for different learning rates
        gate_params = [self.model.gating_layer.gate_logits]
        network_params = [p for n, p in self.model.named_parameters() if 'gate_logits' not in n]
        
        self.gate_lr = lr * lr_gate_factor
        self.optimizer = optim.Adam([
            {'params': network_params, 'lr': lr},
            {'params': gate_params, 'lr': self.gate_lr}
        ])
        self._gate_param_group = self.optimizer.param_groups[1]
        
        self.scheduler = None  # Will be set in train()

    def _generate_mask(self, batch_shape: Tuple[int, int]) -> torch.Tensor:
        """Generate mask tensor based on configured mask mode."""
        if self.mask_mode != "block":
            return torch.bernoulli(
                torch.full(batch_shape, self.mask_prob, device=self.device)
            )

        if not self.mask_shape:
            raise ValueError("mask_shape is required for block masking.")

        height, width = self.mask_shape
        patch = max(1, int(self.mask_patch_size))
        batch_size = batch_shape[0]
        grid_h = int(math.ceil(height / patch))
        grid_w = int(math.ceil(width / patch))
        grid = torch.bernoulli(
            torch.full((batch_size, grid_h, grid_w), self.mask_prob, device=self.device)
        )
        mask = grid.repeat_interleave(patch, dim=1).repeat_interleave(patch, dim=2)
        mask = mask[:, :height, :width]
        mask = mask.view(batch_size, -1)
        if self.mask_extra_features > 0:
            extra_prob = self.mask_prob if self.mask_extra_prob is None else self.mask_extra_prob
            extra = torch.bernoulli(
                torch.full((batch_size, self.mask_extra_features), extra_prob, device=self.device)
            )
            mask = torch.cat([mask, extra], dim=1)
        return mask

    def _gate_smoothness_loss(self) -> torch.Tensor:
        """Spatial smoothness penalty on gate logits for image-like layouts."""
        if self.gate_smoothness <= 0.0 or not self.gate_smoothness_shape:
            return torch.tensor(0.0, device=self.device)

        height, width = self.gate_smoothness_shape
        logits = self.model.gating_layer.gate_logits
        base_features = height * width
        if logits.numel() < base_features:
            return torch.tensor(0.0, device=self.device)

        grid = logits[:base_features].view(1, 1, height, width)
        diff_h = grid[:, :, 1:, :] - grid[:, :, :-1, :]
        diff_w = grid[:, :, :, 1:] - grid[:, :, :, :-1]
        return (diff_h.pow(2).mean() + diff_w.pow(2).mean())

    def _gate_balance_loss(self) -> torch.Tensor:
        """Penalty that keeps average gate probability near 0.5."""
        if self.gate_balance <= 0.0:
            return torch.tensor(0.0, device=self.device)
        pi = self.model.gating_layer.get_gate_probabilities()
        return (pi.mean() - 0.5).pow(2)

    def train(
        self,
        X: np.ndarray,
        X_tilde: np.ndarray,
        epochs: int = 200,
        batch_size: int = 128,
        verbose: bool = False,
        log_interval: int = None,
    ) -> dict:
        """
        Train the SSRK model.
        
        Args:
            X: Original features (n, p)
            X_tilde: Knockoff features (n, p)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to log progress
            log_interval: Epochs between log messages (default: epochs/10)
        
        Returns:
            Dictionary with training history
        """
        if log_interval is None:
            log_interval = max(1, epochs // 10)
        
        # Stage-1 duration (high entropy), remaining epochs use Stage-2 (low entropy)
        stage1_epochs = max(1, int(epochs * self.stage1_frac)) if epochs > 1 else 1
        
        # Setup data
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(X_tilde, dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Cosine annealing LR scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        if verbose:
            logging.info(f"Starting SSRK Training for {epochs} epochs on {self.device}...")
        
        self.model.train()
        history = {'loss': [], 'loss_recon': [], 'loss_reg': [], 'W_range': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_reg = 0.0
            n_batches = 0
            
            stage = 1 if epoch < stage1_epochs else 2
            if stage == 1 and self.freeze_gates_stage1:
                self._gate_param_group['lr'] = 0.0
            else:
                self._gate_param_group['lr'] = self.gate_lr

            for X_batch, Xk_batch in dataloader:
                X_batch = X_batch.to(self.device)
                Xk_batch = Xk_batch.to(self.device)
                
                # Generate random mask dynamically
                M = self._generate_mask(X_batch.shape)
                
                self.optimizer.zero_grad()
                X_recon = self.model(X_batch, Xk_batch, M)
                
                # Two-stage entropy schedule + symmetric loss
                loss, loss_recon, loss_reg = ssrk_loss_function(
                    X_batch, Xk_batch, X_recon, M, 
                    self.model.gating_layer,
                    self.lambda_entropy,
                    stage=stage,
                    entropy_weighting=self.entropy_weighting,
                )

                smooth_loss = self._gate_smoothness_loss()
                if self.gate_smoothness > 0.0:
                    loss = loss + self.gate_smoothness * smooth_loss
                balance_loss = self._gate_balance_loss()
                if self.gate_balance > 0.0:
                    loss = loss + self.gate_balance * balance_loss
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += loss_recon.item()
                epoch_reg += loss_reg.item()
                n_batches += 1
            
            self.scheduler.step()
            
            # Record history
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_reg = epoch_reg / n_batches
            W = self.model.gating_layer.compute_statistics()
            
            history['loss'].append(avg_loss)
            history['loss_recon'].append(avg_recon)
            history['loss_reg'].append(avg_reg)
            history['W_range'].append((W.min(), W.max()))
            
            if verbose and (epoch + 1) % log_interval == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Loss: {avg_loss:.4f}, W range: [{W.min():.3f}, {W.max():.3f}]"
                )

        if verbose:
            logging.info("Training finished.")
        
        return history
    
    def get_W_statistics(self, original_in_first_mask=None) -> np.ndarray:
        """Get slot-aware W statistics from the trained model."""
        return self.model.get_W_statistics(
            original_in_first_mask=original_in_first_mask
        )
