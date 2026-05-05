"""
SSRK Core Model: Knockoff-Gated Autoencoder
核心架构：Knockoff-Gated Autoencoder (Sections 3.3 & 3.4)
"""

import torch
import torch.nn as nn

from .statistics import compute_W_statistics


class KnockoffGatedLayer(nn.Module):
    """
    实现 Section 3.3.2 的 Knockoff-gated 架构。
    
    论文描述使用双参数 (s_j, s̃_j) 的 softmax:
        π_j = exp(s_j) / (exp(s_j) + exp(s̃_j))
    
    数学等价简化：由于对称初始化 s_j = s̃_j = 0，我们可以使用单参数 sigmoid:
        π_j = σ(s_j - s̃_j) = σ(s_j)  (当 s̃_j 固定为 0)

    最终导出的统计量采用 slot-aware 对象：
        α_j = π_j            (原始特征位于第一槽位)
        α_j = 1 - π_j        (原始特征位于第二槽位)
        W_j = 2α_j - 1
    """
    def __init__(self, p_features: int, temperature: float = 1.0):
        super().__init__()
        self.p_features = p_features
        self.temperature = temperature
        
        # 门控逻辑参数 s_j，满足对称初始化要求 (论文 Section 3.4.3, Assumption A.2)
        # 初始化为零确保 π_j = 0.5，即对 X_j 和 X̃_j 无偏好
        self.gate_logits = nn.Parameter(torch.zeros(p_features))
        
    def initialize_symmetrically(self):
        """Critical for A.2: symmetric initialization at zero."""
        nn.init.zeros_(self.gate_logits)

    def get_gate_probabilities(self) -> torch.Tensor:
        """返回 (pi, pi_tilde) 其中 pi + pi_tilde = 1"""
        pi = torch.sigmoid(self.gate_logits / self.temperature)
        return pi

    def forward(self, X: torch.Tensor, X_tilde: torch.Tensor) -> torch.Tensor:
        """
        计算门控输入 H_j = pi_j * X_j + (1-pi_j) * X_tilde_j
        
        Args:
            X: Original features (batch, p)
            X_tilde: Knockoff features (batch, p)
        Returns:
            H: Gated features (batch, p)
        """
        pi = self.get_gate_probabilities().unsqueeze(0)  # (1, P)
        H = pi * X + (1.0 - pi) * X_tilde
        return H

    def compute_entropy_regularization(self, reduction: str = "sum") -> torch.Tensor:
        """
        计算熵正则项 R(γ) (论文 Section 4.3.1)。
        
        论文公式: R(γ) = -Σ(π_j log π_j + π̃_j log π̃_j)
        
        由于 π̃_j = 1 - π_j，这等价于二元熵的总和:
            R(γ) = Σ H(π_j) = -Σ(π_j log π_j + (1-π_j) log(1-π_j))
        
        最小化 R(γ) 鼓励确定性门控（低熵），使门控偏向 0 或 1。
        """
        pi = self.get_gate_probabilities()
        eps = 1e-7
        # Binary entropy: H(π) = -π*log(π) - (1-π)*log(1-π)
        entropy = -pi * torch.log(pi + eps) - (1 - pi) * torch.log(1 - pi + eps)
        if reduction == "none":
            return entropy
        if reduction == "sum":
            return entropy.sum()
        raise ValueError(f"Unknown reduction: {reduction}")

    def compute_statistics(self, original_in_first_mask=None) -> 'np.ndarray':
        """
        计算 slot-aware 特征统计量 W (Section 3.4.4)。
        """
        pi = self.get_gate_probabilities().detach()
        return compute_W_statistics(
            pi.cpu().numpy(),
            original_in_first_mask=original_in_first_mask,
        )


class SSRKModel(nn.Module):
    """
    实现 Section 3.4.3 的 Knockoff-gated Autoencoder。
    """
    def __init__(
        self,
        p_features: int,
        encoder_dims: list = None,
        latent_dim: int = 20,
        decoder_dims: list = None,
        temperature: float = 1.0,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.p_features = p_features
        
        if encoder_dims is None:
            encoder_dims = [128, 64]
        if decoder_dims is None:
            decoder_dims = [64, 128]
        
        # 1. Gating Layer
        self.gating_layer = KnockoffGatedLayer(p_features, temperature)
        
        # 2. Encoder with optional BatchNorm for stability
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
        
        # 3. Decoder
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

    def forward(
        self, 
        X: torch.Tensor, 
        X_tilde: torch.Tensor, 
        M: torch.Tensor
    ) -> torch.Tensor:
        """
        实现 Masked Reconstruction 任务 (Section 3.4.2)。
        
        Args:
            X: Original features (batch, p)
            X_tilde: Knockoff features (batch, p)
            M: Binary mask (batch, p), 1 = masked (to predict), 0 = visible
        Returns:
            X_recon: Reconstructed features (batch, p)
        """
        # 1. 生成门控输入 H
        H = self.gating_layer(X, X_tilde)
        
        # 2. 应用掩码
        H_masked = H * (1.0 - M)
        
        # 3. 编码和解码
        latent = self.encoder(H_masked)
        X_recon = self.decoder(latent)
        
        return X_recon
    
    def get_W_statistics(self, original_in_first_mask=None) -> 'np.ndarray':
        """Convenience method to get slot-aware W statistics from gating layer."""
        return self.gating_layer.compute_statistics(
            original_in_first_mask=original_in_first_mask
        )


class SSRKConvModel(nn.Module):
    """
    Convolutional SSRK model for image-like inputs (e.g., MNIST 28x28).
    Keeps the same gating + symmetric reconstruction framework.
    """
    def __init__(
        self,
        p_features: int,
        image_shape: tuple = (1, 28, 28),
        encoder_channels: tuple = (16, 32),
        latent_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.p_features = p_features
        self.image_shape = image_shape

        if p_features != image_shape[1] * image_shape[2]:
            raise ValueError("p_features must match image_shape for SSRKConvModel.")

        c1, c2 = encoder_channels
        h, w = image_shape[1], image_shape[2]
        if h % 4 != 0 or w % 4 != 0:
            raise ValueError("image_shape must be divisible by 4 for SSRKConvModel.")

        # Gating Layer
        self.gating_layer = KnockoffGatedLayer(p_features, temperature)

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(image_shape[0], c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        enc_h, enc_w = h // 4, w // 4
        self.encoder_fc = nn.Linear(c2 * enc_h * enc_w, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, c2 * enc_h * enc_w)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(c1, image_shape[0], kernel_size=2, stride=2),
        )

    def forward(
        self,
        X: torch.Tensor,
        X_tilde: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        # Gated input and masking
        H = self.gating_layer(X, X_tilde)
        H_masked = H * (1.0 - M)

        # Reshape to image
        batch_size = H_masked.shape[0]
        H_img = H_masked.view(batch_size, *self.image_shape)

        # Encode / decode
        enc = self.encoder_conv(H_img)
        enc = enc.view(batch_size, -1)
        latent = self.encoder_fc(enc)
        dec = self.decoder_fc(latent)
        dec = dec.view(batch_size, -1, self.image_shape[1] // 4, self.image_shape[2] // 4)
        out = self.decoder_conv(dec)

        return out.view(batch_size, -1)

    def get_W_statistics(self, original_in_first_mask=None) -> 'np.ndarray':
        """Convenience method to get slot-aware W statistics from gating layer."""
        return self.gating_layer.compute_statistics(
            original_in_first_mask=original_in_first_mask
        )
