"""
Normalization modules for DyT vs RMSNorm comparison.

DyT (Dynamic Tanh): y = gamma * tanh(alpha * x) + beta
  - Purely elementwise (no cross-channel reduction)
  - Enables kernel fusion (the key motivation of proposal 063)
  - alpha is a learnable scalar, gamma/beta are per-channel

RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps)
  - Requires cross-channel reduction (mean over d)
  - Standard baseline normalization for linear RNNs
"""

import torch
import torch.nn as nn
import math


class DyT(nn.Module):
    """
    Dynamic Tanh normalization (Zhu et al., CVPR 2025).

    y = gamma * tanh(alpha * x) + beta

    - alpha: learnable scalar, initialized based on hidden dim (trick 241)
    - gamma: per-channel scale, initialized to 1
    - beta: per-channel bias, initialized to 0

    No cross-channel reduction => can be fused into adjacent kernels.
    """

    def __init__(self, d: int, alpha_init: float = None):
        super().__init__()
        # Width-dependent alpha initialization (trick 241)
        if alpha_init is None:
            alpha_init = 1.0 / math.sqrt(d)

        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d) or (batch, d)
        Returns:
            Normalized tensor of same shape
        """
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    y = gamma * x / sqrt(mean(x^2) + eps)

    Requires cross-channel reduction (mean over d dimension).
    """

    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d) or (batch, d)
        Returns:
            Normalized tensor of same shape
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms
