"""
Circulant SSM with Fourier-Domain Parallel Scan

From proposal 013-circulant-ssm-fourier-domain-scan:

The key insight is that a circulant matrix A = F^{-1} diag(a_hat) F
is diagonalized by the DFT. So the recurrence:
    h_t = A(x_t) h_{t-1} + B_t x_t
becomes in Fourier domain:
    h_hat_t = diag(a_hat(x_t)) h_hat_{t-1} + B_hat_t x_hat_t

This is n independent scalar recurrences, each amenable to parallel scan.

Input-dependent gating (proposal eq.):
    a(x_t) = gamma * tanh(W_a x_t + b_a)   (circulant defining vector, REAL)
    a_hat(x_t) = FFT(a(x_t))                (Fourier eigenvalues, conjugate-symmetric)

IMPORTANT: The circulant defining vector a(x_t) must be REAL so that a_hat is
conjugate-symmetric, ensuring the circulant matrix A is real-valued and
the hidden state h remains real after IFFT.

Stability: since ||a||_inf <= gamma < 1 and spectral radius of circulant = max|a_hat_i|,
we need to control the spectral radius. For a real vector a with ||a||_inf <= gamma,
we have |a_hat_i| <= n * gamma (by triangle inequality on DFT). But in practice
with learned vectors, the spectral radius is usually much smaller.

Alternative stability: normalize a_hat so max|a_hat_i| < 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CirculantSSMLayer(nn.Module):
    """
    Single Circulant SSM layer.

    Parameterizes the circulant defining vector in spatial domain (real-valued),
    then transforms to Fourier domain for element-wise parallel scan.

    The FFT of a real vector is conjugate-symmetric, ensuring the hidden state
    remains real-valued after IFFT.

    Args:
        d_model: input/output dimension
        state_dim: SSM state dimension n (should be power of 2 for FFT efficiency)
    """

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim  # n

        # Input-dependent circulant defining vector (REAL, spatial domain)
        # a(x_t) = gamma * tanh(W_a x_t + b_a) in R^n
        # Then a_hat = FFT(a) is automatically conjugate-symmetric
        self.W_a = nn.Linear(d_model, state_dim)

        # Stability: learnable scaling gamma in (0, 1)
        # We use sigmoid on a learnable scalar
        self.log_gamma = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        # Input projection: B_t x_t (project input to state space)
        self.W_B = nn.Linear(d_model, state_dim)

        # Output projection: C_t^T h_t (project state back to model dim)
        self.W_C = nn.Linear(state_dim, d_model)

        # Skip connection (like Mamba's D parameter)
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        # Initialize W_a for moderate decay
        nn.init.normal_(self.W_a.weight, std=0.02)
        nn.init.zeros_(self.W_a.bias)

        # Small init for input/output projections
        nn.init.normal_(self.W_B.weight, std=0.02)
        nn.init.normal_(self.W_C.weight, std=0.02)

    def _get_fourier_eigenvalues(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute conjugate-symmetric Fourier eigenvalues from input.

        1. Project input to spatial-domain circulant defining vector a(x_t)
        2. Apply tanh for boundedness, scale by gamma for stability
        3. FFT to get eigenvalues (automatically conjugate-symmetric since a is real)
        4. Normalize so max|a_hat_i| < 1 for stability

        Args:
            x: (B, T, d_model)
        Returns:
            a_hat: (B, T, n) complex, conjugate-symmetric eigenvalues with |a_hat_i| < 1
        """
        # Circulant defining vector in spatial domain (REAL)
        gamma = torch.sigmoid(self.log_gamma)  # in (0, 1)
        a = gamma * torch.tanh(self.W_a(x))   # (B, T, n), real, bounded by gamma

        # FFT to get eigenvalues — conjugate-symmetric since a is real
        a_hat = torch.fft.fft(a.float(), dim=-1)  # (B, T, n) complex

        # Normalize for stability: ensure max|a_hat_i| < 1
        # Scale by 1/n to control spectral radius (since |a_hat_i| <= n*gamma)
        # Then apply tanh-like compression to ensure < 1
        a_hat_mag = a_hat.abs().clamp(min=1e-8)
        # Use soft normalization: a_hat * sigmoid(1 - a_hat_mag) / a_hat_mag
        # This maps large magnitudes close to 0 and preserves small ones
        scale = torch.sigmoid(2.0 * (1.0 - a_hat_mag))  # -> 1 when mag small, -> 0 when mag large
        a_hat = a_hat * scale

        return a_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sequential recurrence (for correctness/training).

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        # Fourier-domain eigenvalues (conjugate-symmetric, stable)
        a_hat = self._get_fourier_eigenvalues(x)  # (B, T, n) complex

        # Input to state space: u_t = W_B x_t (in spatial domain)
        u = self.W_B(x)  # (B, T, n)

        # Transform input to Fourier domain
        u_hat = torch.fft.fft(u.float(), dim=-1)  # (B, T, n) complex

        # --- Parallel scan in Fourier domain ---
        # h_hat_t = a_hat_t * h_hat_{t-1} + u_hat_t
        # This is n independent scalar recurrences
        h_hat = self._parallel_scan(a_hat, u_hat)  # (B, T, n) complex

        # Transform back to spatial domain
        # Since a_hat is conjugate-symmetric and u_hat is conjugate-symmetric (from real u),
        # h_hat should be conjugate-symmetric, so IFFT should be real
        h = torch.fft.ifft(h_hat, dim=-1).real.to(residual.dtype)  # (B, T, n) real

        # Output projection
        y = self.W_C(h)  # (B, T, d_model)

        # Skip connection
        y = y + self.D * residual

        return y

    def _parallel_scan(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """
        Blelloch-style parallel prefix scan for linear recurrence.

        Computes h_t = a_t * h_{t-1} + b_t for each of n independent channels.

        For MVE, we use sequential scan (functionally equivalent, simpler).
        In production, this would use the Blelloch algorithm for O(log T) depth.

        Args:
            a: (B, T, n) complex — per-step multiplicative coefficients
            b: (B, T, n) complex — per-step additive terms
        Returns:
            h: (B, T, n) complex — hidden states at each timestep
        """
        B, T, n = a.shape

        h = torch.zeros(B, n, dtype=a.dtype, device=a.device)
        outputs = []

        for t in range(T):
            h = a[:, t] * h + b[:, t]  # Element-wise: n independent scalar recurrences
            outputs.append(h)

        return torch.stack(outputs, dim=1)  # (B, T, n)

    def forward_with_spatial_check(self, x: torch.Tensor):
        """
        Forward pass that also returns the numerical error between
        spatial-domain and Fourier-domain computation.
        Used for MVE success criterion: ||h_spatial - IFFT(h_scan)||_inf < 1e-4
        """
        B, T, D = x.shape
        x_normed = self.norm(x)

        # Get eigenvalues (conjugate-symmetric)
        a_hat = self._get_fourier_eigenvalues(x_normed)

        # Input
        u = self.W_B(x_normed)
        u_hat = torch.fft.fft(u.float(), dim=-1)

        # Fourier-domain scan
        h_hat = self._parallel_scan(a_hat, u_hat)
        h_from_fourier = torch.fft.ifft(h_hat, dim=-1).real

        # Spatial-domain scan (ground truth)
        # Reconstruct circulant matrix A from eigenvalues and apply in spatial domain
        h_spatial = torch.zeros(B, self.state_dim, dtype=torch.float32, device=x.device)
        h_spatial_list = []

        for t in range(T):
            # A_t h = IFFT(a_hat_t * FFT(h))
            a_hat_t = a_hat[:, t]  # (B, n) complex
            h_freq = torch.fft.fft(h_spatial.to(torch.cfloat), dim=-1)
            Ah = torch.fft.ifft(a_hat_t * h_freq, dim=-1).real

            h_spatial = Ah + u[:, t].float()
            h_spatial_list.append(h_spatial.clone())

        h_spatial_all = torch.stack(h_spatial_list, dim=1)

        # Numerical error
        error = (h_from_fourier - h_spatial_all).abs().max().item()

        # Also check imaginary part magnitude (should be ~0 for real circulant)
        imag_max = torch.fft.ifft(h_hat, dim=-1).imag.abs().max().item()

        return error, imag_max


class CirculantSSMModel(nn.Module):
    """
    Full Circulant SSM model for sequence classification.

    Architecture:
        Embedding -> [CirculantSSMLayer + FFN] x num_layers -> Classification Head

    Args:
        vocab_size: number of tokens
        d_model: model dimension
        state_dim: SSM state dimension
        num_layers: number of SSM layers
        num_classes: number of output classes
        max_seq_len: maximum sequence length
        dropout: dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        state_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 8,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # SSM layers with FFN
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'ssm': CirculantSSMLayer(d_model, state_dim),
                'ffn': nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                ),
            }))

        # Output head
        self.out_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
        Returns:
            logits: (batch, seq_len, num_classes)
        """
        B, T = input_ids.shape

        # Embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_dropout(x)

        # SSM layers
        for layer in self.layers:
            x = x + layer['ssm'](x)     # Residual SSM
            x = x + layer['ffn'](x)     # Residual FFN

        # Classification
        x = self.out_norm(x)
        logits = self.out_head(x)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
