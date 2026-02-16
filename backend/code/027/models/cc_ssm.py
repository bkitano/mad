"""
Cayley-Circulant Orthogonal SSM (CC-SSM)
From proposal 027-cayley-circulant-orthogonal-ssm.

Key idea: Parameterize state transition as Cayley transform of skew-circulant matrix.
  W = (I + A)^{-1}(I - A)  where A is skew-circulant
In Fourier domain:
  lambda_j = (1 - i*omega_j) / (1 + i*omega_j) = e^{-2i*arctan(omega_j)}
  |lambda_j| = 1 exactly (orthogonal by construction)

For efficient training, we decompose the SSM into n independent 1D channels:
  1. Project input: u = B @ x  (d_model -> state_dim)
  2. Apply state transition per channel via 1D convolution with scalar eigenvalue
  3. Project output: y = C @ h  (state_dim -> d_model)

The key difference from diagonal SSMs: the B and C projections interact with
the circulant structure of W. However, for the LTI case, we can still use
convolution mode by computing the per-channel kernels.

Actually, the crucial insight for CC-SSM:
  The transition W is circulant, so it mixes all n state dimensions at each step.
  This is NOT decomposable into n independent scalar channels in the state basis.
  But in the FOURIER basis, W = F^{-1} diag(lambda) F, so the dynamics ARE diagonal.

  We can work entirely in the Fourier domain:
  h_hat_t = lambda * h_hat_{t-1} + B_hat * u_t
  where h_hat = FFT(h), B_hat = FFT(B), etc.

  Then the per-channel 1D convolution kernel is:
  K_hat_t[j] = C_hat[j] * lambda_j^t * B_hat[j]

This gives us O(n * T * log T) training cost via FFT convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CayleyCirculantSSMLayer(nn.Module):
    """
    CC-SSM layer operating in the Fourier domain.

    In the Fourier domain, the circulant transition W is diag(lambda), so the
    SSM decomposes into n independent complex-valued 1D recurrences.

    Per-channel kernel: k_j[t] = lambda_j^t  (scalar geometric series)
    Full kernel: K[t] = C_hat * k[t] * B_hat  (element-wise scaling)
    Output: y = IFFT(K * FFT(u))  (1D causal convolution per channel)
    """
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.half_n = state_dim // 2

        # Free parameters for skew-circulant generator
        self.a_params = nn.Parameter(torch.randn(self.half_n) * 0.1)

        # Input/output projections
        # B: (d_model,) -> (state_dim,) -- per-channel input mixing
        # We use (state_dim, d_model) matrices and compute B @ x
        self.B = nn.Linear(d_model, state_dim, bias=False)
        self.C = nn.Linear(state_dim, d_model, bias=False)

        # Skip connection
        self.D = nn.Parameter(torch.ones(d_model))

    def _build_cayley_eigenvalues(self):
        """Build Cayley-circulant eigenvalues. Returns complex (n,), |lambda| = 1."""
        n = self.state_dim
        a = self.a_params

        a_full = torch.zeros(n, device=a.device, dtype=a.dtype)
        a_full[1:self.half_n + 1] = a
        if self.half_n > 0:
            a_full[self.half_n + 1:] = -a.flip(0)[:n - self.half_n - 1]

        a_hat = torch.fft.fft(a_full)
        omega = a_hat.imag

        numerator = torch.complex(torch.ones_like(omega), -omega)
        denominator = torch.complex(torch.ones_like(omega), omega)
        eigenvalues = numerator / denominator
        return eigenvalues

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Convolution mode in the Fourier domain.

        1. Project input: Bu = B @ u  -> (batch, T, n)
        2. Build per-channel kernels: k_j[t] = lambda_j^t
        3. Apply causal convolution per channel via FFT
        4. Project output: y = C @ h  -> (batch, T, d)
        5. Add skip: y += D * u

        Args:
            u: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, d = u.shape

        # Step 1: Project input to state space
        Bu = self.B(u)  # (batch, seq_len, state_dim)

        # Step 2: Build per-channel kernel from eigenvalues
        eigenvalues = self._build_cayley_eigenvalues()  # (n,) complex, |lambda| = 1

        # Kernel: k[t, j] = lambda_j^t for t = 0..seq_len-1
        t = torch.arange(seq_len, device=eigenvalues.device, dtype=torch.float32)
        log_lambda = torch.log(eigenvalues)  # purely imaginary since |lambda| = 1
        kernel = torch.exp(t.unsqueeze(1) * log_lambda.unsqueeze(0))  # (seq_len, n) complex

        # Step 3: Causal convolution per channel via FFT
        # Pad for causal convolution (avoid circular convolution artifacts)
        fft_len = 1
        while fft_len < 2 * seq_len:
            fft_len *= 2

        # FFT of kernel: (fft_len, n)
        kernel_padded = F.pad(kernel.real, (0, 0, 0, fft_len - seq_len))
        kernel_padded_imag = F.pad(kernel.imag, (0, 0, 0, fft_len - seq_len))
        kernel_complex = torch.complex(kernel_padded, kernel_padded_imag)
        K_hat = torch.fft.fft(kernel_complex, dim=0)  # (fft_len, n)

        # FFT of input Bu: (batch, fft_len, n)
        Bu_padded = F.pad(Bu, (0, 0, 0, fft_len - seq_len))
        Bu_hat = torch.fft.fft(Bu_padded.to(torch.complex64), dim=1)  # (batch, fft_len, n)

        # Multiply in frequency domain
        h_hat = K_hat.unsqueeze(0) * Bu_hat  # (batch, fft_len, n)

        # IFFT back
        h = torch.fft.ifft(h_hat, dim=1).real[:, :seq_len, :]  # (batch, seq_len, n)

        # Step 4: Project output
        y = self.C(h)  # (batch, seq_len, d_model)

        # Step 5: Skip connection
        y = y + self.D * u

        return y

    def get_eigenvalue_magnitudes(self):
        """Return |lambda_j|. Should all be exactly 1.0."""
        with torch.no_grad():
            eigenvalues = self._build_cayley_eigenvalues()
            return eigenvalues.abs()


class CCSSM(nn.Module):
    """
    Full CC-SSM model for sequence-to-sequence tasks.
    Architecture: Embedding -> L x (CC-SSM + SwiGLU) with pre-norm residuals -> Output head
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        state_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(CCSSMBlock(d_model, state_dim, dropout))

        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        return self.head(h)

    def get_eigenvalue_magnitudes(self):
        return [block.ssm.get_eigenvalue_magnitudes() for block in self.blocks]


class CCSSMBlock(nn.Module):
    def __init__(self, d_model: int, state_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = CayleyCirculantSSMLayer(d_model, state_dim)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.ssm(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        d_ff = ((int(8/3 * d_model) + 7) // 8) * 8
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))
