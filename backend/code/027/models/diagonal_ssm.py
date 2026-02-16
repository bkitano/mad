"""
Diagonal SSM baseline (S4D-style) using convolution mode.

State transition: h_t = diag(lambda) @ h_{t-1} + B @ x_t
where lambda_j = sigmoid(log_lambda_j) gives |lambda_j| in (0, 1).

Uses per-channel 1D convolution for efficient parallel training:
  k_j[t] = lambda_j^t  (exponentially decaying kernel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalSSMLayer(nn.Module):
    """
    Diagonal SSM layer using per-channel convolution mode.

    Per-channel kernel: k_j[t] = lambda_j^t
    Causal convolution via FFT.
    """
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # sigmoid(log_lambda) in (0,1), initialized near 0.95-0.99
        self.log_lambda = nn.Parameter(torch.linspace(2.94, 4.60, state_dim))

        self.B = nn.Linear(d_model, state_dim, bias=False)
        self.C = nn.Linear(state_dim, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d = u.shape

        # Project to state space
        Bu = self.B(u)  # (batch, seq_len, n)

        # Build per-channel kernel
        lam = torch.sigmoid(self.log_lambda)  # (n,)
        t = torch.arange(seq_len, device=lam.device, dtype=lam.dtype)
        kernel = lam.unsqueeze(0).pow(t.unsqueeze(1))  # (seq_len, n)

        # Causal convolution via FFT
        fft_len = 1
        while fft_len < 2 * seq_len:
            fft_len *= 2

        K_hat = torch.fft.rfft(kernel, n=fft_len, dim=0)  # (fft_len//2+1, n)
        Bu_hat = torch.fft.rfft(Bu, n=fft_len, dim=1)  # (batch, fft_len//2+1, n)

        h_hat = K_hat.unsqueeze(0) * Bu_hat
        h = torch.fft.irfft(h_hat, n=fft_len, dim=1)[:, :seq_len, :]  # (batch, seq_len, n)

        y = self.C(h) + self.D * u
        return y

    def get_eigenvalue_magnitudes(self):
        with torch.no_grad():
            return torch.sigmoid(self.log_lambda)


class DiagonalSSM(nn.Module):
    def __init__(self, vocab_size, d_model=64, state_dim=32, num_layers=2, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([DiagSSMBlock(d_model, state_dim, dropout) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        return self.head(h)

    def get_eigenvalue_magnitudes(self):
        return [block.ssm.get_eigenvalue_magnitudes() for block in self.blocks]


class DiagSSMBlock(nn.Module):
    def __init__(self, d_model, state_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = DiagonalSSMLayer(d_model, state_dim)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.ssm(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class SwiGLU(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        d_ff = ((int(8/3 * d_model) + 7) // 8) * 8
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))
