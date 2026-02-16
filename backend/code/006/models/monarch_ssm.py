"""
Monarch-Gated State Transition SSM

From proposal 006-monarch-gated-state-transition.md:

Standard Diagonal SSM:
    h_t = diag(alpha_t) * h_{t-1} + B_t * x_t
    y_t = C_t^T * h_t

Monarch-Gated SSM (Proposed):
    h_t = M(x_t) * h_{t-1} + B_t * x_t
    y_t = C_t^T * h_t

where the Monarch transition is:
    M(x_t) = P_b^T * L(x_t) * P_b * R(x_t)

with input-dependent block-diagonal factors:
    L(x_t) = diag(alpha_1(x_t) * L_1, ..., alpha_b(x_t) * L_b)
    R(x_t) = diag(beta_1(x_t) * R_1, ..., beta_b(x_t) * R_b)

where:
    - L_i, R_i are fixed orthogonal blocks (Cayley-parameterized)
    - alpha_i, beta_i are input-dependent scalar gates in (0, 1)
    - P_b is the fixed stride permutation (reshape transpose)
    - b = sqrt(n) is both the block count and block size

Stability: ||M(x_t)|| <= max_i(alpha_i) * max_j(beta_j) < 1 (contractive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def cayley_transform_batched(skew_params: torch.Tensor) -> torch.Tensor:
    """Compute Cayley transform for a batch of skew-symmetric matrices.

    Q = (I - A)(I + A)^{-1} where A = skew - skew^T is skew-symmetric.

    Args:
        skew_params: (num_blocks, b, b) raw parameter matrices

    Returns:
        Q: (num_blocks, b, b) orthogonal matrices
    """
    # Make skew-symmetric: A = skew - skew^T
    A = skew_params - skew_params.transpose(-2, -1)
    b = A.shape[-1]
    I = torch.eye(b, device=A.device, dtype=A.dtype).unsqueeze(0)  # (1, b, b)
    # Cayley: Q = (I - A)(I + A)^{-1}
    Q = torch.linalg.solve(I + A, I - A)
    return Q


class MonarchTransition(nn.Module):
    """Input-dependent Monarch-factored state transition.

    M(x) = P_b^T @ L(x) @ P_b @ R(x)

    where L(x) and R(x) are block-diagonal with input-gated orthogonal blocks.

    Optimized: orthogonal blocks are computed once per forward call using
    batched Cayley transform, then applied efficiently via reshape + BMM.

    Args:
        state_dim: n, the state dimension (must be a perfect square)
        input_dim: d, the input dimension for gating
    """

    def __init__(self, state_dim: int, input_dim: int):
        super().__init__()
        self.n = state_dim
        self.b = int(math.isqrt(state_dim))
        assert self.b * self.b == state_dim, f"state_dim must be perfect square, got {state_dim}"

        # Batched skew-symmetric parameters for Cayley transform
        # L_blocks: (b, b, b) — b orthogonal blocks of size b x b
        self.L_skew = nn.Parameter(torch.randn(self.b, self.b, self.b) * 0.01)
        # R_blocks: (b, b, b) — b orthogonal blocks of size b x b
        self.R_skew = nn.Parameter(torch.randn(self.b, self.b, self.b) * 0.01)

        # Input-dependent gating: produces 2b scalar gates (alpha_1..alpha_b, beta_1..beta_b)
        # Eq: [alpha_1, ..., alpha_b, beta_1, ..., beta_b] = sigma(W_g * x + b_g)
        self.gate_proj = nn.Linear(input_dim, 2 * self.b)

        # Pre-compute the stride permutation indices (fixed, not learned)
        # P_b reshapes (b, b) -> (b, b) via transpose: index i*b+j -> j*b+i
        perm_indices = torch.zeros(state_dim, dtype=torch.long)
        for i in range(self.b):
            for j in range(self.b):
                perm_indices[i * self.b + j] = j * self.b + i
        self.register_buffer('perm_indices', perm_indices)
        # Inverse permutation (P_b^T)
        inv_perm_indices = torch.zeros(state_dim, dtype=torch.long)
        inv_perm_indices[perm_indices] = torch.arange(state_dim)
        self.register_buffer('inv_perm_indices', inv_perm_indices)

    def get_orthogonal_blocks(self):
        """Compute all orthogonal blocks at once using batched Cayley transform.

        Returns:
            L_Q: (b, b, b) — b orthogonal L blocks
            R_Q: (b, b, b) — b orthogonal R blocks
        """
        L_Q = cayley_transform_batched(self.L_skew)  # (b, b, b)
        R_Q = cayley_transform_batched(self.R_skew)  # (b, b, b)
        return L_Q, R_Q

    def forward(self, x: torch.Tensor, h: torch.Tensor, L_Q: torch.Tensor, R_Q: torch.Tensor) -> torch.Tensor:
        """Apply Monarch transition: h_new = M(x) @ h

        Uses pre-computed orthogonal blocks for efficiency.

        Args:
            x: (batch, input_dim) input at current timestep
            h: (batch, n) hidden state
            L_Q: (b, b, b) pre-computed L orthogonal blocks
            R_Q: (b, b, b) pre-computed R orthogonal blocks

        Returns:
            h_new: (batch, n) updated hidden state after Monarch transition
        """
        batch_size = x.shape[0]

        # Compute input-dependent gates
        gates = torch.sigmoid(self.gate_proj(x))  # (batch, 2*b)
        alpha_gates = gates[:, :self.b]   # (batch, b) — gates for L blocks
        beta_gates = gates[:, self.b:]    # (batch, b) — gates for R blocks

        # Step 1: Apply R(x) — gated block-diagonal right factor
        # Reshape h into blocks: (batch, b, b)
        h_blocks = h.view(batch_size, self.b, self.b)
        # BMM: R_Q @ h_blocks^T -> each block_i: R_Q[i] @ h_blocks[:, i, :].T
        # h_blocks: (batch, b_count, b_size) — we need (batch, b_count, b_size)
        # R_Q: (b_count, b_size, b_size) — for each block i, multiply h[:, i, :] by R_Q[i]
        # Use einsum: 'ibj, Bbj -> Bbi' -> result[B, b_count, b_size]
        h_r = torch.einsum('cbj,Bcj->Bcb', R_Q, h_blocks)  # (batch, b, b)
        # Apply beta gates: (batch, b, 1) * (batch, b, b)
        h_r = beta_gates.unsqueeze(-1) * h_r  # (batch, b, b)

        # Step 2: Apply P_b — stride permutation (flatten, permute, reshape)
        h_flat = h_r.reshape(batch_size, self.n)  # (batch, n)
        h_p = h_flat[:, self.perm_indices]  # (batch, n)
        h_p_blocks = h_p.view(batch_size, self.b, self.b)  # (batch, b, b)

        # Step 3: Apply L(x) — gated block-diagonal left factor
        h_l = torch.einsum('cbj,Bcj->Bcb', L_Q, h_p_blocks)  # (batch, b, b)
        h_l = alpha_gates.unsqueeze(-1) * h_l  # (batch, b, b)

        # Step 4: Apply P_b^T — inverse stride permutation
        h_flat = h_l.reshape(batch_size, self.n)
        h_out = h_flat[:, self.inv_perm_indices]

        return h_out


class MonarchGatedSSMLayer(nn.Module):
    """Single layer of Monarch-Gated SSM.

    h_t = M(x_t) * h_{t-1} + B_t * x_t
    y_t = C_t^T * h_t

    Optimized: orthogonal blocks are computed ONCE per forward call (not per timestep).

    Args:
        d_model: Input/output dimension
        state_dim: Hidden state dimension (must be perfect square)
    """

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Monarch transition
        self.monarch = MonarchTransition(state_dim, d_model)

        # Input projection B: maps input to state space
        self.B = nn.Linear(d_model, state_dim)
        # Output projection C: maps state to output
        self.C = nn.Linear(state_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a sequence through the Monarch-Gated SSM.

        Args:
            x: (batch, seq_len, d_model) input sequence

        Returns:
            y: (batch, seq_len, d_model) output sequence
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Pre-compute orthogonal blocks ONCE for the entire sequence
        L_Q, R_Q = self.monarch.get_orthogonal_blocks()

        h = torch.zeros(batch_size, self.state_dim, device=device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)

            # State update: h_t = M(x_t) * h_{t-1} + B * x_t
            h = self.monarch(x_t, h, L_Q, R_Q) + self.B(x_t)

            # Output: y_t = C * h_t
            y_t = self.C(h)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)


class DiagonalSSMLayer(nn.Module):
    """Diagonal SSM baseline (like Mamba-2 simplified).

    h_t = diag(alpha_t) * h_{t-1} + B * x_t
    y_t = C^T * h_t

    where alpha_t = sigmoid(W_alpha * x_t) are input-dependent decay gates.
    This is the standard diagonal SSM that cannot do coordinate mixing.
    """

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Input-dependent diagonal gates
        self.gate_proj = nn.Linear(d_model, state_dim)

        # Input projection B
        self.B = nn.Linear(d_model, state_dim)
        # Output projection C
        self.C = nn.Linear(state_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a sequence through diagonal SSM.

        Args:
            x: (batch, seq_len, d_model) input

        Returns:
            y: (batch, seq_len, d_model) output
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = torch.zeros(batch_size, self.state_dim, device=device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            # Input-dependent diagonal gate
            alpha = torch.sigmoid(self.gate_proj(x_t))  # (batch, state_dim)

            # State update: h_t = diag(alpha_t) * h_{t-1} + B * x_t
            h = alpha * h + self.B(x_t)

            # Output: y_t = C * h_t
            y_t = self.C(h)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class SwiGLU(nn.Module):
    """SwiGLU feedforward block."""

    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SSMBlock(nn.Module):
    """Residual block: SSM layer + FFN with pre-norm."""

    def __init__(self, d_model: int, state_dim: int, ssm_type: str = "monarch"):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        if ssm_type == "monarch":
            self.ssm = MonarchGatedSSMLayer(d_model, state_dim)
        elif ssm_type == "diagonal":
            self.ssm = DiagonalSSMLayer(d_model, state_dim)
        else:
            raise ValueError(f"Unknown SSM type: {ssm_type}")

        self.ffn = SwiGLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class S5SSMModel(nn.Module):
    """Full SSM model for S5 permutation composition.

    Architecture:
        Token embedding -> Positional embedding -> SSM blocks -> Classifier

    The model processes the input sequence and classifies the final non-padding
    position into one of 120 S5 permutations.

    Args:
        num_tokens: Number of input tokens (generators + special tokens)
        num_classes: Number of output classes (120 for S5)
        d_model: Hidden dimension
        state_dim: SSM state dimension (must be perfect square for Monarch)
        num_layers: Number of SSM blocks
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        ssm_type: "monarch" or "diagonal"
        eos_idx: Index of EOS token (used to extract final representation)
    """

    def __init__(
        self,
        num_tokens: int,
        num_classes: int,
        d_model: int,
        state_dim: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.1,
        ssm_type: str = "monarch",
        eos_idx: int = 131,
    ):
        super().__init__()
        self.eos_idx = eos_idx
        self.d_model = d_model

        self.token_emb = nn.Embedding(num_tokens, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SSMBlock(d_model, state_dim, ssm_type=ssm_type)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len) input token indices

        Returns:
            logits: (batch, num_classes) classification logits
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.dropout(h)

        # SSM blocks
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)

        # Extract representation at EOS position for each sample
        # Find EOS position in each sequence
        eos_mask = (x == self.eos_idx)
        # Get the index of the first EOS in each sequence
        eos_positions = eos_mask.float().argmax(dim=1)  # (batch,)

        # Gather the representation at EOS position
        # h: (batch, seq_len, d_model)
        eos_repr = h[torch.arange(batch_size, device=device), eos_positions]  # (batch, d_model)

        # Classify
        logits = self.classifier(eos_repr)  # (batch, num_classes)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
