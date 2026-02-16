"""
Log-Semiring SSM: Softmax-Native Parallel Scan via Logarithmic Semiring Recurrence.

From proposal 014-log-semiring-ssm-scan.

Key idea: Replace the standard (R, +, ×) semiring in SSM scans with the
logarithmic semiring (R, logsumexp, +). The recurrence:

    ℓ_t = logsumexp(a_t + ℓ_{t-1}, b_t)

where:
    - a_t < 0: input-dependent log-decay (controls forgetting)
    - b_t = q_t · k_t / √d: attention logit

The scan operator is:
    (ℓ₁, a₁) • (ℓ₂, a₂) = (logsumexp(a₂ + ℓ₁, ℓ₂), a₁ + a₂)

This is associative (proven in proposal), enabling parallel prefix scan.

The hidden state ℓ_t is the log-partition function of a softmax distribution
over input history — achieving exact softmax attention without approximation.

For the readout, we track log-numerator states for each value dimension:
    n_{t,i,m} = logsumexp(a_{t,i} + n_{t-1,i,m}, b_{t,i} + log|v_{t,m}|)

Output:
    y_{t,m} = Σ_i exp(n_{t,i,m} - ℓ_{t,i}) · sign_{t,i,m}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def stable_logsumexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Numerically stable logsumexp of two tensors: log(exp(a) + exp(b)).

    Uses the identity: logsumexp(a, b) = max(a, b) + log1p(exp(-|a - b|))
    This avoids overflow/underflow.
    """
    max_val = torch.maximum(a, b)
    return max_val + torch.log1p(torch.exp(-torch.abs(a - b)))


def log_semiring_scan_sequential(
    log_decays: torch.Tensor,  # (batch, seq_len, d)  — a_t, negative
    log_inputs: torch.Tensor,  # (batch, seq_len, d)  — b_t
) -> torch.Tensor:
    """Sequential log-semiring scan for the log-normalizer.

    Recurrence: ℓ_t = logsumexp(a_t + ℓ_{t-1}, b_t)

    Args:
        log_decays: a_t values, should be negative (forgetting)
        log_inputs: b_t values (attention logits)

    Returns:
        log_states: (batch, seq_len, d) — ℓ_t at each timestep
    """
    batch, seq_len, d = log_decays.shape
    device = log_decays.device

    # Initialize: ℓ_0 = b_0 (first input, no history)
    log_state = log_inputs[:, 0, :]  # (batch, d)

    states = [log_state]

    for t in range(1, seq_len):
        a_t = log_decays[:, t, :]   # (batch, d)
        b_t = log_inputs[:, t, :]   # (batch, d)

        # ℓ_t = logsumexp(a_t + ℓ_{t-1}, b_t)
        log_state = stable_logsumexp(a_t + log_state, b_t)
        states.append(log_state)

    return torch.stack(states, dim=1)  # (batch, seq_len, d)


def log_semiring_scan_parallel(
    log_decays: torch.Tensor,  # (batch, seq_len, d)
    log_inputs: torch.Tensor,  # (batch, seq_len, d)
) -> torch.Tensor:
    """Parallel prefix scan using log-semiring operator.

    Scan operator: (ℓ₁, a₁) • (ℓ₂, a₂) = (logsumexp(a₂ + ℓ₁, ℓ₂), a₁ + a₂)

    Uses Blelloch-style up-sweep / down-sweep for O(log T) parallel depth.
    For MVE, we use sequential scan (functionally equivalent, simpler).
    """
    # For the MVE, use sequential scan. Parallel scan would give same results
    # but with O(log T) depth instead of O(T). Both are correct.
    return log_semiring_scan_sequential(log_decays, log_inputs)


def log_semiring_scan_with_values(
    log_decays: torch.Tensor,   # (batch, seq_len, d_head)
    log_inputs: torch.Tensor,   # (batch, seq_len, d_head)
    values: torch.Tensor,       # (batch, seq_len, d_val)
) -> torch.Tensor:
    """Full log-semiring scan with value readout.

    Tracks both the log-normalizer ℓ_t and log-numerator n_t states.

    For each value dimension m and head dimension i:
        n_{t,i,m} = logsumexp(a_{t,i} + n_{t-1,i,m}, b_{t,i} + log|v_{t,m}|)

    Output:
        y_{t,m} = Σ_i exp(n_{t,i,m} - ℓ_{t,i}) · sign_{t,i,m}

    Args:
        log_decays: (batch, seq_len, d_head) — a_t < 0
        log_inputs: (batch, seq_len, d_head) — b_t = q·k/√d
        values: (batch, seq_len, d_val) — value vectors

    Returns:
        outputs: (batch, seq_len, d_val) — softmax-weighted output
    """
    batch, seq_len, d_head = log_decays.shape
    d_val = values.shape[-1]
    device = log_decays.device

    # Separate values into sign and log-magnitude
    # v_t can be negative, so we track signs separately
    v_sign = torch.sign(values)  # (batch, seq_len, d_val)
    v_sign = torch.where(v_sign == 0, torch.ones_like(v_sign), v_sign)
    v_log_abs = torch.log(torch.abs(values) + 1e-8)  # (batch, seq_len, d_val)

    # Initialize states
    # ℓ_0 = b_0
    log_norm = log_inputs[:, 0, :]  # (batch, d_head)

    # n_{0,i,m} = b_{0,i} + log|v_{0,m}|
    # Shape: (batch, d_head, d_val)
    log_num = log_inputs[:, 0, :].unsqueeze(-1) + v_log_abs[:, 0, :].unsqueeze(-2)
    # sign_{0,i,m} = sign(v_{0,m}) — same for all heads initially
    num_sign = v_sign[:, 0, :].unsqueeze(-2).expand(-1, d_head, -1).clone()

    outputs = []

    for t in range(seq_len):
        if t > 0:
            a_t = log_decays[:, t, :]   # (batch, d_head)
            b_t = log_inputs[:, t, :]    # (batch, d_head)
            v_log_t = v_log_abs[:, t, :] # (batch, d_val)
            v_sign_t = v_sign[:, t, :]   # (batch, d_val)

            # Update log-normalizer: ℓ_t = logsumexp(a_t + ℓ_{t-1}, b_t)
            decayed = a_t + log_norm  # (batch, d_head)
            log_norm = stable_logsumexp(decayed, b_t)

            # Update log-numerator for each (head_dim, val_dim) pair
            # n_{t,i,m} = logsumexp(a_{t,i} + n_{t-1,i,m}, b_{t,i} + log|v_{t,m}|)
            decayed_num = a_t.unsqueeze(-1) + log_num  # (batch, d_head, d_val)
            new_input = b_t.unsqueeze(-1) + v_log_t.unsqueeze(-2)  # (batch, d_head, d_val)

            # For signed logsumexp, we need to handle the case where
            # the two terms have different signs.
            # signed_logsumexp: log|s₁e^a + s₂e^b| and sign
            s1 = num_sign  # sign of exp(decayed_num)
            s2 = v_sign_t.unsqueeze(-2).expand_as(s1)  # sign of exp(new_input)

            max_val = torch.maximum(decayed_num, new_input)
            exp_diff_1 = s1 * torch.exp(decayed_num - max_val)
            exp_diff_2 = s2 * torch.exp(new_input - max_val)
            sum_val = exp_diff_1 + exp_diff_2

            new_sign = torch.sign(sum_val)
            new_sign = torch.where(new_sign == 0, torch.ones_like(new_sign), new_sign)
            log_num = max_val + torch.log(torch.abs(sum_val) + 1e-8)
            num_sign = new_sign

        # Readout: y_{t,m} = Σ_i exp(n_{t,i,m} - ℓ_{t,i}) · sign_{t,i,m}
        # (batch, d_head, d_val) - (batch, d_head, 1) → (batch, d_head, d_val)
        weights = torch.exp(log_num - log_norm.unsqueeze(-1))  # softmax weights
        weighted = num_sign * weights  # (batch, d_head, d_val)
        y_t = weighted.sum(dim=-2)  # sum over head dims → (batch, d_val)
        outputs.append(y_t)

    return torch.stack(outputs, dim=1)  # (batch, seq_len, d_val)


class LogSSMLayer(nn.Module):
    """Single Log-Semiring SSM layer.

    Architecture (proposal §Full Architecture):
        1. Project input → q, k, v, α
        2. Compute b_t = q_t · k_t / √d (attention logits)
        3. Compute a_t = -softplus(α_t) (negative log-decay)
        4. Run log-semiring scan to get outputs
        5. Output projection
    """

    def __init__(self, d_model: int, d_head: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads

        # Projections for q, k (per-head)
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)

        # Value projection (full d_model, shared across heads)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Decay projection: input-dependent forgetting
        # a_t = -softplus(W_α x_t + c_α) < 0
        self.W_alpha = nn.Linear(d_model, n_heads * d_head)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d_head)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.W_q.weight, gain=0.1)
        nn.init.xavier_normal_(self.W_k.weight, gain=0.1)
        nn.init.xavier_normal_(self.W_v.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_o.weight, gain=0.1)
        # Bias for decay: initialize so softplus(bias) ≈ 1, giving moderate decay
        nn.init.constant_(self.W_alpha.bias, 1.0)
        nn.init.normal_(self.W_alpha.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project to q, k, v, alpha
        q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head)
        k = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_head)
        v = self.W_v(x)  # (batch, seq_len, d_model)

        # Attention logits: b_t = q_t · k_t / √d (element-wise per head dim)
        # Proposal: b_{t,i} = q_{t,i} · k_{t,i} / √d
        b = (q * k * self.scale).sum(dim=-1)  # (batch, seq_len, n_heads) — sum within each head
        # Actually, per the proposal, b_{t,i} is per head dimension, not summed.
        # But for a scalar-state-per-head interpretation, we sum to get one logit per head.
        # This gives n_heads scalar log-states.

        # Log-decay: a_t = -softplus(W_α x_t + c_α) < 0
        alpha = self.W_alpha(x).view(batch, seq_len, self.n_heads, self.d_head)
        a = -F.softplus(alpha).sum(dim=-1)  # (batch, seq_len, n_heads) — sum for scalar per head

        # Run log-semiring scan with values
        # log_decays: (batch, seq_len, n_heads)
        # log_inputs: (batch, seq_len, n_heads)
        # values: (batch, seq_len, d_model)
        y = log_semiring_scan_with_values(a, b, v)  # (batch, seq_len, d_model)

        y = self.W_o(y)
        y = self.dropout(y)

        return y


class LogSSMBlock(nn.Module):
    """LogSSM layer + LayerNorm + Residual + FFN."""

    def __init__(self, d_model: int, d_head: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = LogSSMLayer(d_model, d_head, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LogSSMClassifier(nn.Module):
    """Log-Semiring SSM for sequence classification.

    MVE config (proposal §Minimum Viable Experiment):
        - 2 layers, D=64, d=16, H=4, ~80K params
        - Task: selective copying
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 64,
        d_head: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            LogSSMBlock(d_model, d_head, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.xavier_normal_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len) integer token ids
        Returns:
            logits: (batch, seq_len, num_classes)
        """
        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)

        x = self.embedding(tokens) + self.pos_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.head(x)  # (batch, seq_len, num_classes)
        return logits
