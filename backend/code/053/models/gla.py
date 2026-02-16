"""
Gated Linear Attention (GLA) model for Experiment 053.

Implements a minimal GLA model with:
- Matrix-valued hidden state S_t in R^{d_k x d_v}
- Gated update: S_t = diag(alpha_t) * S_{t-1} + k_t * v_t^T
- Readout: o_t = q_t^T * S_t

References:
- Yang et al. (2024) "Gated Linear Attention Transformers with Hardware-Efficient Training"
- Proposal 053: MLA-Inspired Latent State Compression for Linear RNN Inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict


class GLAHead(nn.Module):
    """
    Single GLA attention head.

    Maintains hidden state S_t in R^{d_k x d_v}.
    Update rule: S_t = diag(alpha_t) * S_{t-1} + k_t * v_t^T
    Readout: o_t = q_t^T * S_t

    Args:
        d_model: input dimension
        d_k: key dimension (state rows)
        d_v: value dimension (state columns)
    """

    def __init__(self, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        # Projections: x_t -> q_t, k_t, v_t, alpha_t
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        # Gate projection: produces d_k values for diagonal gating
        self.W_alpha = nn.Linear(d_model, d_k, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following common practices."""
        for proj in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(proj.weight)
        # Initialize gate bias so alpha starts near 0.9 (retain most of state)
        nn.init.zeros_(self.W_alpha.weight)
        nn.init.constant_(self.W_alpha.bias, 2.0)  # sigmoid(2.0) ≈ 0.88

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with sequential recurrence.

        Args:
            x: input tensor (batch, seq_len, d_model)
            return_states: if True, return list of all states S_t

        Returns:
            output: (batch, seq_len, d_v)
            states: list of S_t tensors if return_states=True, else None
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Project inputs
        q = self.W_q(x)  # (batch, seq_len, d_k)
        k = self.W_k(x)  # (batch, seq_len, d_k)
        v = self.W_v(x)  # (batch, seq_len, d_v)
        alpha = torch.sigmoid(self.W_alpha(x))  # (batch, seq_len, d_k), in (0, 1)

        # Initialize state
        S = torch.zeros(batch, self.d_k, self.d_v, device=device, dtype=dtype)

        outputs = []
        states = [] if return_states else None

        for t in range(seq_len):
            q_t = q[:, t, :]           # (batch, d_k)
            k_t = k[:, t, :]           # (batch, d_k)
            v_t = v[:, t, :]           # (batch, d_v)
            alpha_t = alpha[:, t, :]   # (batch, d_k)

            # State update: S_t = diag(alpha_t) * S_{t-1} + k_t * v_t^T
            # diag(alpha_t) * S: broadcast alpha_t (batch, d_k, 1) * S (batch, d_k, d_v)
            S = alpha_t.unsqueeze(-1) * S + torch.bmm(
                k_t.unsqueeze(-1),  # (batch, d_k, 1)
                v_t.unsqueeze(-2),  # (batch, 1, d_v)
            )  # (batch, d_k, d_v)

            # Readout: o_t = q_t^T * S_t
            # q_t (batch, d_k) -> (batch, 1, d_k) @ S (batch, d_k, d_v) -> (batch, 1, d_v)
            o_t = torch.bmm(q_t.unsqueeze(-2), S).squeeze(-2)  # (batch, d_v)

            outputs.append(o_t)

            if return_states:
                states.append(S.detach().clone())

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_v)
        return output, states

    def recurrent_step(
        self,
        x_t: torch.Tensor,
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Single recurrent step for inference analysis.

        Args:
            x_t: input at time t (batch, d_model)
            S: current state (batch, d_k, d_v)

        Returns:
            o_t: output (batch, d_v)
            S_new: updated state (batch, d_k, d_v)
            intermediates: dict with q_t, k_t, v_t, alpha_t for analysis
        """
        q_t = self.W_q(x_t)            # (batch, d_k)
        k_t = self.W_k(x_t)            # (batch, d_k)
        v_t = self.W_v(x_t)            # (batch, d_v)
        alpha_t = torch.sigmoid(self.W_alpha(x_t))  # (batch, d_k)

        # State update
        S_new = alpha_t.unsqueeze(-1) * S + torch.bmm(
            k_t.unsqueeze(-1), v_t.unsqueeze(-2)
        )

        # Readout
        o_t = torch.bmm(q_t.unsqueeze(-2), S_new).squeeze(-2)

        intermediates = {
            'q_t': q_t,
            'k_t': k_t,
            'v_t': v_t,
            'alpha_t': alpha_t,
        }

        return o_t, S_new, intermediates


class GLALayer(nn.Module):
    """
    GLA layer with multi-head attention + FFN.

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        d_k: key dimension per head
        d_v: value dimension per head
        dropout: dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_v = d_v

        # Multi-head GLA
        self.heads = nn.ModuleList([
            GLAHead(d_model, d_k, d_v) for _ in range(n_heads)
        ])

        # Output projection: concatenated head outputs -> d_model
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

        # Layer norm + FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[List[torch.Tensor]]]]:
        """
        Forward pass.

        Args:
            x: input (batch, seq_len, d_model)
            return_states: if True, return states from all heads

        Returns:
            output: (batch, seq_len, d_model)
            all_states: list of head states if return_states=True
        """
        residual = x
        x = self.norm1(x)

        # Multi-head GLA
        head_outputs = []
        all_states = [] if return_states else None

        for head in self.heads:
            head_out, head_states = head(x, return_states=return_states)
            head_outputs.append(head_out)
            if return_states:
                all_states.append(head_states)

        # Concatenate heads and project
        concat = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, n_heads * d_v)
        attn_out = self.W_o(concat)
        x = residual + self.dropout(attn_out)

        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x, all_states


class GLAModel(nn.Module):
    """
    GLA Language Model.

    Args:
        vocab_size: vocabulary size
        d_model: model dimension
        n_layers: number of GLA layers
        n_heads: number of attention heads per layer
        d_k: key dimension per head
        d_v: value dimension per head
        dropout: dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 2,
        d_k: int = 64,
        d_v: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)  # Positional embeddings up to 1024
        self.layers = nn.ModuleList([
            GLALayer(d_model, n_heads, d_k, d_v, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass.

        Args:
            input_ids: token ids (batch, seq_len)
            return_states: if True, return hidden states from all layers/heads

        Returns:
            logits: (batch, seq_len, vocab_size)
            all_layer_states: nested list [layer][head][timestep] of state tensors
        """
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embedding(input_ids) + self.pos_embedding(positions)

        all_layer_states = [] if return_states else None

        for layer in self.layers:
            x, layer_states = layer(x, return_states=return_states)
            if return_states:
                all_layer_states.append(layer_states)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, all_layer_states

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
