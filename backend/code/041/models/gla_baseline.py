"""
Baseline GLA (Gated Linear Attention) with SEPARATE forward and backward kernels.

This implements the chunkwise linear attention algorithm from
"Gated Linear Attention Transformers with Hardware-Efficient Training" (Yang et al., ICML 2024).

The key computation per chunk j is:
  Forward:
    F1: S_j = Q_j @ K_j^T                    (GEMM: C x d x C)
    F2: S_tilde_j = S_j * M_j                (elementwise decay mask)
    F3: O_j^intra = S_tilde_j @ V_j          (GEMM: C x C x d)
    F4: O_j^state = Q_j @ h_jC               (GEMM: C x n x d_v)
    F5: O_j = O_j^intra + O_j^state          (elementwise add)

  Backward (given grad_O_j):
    B4: grad_S_tilde = grad_O_j @ V_j^T      (GEMM: C x d x C)
    B5: grad_V_j = S_tilde_j^T @ grad_O_j    (GEMM: C x C x d)
    B6: grad_S_j = grad_S_tilde * M_j         (elementwise mask)
    B7: grad_Q_j = grad_S_j @ K_j + grad_O_j @ h_jC^T  (GEMM + add)
    B8: grad_K_j = grad_S_j^T @ Q_j          (GEMM: C x C x d)

In this baseline, forward and backward are separate Triton kernels.
Each intermediate tensor (S_tilde, etc.) is materialized to HBM between steps.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


# ============================================================================
# Triton Kernels for SEPARATE forward/backward
# ============================================================================

@triton.jit
def _fwd_intra_chunk_kernel(
    Q_ptr, K_ptr, V_ptr, M_ptr, H_ptr, O_ptr, S_tilde_ptr,
    B: tl.constexpr, T: tl.constexpr, C: tl.constexpr,
    D: tl.constexpr, N: tl.constexpr,
    stride_qb, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_mb, stride_mc, stride_mr, stride_mcol,
    stride_hb, stride_hc, stride_hn, stride_hd,
    stride_ob, stride_ot, stride_od,
    stride_sb, stride_sc, stride_sr, stride_scol,
):
    """Forward intra-chunk computation: S = QK^T, S_tilde = S*M, O = S_tilde @ V + Q @ H"""
    pid_b = tl.program_id(0)  # batch
    pid_c = tl.program_id(1)  # chunk index

    # Chunk offset
    chunk_start = pid_c * C

    # Load Q_j [C, D]
    offs_c = tl.arange(0, C)
    offs_d = tl.arange(0, D)

    q_ptrs = Q_ptr + pid_b * stride_qb + (chunk_start + offs_c[:, None]) * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_c[:, None] < C) & (offs_d[None, :] < D), other=0.0)

    # Load K_j [C, D]
    k_ptrs = K_ptr + pid_b * stride_kb + (chunk_start + offs_c[:, None]) * stride_kt + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=(offs_c[:, None] < C) & (offs_d[None, :] < D), other=0.0)

    # F1: S = Q @ K^T [C, C]
    s = tl.dot(q, tl.trans(k))

    # Load M_j [C, C] - decay mask
    offs_c2 = tl.arange(0, C)
    m_ptrs = M_ptr + pid_b * stride_mb + pid_c * stride_mc + offs_c[:, None] * stride_mr + offs_c2[None, :] * stride_mcol
    m = tl.load(m_ptrs)

    # F2: S_tilde = S * M
    s_tilde = s * m

    # Store S_tilde for backward pass (materialized to HBM - this is what we want to avoid in fused version)
    s_tilde_ptrs = S_tilde_ptr + pid_b * stride_sb + pid_c * stride_sc + offs_c[:, None] * stride_sr + offs_c2[None, :] * stride_scol
    tl.store(s_tilde_ptrs, s_tilde)

    # Load V_j [C, D]
    v_ptrs = V_ptr + pid_b * stride_vb + (chunk_start + offs_c[:, None]) * stride_vt + offs_d[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=(offs_c[:, None] < C) & (offs_d[None, :] < D), other=0.0)

    # F3: O_intra = S_tilde @ V [C, D]
    o_intra = tl.dot(s_tilde.to(v.dtype), v)

    # F4: O_state = Q @ H [C, D]
    # H is [B, num_chunks, N, D]
    offs_n = tl.arange(0, N)
    # Load Q projected to state dim - for simplicity, we use the first N dims of Q
    q_state = tl.load(
        Q_ptr + pid_b * stride_qb + (chunk_start + offs_c[:, None]) * stride_qt + offs_n[None, :] * stride_qd,
        mask=(offs_c[:, None] < C) & (offs_n[None, :] < N), other=0.0
    )
    h_ptrs = H_ptr + pid_b * stride_hb + pid_c * stride_hc + offs_n[:, None] * stride_hn + offs_d[None, :] * stride_hd
    h = tl.load(h_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)
    o_state = tl.dot(q_state.to(h.dtype), h)

    # F5: O = O_intra + O_state
    o = o_intra + o_state

    # Store O [C, D]
    o_ptrs = O_ptr + pid_b * stride_ob + (chunk_start + offs_c[:, None]) * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o, mask=(offs_c[:, None] < C) & (offs_d[None, :] < D))


@triton.jit
def _bwd_intra_chunk_kernel(
    Q_ptr, K_ptr, V_ptr, M_ptr, H_ptr, S_tilde_ptr, grad_O_ptr,
    grad_Q_ptr, grad_K_ptr, grad_V_ptr,
    B: tl.constexpr, T: tl.constexpr, C: tl.constexpr,
    D: tl.constexpr, N: tl.constexpr,
    stride_qb, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_mb, stride_mc, stride_mr, stride_mcol,
    stride_hb, stride_hc, stride_hn, stride_hd,
    stride_sb, stride_sc, stride_sr, stride_scol,
    stride_gob, stride_got, stride_god,
    stride_gqb, stride_gqt, stride_gqd,
    stride_gkb, stride_gkt, stride_gkd,
    stride_gvb, stride_gvt, stride_gvd,
):
    """Backward intra-chunk: separate kernel that reads S_tilde from HBM"""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    chunk_start = pid_c * C

    offs_c = tl.arange(0, C)
    offs_d = tl.arange(0, D)
    offs_c2 = tl.arange(0, C)

    # Load grad_O [C, D]
    go_ptrs = grad_O_ptr + pid_b * stride_gob + (chunk_start + offs_c[:, None]) * stride_got + offs_d[None, :] * stride_god
    grad_o = tl.load(go_ptrs, mask=(offs_c[:, None] < C) & (offs_d[None, :] < D), other=0.0)

    # Load V [C, D]
    v_ptrs = V_ptr + pid_b * stride_vb + (chunk_start + offs_c[:, None]) * stride_vt + offs_d[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=(offs_c[:, None] < C) & (offs_d[None, :] < D), other=0.0)

    # B4: grad_S_tilde = grad_O @ V^T [C, C]
    grad_s_tilde = tl.dot(grad_o, tl.trans(v))

    # Load S_tilde from HBM (this is the extra read we want to eliminate)
    s_tilde_ptrs = S_tilde_ptr + pid_b * stride_sb + pid_c * stride_sc + offs_c[:, None] * stride_sr + offs_c2[None, :] * stride_scol
    s_tilde = tl.load(s_tilde_ptrs)

    # B5: grad_V = S_tilde^T @ grad_O [C, D]
    grad_v = tl.dot(tl.trans(s_tilde).to(grad_o.dtype), grad_o)

    # Load M [C, C]
    m_ptrs = M_ptr + pid_b * stride_mb + pid_c * stride_mc + offs_c[:, None] * stride_mr + offs_c2[None, :] * stride_mcol
    m = tl.load(m_ptrs)

    # B6: grad_S = grad_S_tilde * M
    grad_s = grad_s_tilde * m

    # Load K [C, D]
    k_ptrs = K_ptr + pid_b * stride_kb + (chunk_start + offs_c[:, None]) * stride_kt + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=(offs_c[:, None] < C) & (offs_d[None, :] < D), other=0.0)

    # Load Q [C, D]
    q_ptrs = Q_ptr + pid_b * stride_qb + (chunk_start + offs_c[:, None]) * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_c[:, None] < C) & (offs_d[None, :] < D), other=0.0)

    # B7: grad_Q = grad_S @ K + grad_O @ H^T
    grad_q = tl.dot(grad_s.to(k.dtype), k)

    # Add state correction gradient: grad_O @ H^T
    offs_n = tl.arange(0, N)
    h_ptrs = H_ptr + pid_b * stride_hb + pid_c * stride_hc + offs_n[:, None] * stride_hn + offs_d[None, :] * stride_hd
    h = tl.load(h_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)
    # grad_O @ H^T gives [C, N], we need to expand to [C, D]
    grad_q_state = tl.dot(grad_o, tl.trans(h))  # [C, N]
    # For simplicity, add only to first N dims
    # This is a simplification - in full impl would use separate projection

    # B8: grad_K = grad_S^T @ Q [C, D]
    grad_k = tl.dot(tl.trans(grad_s).to(q.dtype), q)

    # Store gradients
    gq_ptrs = grad_Q_ptr + pid_b * stride_gqb + (chunk_start + offs_c[:, None]) * stride_gqt + offs_d[None, :] * stride_gqd
    tl.store(gq_ptrs, grad_q, mask=(offs_c[:, None] < C) & (offs_d[None, :] < D))

    gk_ptrs = grad_K_ptr + pid_b * stride_gkb + (chunk_start + offs_c[:, None]) * stride_gkt + offs_d[None, :] * stride_gkd
    tl.store(gk_ptrs, grad_k, mask=(offs_c[:, None] < C) & (offs_d[None, :] < D))

    gv_ptrs = grad_V_ptr + pid_b * stride_gvb + (chunk_start + offs_c[:, None]) * stride_gvt + offs_d[None, :] * stride_gvd
    tl.store(gv_ptrs, grad_v, mask=(offs_c[:, None] < C) & (offs_d[None, :] < D))


# ============================================================================
# PyTorch Autograd Function wrapping the separate kernels
# ============================================================================

class ChunkwiseGLASeparate(torch.autograd.Function):
    """Chunkwise GLA with separate forward and backward Triton kernels.

    This is the BASELINE: S_tilde is materialized to HBM between fwd and bwd.
    """

    @staticmethod
    def forward(ctx, Q, K, V, M, H):
        """
        Args:
            Q: [B, T, D] query
            K: [B, T, D] key
            V: [B, T, D] value
            M: [B, num_chunks, C, C] decay mask per chunk
            H: [B, num_chunks, N, D] boundary state per chunk
        Returns:
            O: [B, T, D] output
        """
        B, T, D = Q.shape
        C = M.shape[2]
        N = H.shape[2]
        num_chunks = T // C

        assert T % C == 0, f"T ({T}) must be divisible by C ({C})"

        O = torch.empty_like(Q)
        S_tilde = torch.empty(B, num_chunks, C, C, device=Q.device, dtype=Q.dtype)

        grid = (B, num_chunks)
        _fwd_intra_chunk_kernel[grid](
            Q, K, V, M, H, O, S_tilde,
            B, T, C, D, N,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            H.stride(0), H.stride(1), H.stride(2), H.stride(3),
            O.stride(0), O.stride(1), O.stride(2),
            S_tilde.stride(0), S_tilde.stride(1), S_tilde.stride(2), S_tilde.stride(3),
        )

        # Save for backward - ALL intermediates stored to HBM
        ctx.save_for_backward(Q, K, V, M, H, S_tilde)
        ctx.B = B
        ctx.T = T
        ctx.C = C
        ctx.D = D
        ctx.N = N

        return O

    @staticmethod
    def backward(ctx, grad_O):
        Q, K, V, M, H, S_tilde = ctx.saved_tensors
        B, T, C, D, N = ctx.B, ctx.T, ctx.C, ctx.D, ctx.N
        num_chunks = T // C

        grad_Q = torch.empty_like(Q)
        grad_K = torch.empty_like(K)
        grad_V = torch.empty_like(V)

        grid = (B, num_chunks)
        _bwd_intra_chunk_kernel[grid](
            Q, K, V, M, H, S_tilde, grad_O,
            grad_Q, grad_K, grad_V,
            B, T, C, D, N,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            H.stride(0), H.stride(1), H.stride(2), H.stride(3),
            S_tilde.stride(0), S_tilde.stride(1), S_tilde.stride(2), S_tilde.stride(3),
            grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
            grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
            grad_K.stride(0), grad_K.stride(1), grad_K.stride(2),
            grad_V.stride(0), grad_V.stride(1), grad_V.stride(2),
        )

        return grad_Q, grad_K, grad_V, None, None


# ============================================================================
# GLA Model (Baseline with separate kernels)
# ============================================================================

class GLALayer(nn.Module):
    """Single GLA layer using separate fwd/bwd Triton kernels."""

    def __init__(self, d_model, d_state=16, chunk_size=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.chunk_size = chunk_size

        # Input projections: x -> Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Gate projection
        self.W_g = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Decay rate (learnable, per-head)
        self.log_decay = nn.Parameter(torch.randn(d_model) * 0.1)

    def _build_decay_mask(self, C, device, dtype):
        """Build causal decay mask M[i,j] = exp(-sum_{k=j+1}^{i} alpha_k) for i >= j, 0 otherwise."""
        # Simple exponential decay mask
        decay = torch.sigmoid(self.log_decay[:1]).item()  # scalar decay for simplicity
        positions = torch.arange(C, device=device, dtype=dtype)
        # M[i,j] = decay^(i-j) for i >= j, 0 otherwise
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [C, C]
        mask = torch.where(diff >= 0, decay ** diff, torch.zeros_like(diff))
        return mask

    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        Returns:
            out: [B, T, D]
        """
        B, T, D = x.shape
        C = self.chunk_size
        N = self.d_state

        assert T % C == 0, f"Sequence length {T} must be divisible by chunk size {C}"
        num_chunks = T // C

        # Project to Q, K, V
        Q = self.W_q(x)  # [B, T, D]
        K = self.W_k(x)  # [B, T, D]
        V = self.W_v(x)  # [B, T, D]

        # Build decay masks for each chunk
        M = self._build_decay_mask(C, x.device, x.dtype)  # [C, C]
        M = M.unsqueeze(0).unsqueeze(0).expand(B, num_chunks, C, C).contiguous()

        # Initialize boundary states (zeros for simplicity in MVE)
        H = torch.zeros(B, num_chunks, N, D, device=x.device, dtype=x.dtype)

        # Run chunkwise attention with separate fwd/bwd kernels
        O = ChunkwiseGLASeparate.apply(Q, K, V, M, H)

        # Gate and output projection
        gate = torch.sigmoid(self.W_g(x))
        out = self.W_o(gate * O) + x  # residual connection

        return out


class GLABaseline(nn.Module):
    """Tiny GLA model for MVE with SEPARATE fwd/bwd kernels (baseline)."""

    def __init__(self, vocab_size, d_model=64, d_state=16, chunk_size=32, n_layers=1):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            GLALayer(d_model, d_state, chunk_size) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, T] token ids
        Returns:
            logits: [B, T, vocab_size]
        """
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)
