"""
Fused megakernel: Multiple fusion strategies for linear RNN layer MVE.

Approach 1 - "Scan+Gate Fusion" (fused_scan_gate):
  Fuses the scan and SiLU gating into one kernel. The GEMM (V = x @ W_V)
  runs as a separate kernel (via PyTorch/cuBLAS), then scan+gate is fused.
  This eliminates one HBM round-trip (scan_output intermediate).
  2 kernel launches instead of 3.

Approach 2 - "Full Fusion" (fused_proj_scan_gate):
  All 3 ops in one kernel, per-timestep vector-matrix product.
  V never touches HBM. 1 kernel launch.
  NOTE: This is expected to be slower due to losing tensor core utilization
  on the GEMM (per-timestep scalar ops instead of batched tl.dot).

From proposal 040 MVE:
  - Single-head, d=256, d_v=64, T=2048, B=4
  - Forward-pass only
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Approach 1: Fuse scan + SiLU gating only (GEMM stays separate)
# This is the most practical fusion: scan+gate are both sequential/elementwise
# and fusing them eliminates one HBM intermediate (scan_output).
# ============================================================

@triton.jit
def _fused_scan_gate_kernel(
    # Input pointers
    V_ptr,      # [B, T, d_v] - projected values (from separate GEMM)
    Gamma_ptr,  # [B, T, d_v] - decay gates
    Gate_ptr,   # [B, T, d_v] - output gate for SiLU
    Out_ptr,    # [B, T, d_v] - output
    # Dimensions
    B, T, d_v,
    # Strides for V/Gamma/Gate/Out [B, T, d_v]
    stride_b, stride_t, stride_dv,
    # Block size
    BLOCK_DV: tl.constexpr,
):
    """
    Fused scan + SiLU gating.
    Eliminates one HBM round-trip: scan_output never written to HBM.

    Baseline does (2 kernels):
      kernel 1: read V, gamma → write scan_out to HBM
      kernel 2: read scan_out, gate → write out to HBM
      HBM: read V + gamma + scan_out + gate = 4 * B*T*d_v, write scan_out + out = 2 * B*T*d_v

    Fused does (1 kernel):
      read V, gamma, gate → compute scan + gate → write out to HBM
      HBM: read V + gamma + gate = 3 * B*T*d_v, write out = 1 * B*T*d_v
      Saves: 1 read + 1 write of B*T*d_v = 2 * B*T*d_v bytes
    """
    pid_b = tl.program_id(0)
    pid_dv = tl.program_id(1)

    offs_dv = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    dv_mask = offs_dv < d_v

    # Scan state in registers (FP32)
    s = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    for t in range(T):
        base = pid_b * stride_b + t * stride_t

        # Load v_t, gamma_t, gate_t
        v_ptrs = V_ptr + base + offs_dv * stride_dv
        g_ptrs = Gamma_ptr + base + offs_dv * stride_dv
        gate_ptrs = Gate_ptr + base + offs_dv * stride_dv

        v = tl.load(v_ptrs, mask=dv_mask, other=0.0).to(tl.float32)
        gamma = tl.load(g_ptrs, mask=dv_mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptrs, mask=dv_mask, other=0.0).to(tl.float32)

        # Scan update: s_t = gamma_t * s_{t-1} + v_t
        s = gamma * s + v

        # Fused SiLU gating (scan_out never goes to HBM!)
        silu_gate = gate * tl.sigmoid(gate)
        out = silu_gate * s

        # Write final output
        out_ptrs = Out_ptr + base + offs_dv * stride_dv
        tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=dv_mask)


def triton_fused_scan_gate(
    V: torch.Tensor,      # [B, T, d_v] - already projected
    gamma: torch.Tensor,  # [B, T, d_v]
    gate: torch.Tensor,   # [B, T, d_v]
) -> torch.Tensor:
    """Fuse scan + SiLU gating into one kernel (approach 1)."""
    B, T, d_v = V.shape
    V = V.contiguous()
    gamma = gamma.contiguous()
    gate = gate.contiguous()

    out = torch.empty_like(V)
    BLOCK_DV = min(64, d_v)
    grid = (B, triton.cdiv(d_v, BLOCK_DV))

    _fused_scan_gate_kernel[grid](
        V, gamma, gate, out,
        B, T, d_v,
        V.stride(0), V.stride(1), V.stride(2),
        BLOCK_DV=BLOCK_DV,
    )
    return out


def triton_fused_forward_v2(
    x: torch.Tensor,      # [B, T, d]
    W_V: torch.Tensor,    # [d, d_v]
    gamma: torch.Tensor,  # [B, T, d_v]
    gate: torch.Tensor,   # [B, T, d_v]
) -> torch.Tensor:
    """
    Approach 1: GEMM via PyTorch/cuBLAS + fused scan+gate via Triton.
    2 kernel launches instead of 3.
    """
    # Kernel 1: GEMM (cuBLAS, tensor cores)
    V = x @ W_V  # [B, T, d_v]
    # Kernel 2: Fused scan + SiLU gating
    out = triton_fused_scan_gate(V, gamma, gate)
    return out


# ============================================================
# Approach 2: Full fusion (all 3 ops in one kernel)
# Per-timestep vector-matrix product (no tensor cores)
# Kept for completeness but expected to be slower for GEMM-heavy workloads
# ============================================================

@triton.jit
def _fused_full_kernel(
    X_ptr, W_V_ptr, Gamma_ptr, Gate_ptr, Out_ptr,
    B, T, d, d_v,
    stride_x_b, stride_x_t, stride_x_d,
    stride_w_d, stride_w_dv,
    stride_g_b, stride_g_t, stride_g_dv,
    BLOCK_DV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Full fusion: per-timestep GEMM + scan + gate. 1 kernel launch."""
    pid_b = tl.program_id(0)
    pid_dv = tl.program_id(1)

    offs_dv = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    dv_mask = offs_dv < d_v

    s = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    for t in range(T):
        # Phase 1: v_t = x_t @ W_V[:, block_dv]
        v = tl.zeros((BLOCK_DV,), dtype=tl.float32)
        for k in range(0, d, BLOCK_D):
            offs_d = k + tl.arange(0, BLOCK_D)
            d_mask = offs_d < d

            x_ptrs = X_ptr + pid_b * stride_x_b + t * stride_x_t + offs_d * stride_x_d
            x_tile = tl.load(x_ptrs, mask=d_mask, other=0.0).to(tl.float32)

            w_ptrs = W_V_ptr + offs_d[:, None] * stride_w_d + offs_dv[None, :] * stride_w_dv
            w_mask = d_mask[:, None] & dv_mask[None, :]
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

            v += tl.sum(x_tile[:, None] * w_tile, axis=0)

        # Phase 2: Scan
        base = pid_b * stride_g_b + t * stride_g_t
        gamma_t = tl.load(Gamma_ptr + base + offs_dv * stride_g_dv, mask=dv_mask, other=0.0).to(tl.float32)
        s = gamma_t * s + v

        # Phase 3: SiLU gate + store
        gate_t = tl.load(Gate_ptr + base + offs_dv * stride_g_dv, mask=dv_mask, other=0.0).to(tl.float32)
        out_t = gate_t * tl.sigmoid(gate_t) * s
        tl.store(Out_ptr + base + offs_dv * stride_g_dv, out_t.to(Out_ptr.dtype.element_ty), mask=dv_mask)


def triton_fused_forward(
    x: torch.Tensor,      # [B, T, d]
    W_V: torch.Tensor,    # [d, d_v]
    gamma: torch.Tensor,  # [B, T, d_v]
    gate: torch.Tensor,   # [B, T, d_v]
) -> torch.Tensor:
    """
    Full fusion: GEMM + scan + gate in 1 kernel.
    """
    B, T, d = x.shape
    d_v = W_V.shape[1]

    x = x.contiguous()
    W_V = W_V.contiguous()
    gamma = gamma.contiguous()
    gate = gate.contiguous()

    out = torch.empty(B, T, d_v, device=x.device, dtype=x.dtype)

    BLOCK_DV = min(64, d_v)
    BLOCK_D = 64

    grid = (B, triton.cdiv(d_v, BLOCK_DV))

    _fused_full_kernel[grid](
        x, W_V, gamma, gate, out,
        B, T, d, d_v,
        x.stride(0), x.stride(1), x.stride(2),
        W_V.stride(0), W_V.stride(1),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        BLOCK_DV=BLOCK_DV, BLOCK_D=BLOCK_D,
    )

    return out
