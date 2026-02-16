"""
Baseline: 3 separate operations for the linear RNN layer MVE.

Operations (each a separate kernel launch):
1. Linear projection: V = x @ W_V  (GEMM: [B,T,d] @ [d, d_v] -> [B,T,d_v])
2. Gated scan: s_t = gamma_t * s_{t-1} + v_t  (sequential scan per feature dim)
3. Output gating: out = SiLU(gate) * scan_output  (elementwise)

This baseline uses PyTorch ops (which launch separate CUDA kernels for each op)
and Triton kernels for a fairer comparison.

From proposal 040 MVE:
  - Single-head, d=256, d_v=64, T=2048, B=4
  - Forward-pass only
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================
# PyTorch baseline (reference implementation, separate kernels)
# ============================================================

def pytorch_baseline_forward(
    x: torch.Tensor,      # [B, T, d]
    W_V: torch.Tensor,    # [d, d_v]
    gamma: torch.Tensor,  # [B, T, d_v] - decay gates in (0, 1)
    gate: torch.Tensor,   # [B, T, d_v] - gate for SiLU
) -> torch.Tensor:
    """
    Three-kernel baseline using PyTorch ops.
    Each op is a separate CUDA kernel launch.
    """
    # Kernel 1: Linear projection (GEMM)
    V = x @ W_V  # [B, T, d_v]

    # Kernel 2: Gated scan (sequential along T)
    B, T, d_v = V.shape
    scan_out = torch.zeros_like(V)
    s = torch.zeros(B, d_v, device=V.device, dtype=V.dtype)
    for t in range(T):
        s = gamma[:, t, :] * s + V[:, t, :]
        scan_out[:, t, :] = s

    # Kernel 3: Output gating (elementwise)
    out = F.silu(gate) * scan_out  # [B, T, d_v]

    return out


# ============================================================
# Triton baseline kernels (3 separate kernel launches)
# ============================================================

@triton.jit
def _matmul_kernel(
    # Pointers
    X_ptr, W_ptr, Out_ptr,
    # Dimensions
    B_T, d, d_v,
    # Strides for X [B*T, d]
    stride_x_bt, stride_x_d,
    # Strides for W [d, d_v]
    stride_w_d, stride_w_dv,
    # Strides for Out [B*T, d_v]
    stride_o_bt, stride_o_dv,
    # Block sizes
    BLOCK_BT: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    """Tiled matmul: Out = X @ W. X:[B*T, d], W:[d, d_v], Out:[B*T, d_v]"""
    pid_bt = tl.program_id(0)
    pid_dv = tl.program_id(1)

    # Row and column offsets
    offs_bt = pid_bt * BLOCK_BT + tl.arange(0, BLOCK_BT)
    offs_dv = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)

    # Accumulator
    acc = tl.zeros((BLOCK_BT, BLOCK_DV), dtype=tl.float32)

    # Tile over d dimension
    for k in range(0, d, BLOCK_D):
        offs_d = k + tl.arange(0, BLOCK_D)

        # Load X tile [BLOCK_BT, BLOCK_D]
        x_ptrs = X_ptr + offs_bt[:, None] * stride_x_bt + offs_d[None, :] * stride_x_d
        x_mask = (offs_bt[:, None] < B_T) & (offs_d[None, :] < d)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile [BLOCK_D, BLOCK_DV]
        w_ptrs = W_ptr + offs_d[:, None] * stride_w_d + offs_dv[None, :] * stride_w_dv
        w_mask = (offs_d[:, None] < d) & (offs_dv[None, :] < d_v)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(x_tile, w_tile)

    # Store result
    out_ptrs = Out_ptr + offs_bt[:, None] * stride_o_bt + offs_dv[None, :] * stride_o_dv
    out_mask = (offs_bt[:, None] < B_T) & (offs_dv[None, :] < d_v)
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def _gated_scan_kernel(
    # Pointers
    V_ptr, Gamma_ptr, Out_ptr,
    # Dimensions
    B, T, d_v,
    # Strides for V, Gamma, Out: [B, T, d_v]
    stride_b, stride_t, stride_dv,
    # Block size
    BLOCK_DV: tl.constexpr,
):
    """
    Gated scan: s_t = gamma_t * s_{t-1} + v_t
    Each program handles one batch element and a block of d_v features.
    Sequential over T (inherent to scan), parallel over B and d_v.
    """
    pid_b = tl.program_id(0)
    pid_dv = tl.program_id(1)

    offs_dv = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    dv_mask = offs_dv < d_v

    # Initialize state
    s = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    for t in range(T):
        # Load v_t and gamma_t
        v_ptrs = V_ptr + pid_b * stride_b + t * stride_t + offs_dv * stride_dv
        g_ptrs = Gamma_ptr + pid_b * stride_b + t * stride_t + offs_dv * stride_dv

        v = tl.load(v_ptrs, mask=dv_mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptrs, mask=dv_mask, other=0.0).to(tl.float32)

        # Scan update: s_t = gamma_t * s_{t-1} + v_t
        s = g * s + v

        # Store output
        out_ptrs = Out_ptr + pid_b * stride_b + t * stride_t + offs_dv * stride_dv
        tl.store(out_ptrs, s.to(Out_ptr.dtype.element_ty), mask=dv_mask)


@triton.jit
def _silu_gate_kernel(
    # Pointers
    Gate_ptr, Scan_ptr, Out_ptr,
    # Total elements
    n_elements,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """Elementwise: out = SiLU(gate) * scan_output"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    gate = tl.load(Gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scan = tl.load(Scan_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) = gate * sigmoid(gate)
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * scan

    tl.store(Out_ptr + offs, out.to(Out_ptr.dtype.element_ty), mask=mask)


def triton_baseline_forward(
    x: torch.Tensor,      # [B, T, d]
    W_V: torch.Tensor,    # [d, d_v]
    gamma: torch.Tensor,  # [B, T, d_v]
    gate: torch.Tensor,   # [B, T, d_v]
) -> torch.Tensor:
    """
    Three separate Triton kernel launches (baseline).
    """
    B, T, d = x.shape
    d_v = W_V.shape[1]

    # Reshape x for matmul: [B*T, d]
    x_flat = x.reshape(B * T, d).contiguous()

    # ---- Kernel 1: Matmul V = x @ W_V ----
    V_flat = torch.empty(B * T, d_v, device=x.device, dtype=x.dtype)
    BLOCK_BT = 32
    BLOCK_D = 64
    BLOCK_DV = 32
    grid_matmul = (triton.cdiv(B * T, BLOCK_BT), triton.cdiv(d_v, BLOCK_DV))
    _matmul_kernel[grid_matmul](
        x_flat, W_V, V_flat,
        B * T, d, d_v,
        x_flat.stride(0), x_flat.stride(1),
        W_V.stride(0), W_V.stride(1),
        V_flat.stride(0), V_flat.stride(1),
        BLOCK_BT=BLOCK_BT, BLOCK_D=BLOCK_D, BLOCK_DV=BLOCK_DV,
    )
    V = V_flat.reshape(B, T, d_v)

    # ---- Kernel 2: Gated scan ----
    scan_out = torch.empty_like(V)
    BLOCK_DV_SCAN = min(64, d_v)  # Process d_v features in blocks
    grid_scan = (B, triton.cdiv(d_v, BLOCK_DV_SCAN))
    _gated_scan_kernel[grid_scan](
        V, gamma, scan_out,
        B, T, d_v,
        V.stride(0), V.stride(1), V.stride(2),
        BLOCK_DV=BLOCK_DV_SCAN,
    )

    # ---- Kernel 3: SiLU gating ----
    out = torch.empty_like(scan_out)
    n_elements = B * T * d_v
    BLOCK_SIZE = 1024
    grid_gate = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _silu_gate_kernel[grid_gate](
        gate.reshape(-1), scan_out.reshape(-1), out.reshape(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
