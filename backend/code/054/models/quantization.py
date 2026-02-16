"""
Simulated INT4 and FP8 quantization for GLA attention.

Implements the core quantization techniques from SageAttention2:
1. Per-thread INT4 quantization (simulated as per-row-group quantization)
2. Q+K smoothing (channel mean subtraction)
3. Smoothing correction (GEMV for bias term)
4. FP8 quantization for SV matmul

These are "simulated" in that we quantize/dequantize in PyTorch rather than
using actual INT4/FP8 tensor core instructions. This validates:
- Accuracy of the quantization scheme (cosine similarity)
- Quality impact on training (loss curves)
- Correctness of the smoothing correction math

The actual throughput gain (4x for INT4, 2x for FP8) is a hardware property
that doesn't need to be proven in simulation.

References:
- Proposal 054, Steps 1-4 of the mathematical formulation
- SageAttention2 (Zhang et al., ICML 2025) — per-thread INT4 + smoothing
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def smooth_qk(
    Q: torch.Tensor,  # (B, H, c, dk)
    K: torch.Tensor,  # (B, H, c, dk)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Q+K Smoothing (Proposal 054, Step 1).

    Subtracts per-block channel means to reduce dynamic range before INT4 quantization.

    Q_smooth = Q - 1_c @ q_bar^T
    K_smooth = K - 1_c @ k_bar^T

    where q_bar = (1/c) * sum_m Q_m, k_bar = (1/c) * sum_n K_n

    Returns:
        Q_smooth: Smoothed Q (zero-mean per channel)
        K_smooth: Smoothed K (zero-mean per channel)
        q_bar: Channel mean of Q (B, H, dk)
        k_bar: Channel mean of K (B, H, dk)
    """
    # Compute channel means: (B, H, dk)
    q_bar = Q.mean(dim=-2)  # Average over the c dimension
    k_bar = K.mean(dim=-2)

    # Subtract means (centering)
    Q_smooth = Q - q_bar.unsqueeze(-2)
    K_smooth = K - k_bar.unsqueeze(-2)

    return Q_smooth, K_smooth, q_bar, k_bar


def quantize_int4_per_thread(
    X: torch.Tensor,  # (B, H, c, dk)
    group_size: int = 8,  # Simulates per-thread grouping in MMA instruction
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-thread INT4 quantization (Proposal 054, Step 2).

    In actual hardware, each thread in mma.m16n8k64 handles a specific subset
    of elements. We simulate this with per-group quantization along the row dim.

    For each group of `group_size` rows:
        scale = max(|X[group]|) / 7
        X_int4 = round(X[group] / scale)  # clamp to [-7, 7]

    Args:
        X: Input tensor (B, H, c, dk)
        group_size: Number of rows per quantization group

    Returns:
        X_q: Quantized tensor (same shape, but values in [-7, 7])
        scales: Per-group scales (B, H, n_groups, 1)
    """
    B, H, c, dk = X.shape

    # Number of groups (simulating per-thread groups)
    # In real hardware, group_size would be determined by the MMA thread mapping
    n_groups = (c + group_size - 1) // group_size

    # Pad if c is not divisible by group_size
    pad_c = n_groups * group_size - c
    if pad_c > 0:
        X_padded = F.pad(X, (0, 0, 0, pad_c))
    else:
        X_padded = X

    # Reshape into groups: (B, H, n_groups, group_size, dk)
    X_grouped = X_padded.reshape(B, H, n_groups, group_size, dk)

    # Per-group max absolute value (scale computation)
    # scale = max(|X|) / 7  (INT4 range is [-7, 7] for signed 4-bit)
    abs_max = X_grouped.abs().amax(dim=(-2, -1), keepdim=True)  # (B, H, n_groups, 1, 1)
    scales = abs_max / 7.0
    scales = scales.clamp(min=1e-10)  # Avoid division by zero

    # Quantize: round to nearest integer in [-7, 7]
    X_q = torch.round(X_grouped / scales).clamp(-7, 7)

    # Dequantize (for simulation — on real hardware this happens in the MMA accumulator)
    X_deq = X_q * scales

    # Reshape back
    X_deq = X_deq.reshape(B, H, n_groups * group_size, dk)
    scales_flat = scales.reshape(B, H, n_groups, 1)

    if pad_c > 0:
        X_deq = X_deq[:, :, :c, :]

    return X_deq, scales_flat


def int4_matmul_with_smoothing(
    Q: torch.Tensor,  # (B, H, c, dk)
    K: torch.Tensor,  # (B, H, c, dk)
    smooth: bool = True,
    group_size: int = 8,
) -> torch.Tensor:
    """
    INT4 QK^T matmul with optional Q+K smoothing (Proposal 054, Steps 1-3).

    Computes P = Q @ K^T using simulated INT4 quantization.

    With smoothing (Step 3):
        P = psi_inv(Q_hat @ K_hat^T)   # INT4 matmul (dequantized)
            + q_bar @ gamma(K)^T         # GEMV correction
            + gamma(Q) @ k_bar^T + q_bar @ k_bar^T   # Bias term b

    Without smoothing:
        P = psi_inv(Q_hat_raw @ K_hat_raw^T)   # Direct INT4, no centering

    Args:
        Q, K: Input tensors (B, H, c, dk)
        smooth: Whether to apply Q+K smoothing
        group_size: Quantization group size (simulates per-thread)

    Returns:
        P: Attention scores (B, H, c, c)
    """
    if smooth:
        # Step 1: Smooth Q and K
        Q_smooth, K_smooth, q_bar, k_bar = smooth_qk(Q, K)

        # Step 2: INT4 quantize smoothed Q and K
        Q_deq, _ = quantize_int4_per_thread(Q_smooth, group_size)
        K_deq, _ = quantize_int4_per_thread(K_smooth, group_size)

        # Step 3: INT4 matmul + correction terms
        # Term 1: INT4 tensor core matmul (dequantized)
        P_int4 = torch.matmul(Q_deq, K_deq.transpose(-2, -1))

        # Term 2: GEMV correction — q_bar @ gamma(K_smooth)^T
        # This is (B, H, dk) @ (B, H, dk, c) = (B, H, c)
        correction_q = torch.einsum('bhd,bhcd->bhc', q_bar, K_smooth)
        # Broadcast to (B, H, c, c): each row of P gets the same correction
        correction_q = correction_q.unsqueeze(-2).expand_as(P_int4)

        # Term 3: gamma(Q_smooth) @ k_bar^T
        correction_k = torch.einsum('bhcd,bhd->bhc', Q_smooth, k_bar)
        correction_k = correction_k.unsqueeze(-1).expand_as(P_int4)

        # Term 4: q_bar @ k_bar^T (scalar per batch/head)
        correction_qk = torch.einsum('bhd,bhd->bh', q_bar, k_bar)
        correction_qk = correction_qk.unsqueeze(-1).unsqueeze(-1).expand_as(P_int4)

        P = P_int4 + correction_q + correction_k + correction_qk

    else:
        # No smoothing — direct INT4 quantization (expected to be less accurate)
        Q_deq, _ = quantize_int4_per_thread(Q, group_size)
        K_deq, _ = quantize_int4_per_thread(K, group_size)
        P = torch.matmul(Q_deq, K_deq.transpose(-2, -1))

    return P


def quantize_fp8(
    X: torch.Tensor,  # Any shape
    scale: float = 448.0,  # Static scale to FP8 E4M3 range
) -> torch.Tensor:
    """
    Simulated FP8 E4M3 quantization (Proposal 054, Step 4).

    FP8 E4M3 has range [-448, 448] with ~3.5 bits of mantissa.
    We simulate by:
    1. Scale to FP8 range
    2. Clamp to representable range
    3. Round to nearest representable value (simulate reduced precision)
    4. Scale back

    In real hardware, this would use FP8 tensor cores directly.
    """
    # Per-tensor scaling (following SageAttention2's static scaling for P)
    abs_max = X.abs().amax()
    if abs_max == 0:
        return X

    # Scale to fit in FP8 E4M3 range
    s = scale / abs_max.clamp(min=1e-10)

    # Simulate FP8 E4M3 precision:
    # E4M3 has 3 mantissa bits → precision of 2^(-3) = 0.125 relative
    # We simulate by rounding to a grid of 2^(-3) = 1/8 precision
    X_scaled = X * s
    X_scaled = X_scaled.clamp(-scale, scale)

    # Simulate reduced mantissa precision (3 bits → 8 levels per power of 2)
    # This is an approximation of FP8 rounding
    sign = X_scaled.sign()
    abs_val = X_scaled.abs().clamp(min=1e-10)
    exponent = torch.floor(torch.log2(abs_val))
    mantissa = abs_val / (2.0 ** exponent)
    # Round mantissa to 3-bit precision (8 levels: 1.000 to 1.875)
    mantissa_q = torch.round(mantissa * 8.0) / 8.0
    X_fp8 = sign * mantissa_q * (2.0 ** exponent)

    # Scale back
    X_deq = X_fp8 / s
    return X_deq


def fp8_matmul(
    A: torch.Tensor,  # (B, H, m, k)
    B_mat: torch.Tensor,  # (B, H, k, n)
) -> torch.Tensor:
    """
    Simulated FP8 matmul for the SV computation (Proposal 054, Step 4).

    P (attention scores) is statically scaled to FP8 E4M3 range (x448).
    V uses per-channel FP8 scaling.

    In real hardware, this would use FP8 WGMMA with FP32 accumulator.
    We simulate the quantization error but compute in FP32.
    """
    A_fp8 = quantize_fp8(A)
    B_fp8 = quantize_fp8(B_mat)
    return torch.matmul(A_fp8, B_fp8)


def compute_cosine_similarity(
    P_ref: torch.Tensor,   # (B, H, c, c) — BF16 reference
    P_test: torch.Tensor,  # (B, H, c, c) — quantized result
) -> float:
    """
    Compute mean cosine similarity between reference and quantized QK^T.

    This is the primary accuracy metric from the proposal:
    - Target: > 99% with smoothing
    - Expected: < 85% without smoothing
    """
    # Flatten spatial dims for cosine similarity
    P_ref_flat = P_ref.reshape(-1, P_ref.shape[-1])
    P_test_flat = P_test.reshape(-1, P_test.shape[-1])

    cos_sim = F.cosine_similarity(P_ref_flat, P_test_flat, dim=-1)
    return cos_sim.mean().item()
