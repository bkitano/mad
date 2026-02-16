"""
FlashRNN-Style Fused Inter-Chunk State Recurrence for GLA (Diagonal Case)

Proposal 057: Replace the separate inter-chunk state propagation kernel
with a FlashRNN-style persistent fused recurrence where:
- Per-chunk transition matrices A_k^{(C)} are diagonal (GLA case)
- State h_k lives in registers/SRAM for the entire scan
- No HBM round-trips for intermediate states

The inter-chunk state scan:
    h_0 = 0
    h_k = gamma_k * h_{k-1} + h_hat_k,  k = 1, ..., G

For GLA (diagonal A_k), this decomposes into dk independent scalar-vector recurrences:
    h_k[i, :] = gamma_k[i] * h_{k-1}[i, :] + h_hat_k[i, :],  i = 1, ..., dk

Baseline: Each step reads h_{k-1} from HBM, computes, writes h_k to HBM
Proposed: State h stays in registers; only reads gamma_k and h_hat_k from HBM
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Baseline: Sequential loop with HBM round-trips (fla-org style)
# ============================================================================

@triton.jit
def baseline_inter_chunk_scan_kernel(
    h_local_ptr,   # [B, G, H, dk, dv] — isolated chunk states (HBM)
    gamma_ptr,     # [B, G, H, dk] — cumulative per-chunk decays (HBM)
    h_out_ptr,     # [B, G, H, dk, dv] — propagated states (HBM)
    h_buf_ptr,     # [B, H, dk, dv] — buffer for h_{k-1} in HBM
    B: tl.constexpr, G: tl.constexpr, H: tl.constexpr,
    dk: tl.constexpr, dv: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    Baseline inter-chunk scan: reads/writes state from HBM at every step.

    Each program handles one (batch, head, row) combination.
    For each chunk k:
      1. Read h_{k-1}[row, :] from HBM buffer
      2. Read gamma_k[row] from HBM
      3. Read h_hat_k[row, :] from HBM
      4. Compute h_k[row, :] = gamma_k[row] * h_{k-1}[row, :] + h_hat_k[row, :]
      5. Write h_k[row, :] to HBM output AND buffer

    HBM traffic per step: read gamma (1), read h_hat (dv), read h_prev (dv), write h_k (dv), write buf (dv)
    = 1 + 4*dv elements per step per row
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    row_id = tl.program_id(2)

    # Offsets for this (batch, head, row)
    dv_offsets = tl.arange(0, BLOCK_DV)

    # Buffer offset for h_{k-1}: [B, H, dk, dv]
    buf_base = batch_id * H * dk * dv + head_id * dk * dv + row_id * dv

    # Zero-initialize buffer (h_0 = 0)
    tl.store(h_buf_ptr + buf_base + dv_offsets, tl.zeros([BLOCK_DV], dtype=tl.float32))

    for k in range(G):
        # Strides: [B, G, H, dk, dv] for h_local and h_out
        chunk_base = (batch_id * G * H * dk * dv
                      + k * H * dk * dv
                      + head_id * dk * dv
                      + row_id * dv)
        # Strides: [B, G, H, dk] for gamma
        gamma_base = (batch_id * G * H * dk
                      + k * H * dk
                      + head_id * dk
                      + row_id)

        # Step 1: Read h_{k-1} from HBM buffer
        h_prev = tl.load(h_buf_ptr + buf_base + dv_offsets).to(tl.float32)

        # Step 2: Read gamma_k scalar
        gamma = tl.load(gamma_ptr + gamma_base).to(tl.float32)

        # Step 3: Read h_hat_k row from HBM
        h_hat = tl.load(h_local_ptr + chunk_base + dv_offsets).to(tl.float32)

        # Step 4: Compute recurrence
        h_new = gamma * h_prev + h_hat

        # Step 5: Write h_k to output AND buffer
        tl.store(h_out_ptr + chunk_base + dv_offsets, h_new.to(tl.float32))
        tl.store(h_buf_ptr + buf_base + dv_offsets, h_new.to(tl.float32))


# ============================================================================
# Proposed: FlashRNN-style fused scan (state in registers)
# ============================================================================

@triton.jit
def flashrnn_inter_chunk_scan_kernel(
    h_local_ptr,   # [B, G, H, dk, dv] — isolated chunk states (HBM)
    gamma_ptr,     # [B, G, H, dk] — cumulative per-chunk decays (HBM)
    h_out_ptr,     # [B, G, H, dk, dv] — propagated states (HBM)
    B: tl.constexpr, G: tl.constexpr, H: tl.constexpr,
    dk: tl.constexpr, dv: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    FlashRNN-style fused inter-chunk scan: state lives in registers.

    Each program handles one (batch, head, row) combination.
    The state vector h_row (dv elements) stays in registers for all G steps.

    For each chunk k:
      1. Read gamma_k[row] from HBM (1 scalar)
      2. Read h_hat_k[row, :] from HBM (dv elements)
      3. Compute h_row = gamma_k[row] * h_row + h_hat_k[row, :] (in registers)
      4. Write h_row to HBM output (dv elements)

    HBM traffic per step: read gamma (1), read h_hat (dv), write h_k (dv)
    = 1 + 2*dv elements per step per row
    SAVES: one full dv read (h_{k-1}) per step vs baseline
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    row_id = tl.program_id(2)

    dv_offsets = tl.arange(0, BLOCK_DV)

    # State row lives in REGISTERS for the entire scan — key FlashRNN insight
    h_row = tl.zeros([BLOCK_DV], dtype=tl.float32)

    for k in range(G):
        # Strides: [B, G, H, dk, dv] for h_local and h_out
        chunk_base = (batch_id * G * H * dk * dv
                      + k * H * dk * dv
                      + head_id * dk * dv
                      + row_id * dv)
        # Strides: [B, G, H, dk] for gamma
        gamma_base = (batch_id * G * H * dk
                      + k * H * dk
                      + head_id * dk
                      + row_id)

        # Load gamma (1 scalar) and h_hat (dv elements) from HBM
        gamma = tl.load(gamma_ptr + gamma_base).to(tl.float32)
        h_hat = tl.load(h_local_ptr + chunk_base + dv_offsets).to(tl.float32)

        # Recurrence in registers — NO HBM read of h_{k-1}
        h_row = gamma * h_row + h_hat

        # Write result to HBM (needed by downstream intra-chunk output kernel)
        tl.store(h_out_ptr + chunk_base + dv_offsets, h_row.to(tl.float32))


# ============================================================================
# PyTorch wrappers
# ============================================================================

def baseline_inter_chunk_scan(
    h_local: torch.Tensor,  # [B, G, H, dk, dv]
    gamma: torch.Tensor,    # [B, G, H, dk]
) -> torch.Tensor:
    """
    Baseline inter-chunk scan using HBM buffer for state.

    Simulates fla-org's sequential loop where each step reads h_{k-1} from HBM.
    """
    B, G, H, dk, dv = h_local.shape
    assert gamma.shape == (B, G, H, dk), f"gamma shape mismatch: {gamma.shape}"

    h_out = torch.empty_like(h_local)
    h_buf = torch.zeros(B, H, dk, dv, device=h_local.device, dtype=torch.float32)

    BLOCK_DV = triton.next_power_of_2(dv)

    grid = (B, H, dk)
    baseline_inter_chunk_scan_kernel[grid](
        h_local, gamma, h_out, h_buf,
        B, G, H, dk, dv,
        BLOCK_DV=BLOCK_DV,
    )
    return h_out


def flashrnn_inter_chunk_scan(
    h_local: torch.Tensor,  # [B, G, H, dk, dv]
    gamma: torch.Tensor,    # [B, G, H, dk]
) -> torch.Tensor:
    """
    FlashRNN-style fused inter-chunk scan with state in registers.

    Key difference from baseline: h_{k-1} is never read from HBM;
    it stays in Triton registers across all G loop iterations.
    """
    B, G, H, dk, dv = h_local.shape
    assert gamma.shape == (B, G, H, dk), f"gamma shape mismatch: {gamma.shape}"

    h_out = torch.empty_like(h_local)

    BLOCK_DV = triton.next_power_of_2(dv)

    grid = (B, H, dk)
    flashrnn_inter_chunk_scan_kernel[grid](
        h_local, gamma, h_out,
        B, G, H, dk, dv,
        BLOCK_DV=BLOCK_DV,
    )
    return h_out


def pytorch_inter_chunk_scan(
    h_local: torch.Tensor,  # [B, G, H, dk, dv]
    gamma: torch.Tensor,    # [B, G, H, dk]
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation for correctness verification.

    h_0 = 0
    h_k = diag(gamma_k) * h_{k-1} + h_hat_k
    """
    B, G, H, dk, dv = h_local.shape
    h_out = torch.zeros_like(h_local)
    h_prev = torch.zeros(B, H, dk, dv, device=h_local.device, dtype=h_local.dtype)

    for k in range(G):
        # gamma_k: [B, H, dk] -> [B, H, dk, 1] for broadcasting
        gamma_k = gamma[:, k, :, :].unsqueeze(-1)  # [B, H, dk, 1]
        h_hat_k = h_local[:, k, :, :, :]           # [B, H, dk, dv]
        h_prev = gamma_k * h_prev + h_hat_k
        h_out[:, k, :, :, :] = h_prev

    return h_out
