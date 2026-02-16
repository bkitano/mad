# 208: GEMM-Softmax Two-Stage Async Pipelining

**Category**: kernel
**Gain type**: efficiency
**Source**: Shah, Bikshandi, Zhang, Thakkar, Ramani & Dao (2024) — FlashAttention-3 (NeurIPS 2024)
**Paper**: [papers/flashattention3-pingpong-scheduling.pdf]
**Documented**: 2026-02-15

## Description

In tiled attention kernels (FlashAttention-2, TFLA), each iteration of the inner loop computes two GEMMs and one softmax/gate operation in sequence:

1. **GEMM0** ($S = QK^\top$): Score computation
2. **Softmax/Gate** ($\tilde{P} = \text{softmax}(S)$ or $\tilde{P} = \text{gate}(S)$): Non-linear transformation
3. **GEMM1** ($O \mathrel{+}= \tilde{P}V$): Output accumulation

The key problem: softmax depends on GEMM0's output $S$, and GEMM1 depends on softmax's output $\tilde{P}$, creating sequential dependencies within each iteration. On Hopper GPUs, the tensor core (TC) unit and the special function unit (SFU, for exponentials) are separate hardware — but naively, the TC sits idle during softmax, and the SFU sits idle during GEMMs. The H100 has 989 TFLOPS of FP16 matmul throughput but only 3.9 TFLOPS of special function throughput (254× less), so even a small softmax computation can consume a significant fraction of the cycle time.

**Two-stage GEMM-softmax pipelining** breaks these sequential dependencies *within a single warp group* by using the asynchronous nature of WGMMA: "commit but do not wait" allows the next GEMM to begin executing asynchronously on the tensor cores while the current iteration's softmax runs on the SFU/ALU pipeline. Specifically:

- In iteration $j$, after GEMM0 computes $S_{\text{next}} = QK_j^\top$, the warp group commits GEMM0 to the TC but **does not wait** for it.
- While GEMM0 of iteration $j$ executes asynchronously on the TC, the warp group computes **softmax of the previous iteration's** $S_{\text{cur}}$ on the SFU.
- After softmax completes, the warp group waits for GEMM0, then issues GEMM1 ($O \mathrel{+}= \tilde{P}_{\text{cur}} V_{j-1}$).

This overlaps the most expensive non-GEMM operation (exponential in softmax) with the asynchronous execution of GEMM0, increasing tensor core utilization from ~570 TFLOPS to ~620-660 TFLOPS on the FlashAttention-3 forward pass.

**Why this matters for TFLA:** TFLA's intra-chunk computation has the same three-operation structure: GEMM0 ($Q_i K_j^\top$), gate application ($S_{ij} \odot D_{ij}$), and GEMM1 ($S_{ij} V_j$). The gate operation — which for mLSTM involves exponential or sigmoid activations and causal mask application — can be overlapped with the next tile's GEMM0 using this exact 2-stage pipelining pattern. The paper reports that TFLA's current Triton implementation does not exploit this overlap, leaving performance on the table.

## Mathematical Form

**Standard FlashAttention-2 inner loop (no overlap):**

For each key block $j = 0, \ldots, T_c - 1$:

$$
S_i^{(j)} = Q_i K_j^\top \quad \text{(GEMM0 — wait for result)}
$$
$$
m_i^{\text{old}} \leftarrow m_i, \quad m_i \leftarrow \max(m_i^{\text{old}}, \text{rowmax}(S_i^{(j)}))
$$
$$
\tilde{P}_i^{(j)} = \exp(S_i^{(j)} - m_i), \quad \ell_i \leftarrow \exp(m_i^{\text{old}} - m_i) \ell_i + \text{rowsum}(\tilde{P}_i^{(j)})
$$
$$
O_i \leftarrow \text{diag}(\exp(m_i^{\text{old}} - m_i))^{-1} O_i + \tilde{P}_i^{(j)} V_j \quad \text{(GEMM1 — wait for result)}
$$

All operations are sequential within each iteration.

**Two-stage pipelined inner loop (FlashAttention-3, Algorithm 2):**

Prologue:
$$
S_{\text{cur}} = Q_i K_0^\top \quad \text{(GEMM0, commit and wait)}
$$

For each key block $j = 1, \ldots, T_c - 1$:

$$
S_{\text{next}} = Q_i K_j^\top \quad \text{(GEMM0, commit but DO NOT WAIT)}
$$

$$
\underbrace{m_i, \tilde{P}_{\text{next}} \leftarrow \text{softmax}(S_{\text{next}})}_{\text{SFU: runs while GEMM0 executes on TC}} \quad \underbrace{O_i \mathrel{+}= \tilde{P}_{\text{cur}} V_{j-1}}_{\text{GEMM1: wait for WGMMA } Q_i K_j^\top}
$$

$$
S_{\text{cur}} \leftarrow S_{\text{next}}
$$

Epilogue:
$$
O_i \leftarrow O_i + \tilde{P}_{\text{last}} V_{T_c - 1} \quad \text{(final GEMM1)}
$$

**Key insight:** The softmax of iteration $j$ uses $S_{\text{next}}$ from the *previous* GEMM0, while the current GEMM0 ($Q_i K_j^\top$) runs asynchronously. The dependency chain becomes:

$$
\text{GEMM0}_{j} \xrightarrow{\text{async}} \text{softmax}_{j} \parallel \text{GEMM0}_{j+1} \xrightarrow{\text{wait}} \text{GEMM1}_{j}
$$

**Register buffer requirement:**

The 2-stage pipeline requires keeping two score matrices simultaneously in registers:

$$
\text{Extra registers} = B_r \times B_c \times \text{sizeof(float)} \text{ per threadblock}
$$

For $B_r = B_c = 128$: $128 \times 128 \times 4 = 64$ KB of extra registers, which limits occupancy and constrains tile sizes.

**3-stage extension (Appendix B.3):**

A 3-stage variant further overlaps the second WGMMA (GEMM1) with softmax:

$$
\text{GEMM0}_{j+2} \parallel \text{softmax}_{j+1} \parallel \text{GEMM1}_{j}
$$

This requires three register buffers and even higher register pressure, making the trade-off between pipeline depth and tile size more difficult.

## Complexity

The 2-stage pipelining does **not** reduce FLOPs — it increases hardware utilization by overlapping operations on different functional units.

| Metric | No Overlap (FA2 style) | 2-Stage Pipeline (FA3) |
|--------|----------------------|----------------------|
| TC utilization | ~35% (570 TFLOPS) | ~42% (661 TFLOPS) |
| SFU overlap | 0% | ~100% |
| Extra registers | 0 | $B_r \times B_c \times 4$ B |
| Throughput | 570 TFLOPS | 661 TFLOPS |

**Measured ablation (H100 SXM5, FP16, non-causal, batch=4, seq=8448, 16 heads, d=128):**

| Configuration | TFLOPS/s |
|---------------|---------|
| FA2 algorithm (no overlap, no warp-spec) | 432 |
| + Warp specialization only | 570 |
| + 2-stage GEMM-softmax overlap | **661** |
| FA3 with all optimizations | 661 |

The 2-stage pipelining provides a **16% speedup** on top of warp specialization alone (570 → 661 TFLOPS).

**Combined with pingpong scheduling (inter-warpgroup):**

When two consumer warp groups use pingpong scheduling (trick documented separately in existing warp-specialized pipelining trick 141), the softmax of warp group 1 overlaps with the GEMMs of warp group 2. The 2-stage intra-warpgroup pipelining provides *additional* overlap on top of pingpong:

| Level | What overlaps | Hardware units |
|-------|--------------|----------------|
| Pingpong (inter-warpgroup) | WG1 softmax ‖ WG2 GEMM | SFU ‖ TC |
| 2-stage (intra-warpgroup) | GEMM0$_{j+1}$ ‖ softmax$_j$ | TC ‖ SFU (same WG) |

**Memory:** No additional HBM access — the extra registers store $S_{\text{next}}$ which was already computed.

## Applicability

- **FlashAttention-3 forward pass (primary):** Directly implemented and validated. Provides 16% speedup over warp-specialized-only baseline on H100.

- **TFLA intra-chunk computation:** TFLA's inner loop has the same GEMM-gate-GEMM structure. The gate operation (sigmoid/exponential for mLSTM, or causal mask for vanilla linear attention) can be overlapped with the next tile's $Q_i K_j^\top$ GEMM using the same 2-stage pattern. This requires the TFLA Triton kernel to be rewritten using WGMMA async semantics (either via ThunderKittens or CUTLASS CUDA).

- **GLA / DeltaNet chunkwise kernels:** The chunkwise-parallel kernels for Gated Linear Attention and DeltaNet have similar GEMM-gate-GEMM patterns in their intra-chunk computation. The 2-stage overlap applies directly.

- **Any tiled kernel with GEMM + non-GEMM interleaving:** The technique generalizes to any fused kernel where matmuls alternate with elementwise or reduction operations. Examples: fused MLP layers (GEMM + activation + GEMM), fused attention + FFN.

- **Backward pass:** FlashAttention-3's backward pass uses a different decomposition (5 GEMMs + 1 exponential), and the pipelining structure is more constrained by register pressure. The 2-stage overlap is less beneficial for backward because the backward is more compute-heavy relative to non-GEMM ops.

## Limitations

- **Hopper-specific:** Requires WGMMA's "commit but do not wait" asynchronous execution model, which is only available on Hopper (SM90) and Blackwell (SM100). On Ampere (SM80), the MMA instruction is synchronous and this overlap is not possible.

- **Register pressure:** The extra $S_{\text{next}}$ buffer consumes $B_r \times B_c \times 4$ bytes of registers per threadblock. For $B_r = B_c = 128$, this is 64 KB — a significant fraction of the 256 KB register file per SM on H100. This may force smaller tile sizes, partially offsetting the throughput gain.

- **Compiler interference:** The NVCC compiler may reorder instructions in ways that break the carefully crafted WGMMA-softmax interleaving. The FlashAttention-3 authors verified via SASS analysis that the compiler preserves the intended schedule (Section B.2), but this is fragile and may break across CUDA toolkit versions.

- **Diminishing returns with FP8:** With FP8 precision, the matmul throughput doubles (1978 TFLOPS) while SFU throughput stays at 3.9 TFLOPS, making the exponential an even larger fraction of the cycle. The 2-stage overlap becomes more important but also harder to fully hide.

- **Not available in Triton:** Triton's compilation model does not expose WGMMA async commit/wait semantics, so this optimization cannot be implemented in Triton-based TFLA kernels. Requires CUTLASS/CuTe (C++) or ThunderKittens.

- **3-stage variant has high register cost:** The 3-stage extension that further overlaps GEMM1 requires three $B_r \times B_c$ register buffers, making it impractical for large tile sizes.

## Implementation Notes

```python
# Pseudocode for 2-stage GEMM-softmax pipelined consumer warpgroup
# (Algorithm 2 from FlashAttention-3, adapted for TFLA)
#
# Key: WGMMA "commit but do not wait" enables async execution

def fa3_consumer_2stage(Q_i, K_blocks, V_blocks, T_c):
    """
    FlashAttention-3 consumer warpgroup with 2-stage pipelining.

    The critical pattern:
      WGMMA(Q, K[j]) → commit, DO NOT WAIT → softmax(S_prev) → WAIT → WGMMA(P, V[j-1])
    """
    # Initialize accumulators in registers
    O_i = zeros(B_r, d)    # output accumulator (FP32, in registers)
    ell_i = zeros(B_r)     # log-sum-exp (FP32)
    m_i = full(B_r, -inf)  # row max (FP32)

    # === PROLOGUE: Compute first S tile ===
    wait_for(K_blocks[0])  # wait for TMA load
    S_cur = wgmma(Q_i, K_blocks[0].T)  # GEMM0, commit AND wait

    # Compute softmax of first tile
    m_i, P_cur, ell_i = softmax_update(S_cur, m_i, ell_i)

    # === STEADY STATE: 2-stage pipelined loop ===
    for j in range(1, T_c - 1):
        wait_for(K_blocks[j])  # wait for TMA load of K[j]

        # GEMM0: compute next scores — COMMIT BUT DO NOT WAIT
        S_next = wgmma_async(Q_i, K_blocks[j].T)  # TC starts executing

        # While TC computes S_next, do softmax on S_next's data
        # (Actually computing on S_next from PREVIOUS iteration)
        # AND issue GEMM1 for previous tile

        wait_for(V_blocks[j-1])  # wait for TMA load of V[j-1]

        # GEMM1: accumulate output — this WAITS for S_next WGMMA
        wgmma_wait()  # ensure GEMM0 Q@K[j] is done
        O_i = rescale(O_i, m_i) + wgmma(P_cur, V_blocks[j-1])

        # Softmax of S_next (overlaps with GEMM1's execution? No —
        # the overlap is: softmax(S_cur) overlaps with GEMM0(Q, K[j]))
        m_i, P_next, ell_i = softmax_update(S_next, m_i, ell_i)

        # Swap buffers
        S_cur = S_next
        P_cur = P_next

    # === EPILOGUE: Final GEMM1 ===
    wait_for(V_blocks[T_c - 1])
    O_i = rescale(O_i, m_i) + wgmma(P_cur, V_blocks[T_c - 1])
    O_i = diag(ell_i)^{-1} @ O_i

    write_to_hbm(O_i)
```

**Applying to TFLA gate operations:**

For TFLA's mLSTM with sigmoid gating, the softmax is replaced by a gate application:

```python
# TFLA 2-stage pipelined inner loop (conceptual)
for j in range(L // B_Lkv):
    # GEMM0: Q_i @ K_j^T — async commit
    S_next = wgmma_async(Q_tile_i, K_tile_j.T)

    # While TC computes S_next, apply gates to PREVIOUS tile
    # This is the "gate" equivalent of softmax:
    S_cur_gated = S_cur * D_ij_prev  # sigmoid gate + causal mask
    # ^^^ runs on SFU/ALU while TC executes GEMM0

    wgmma_wait()  # wait for S_next

    # GEMM1: accumulate gated scores @ V
    h_acc += wgmma(S_cur_gated, V_tile_prev)

    S_cur = S_next
```

**GPU efficiency analysis:**

1. **Overlaps TC and SFU:** The exponential/sigmoid in softmax/gating runs on the SFU while WGMMA runs on the TC — true hardware parallelism, not just instruction-level overlap.

2. **No extra HBM access:** All overlapped operations use data already in registers/SMEM. The only cost is extra register pressure for the additional $S_{\text{next}}$ buffer.

3. **Coalesced memory:** The pipelining does not change memory access patterns — TMA loads of K and V blocks remain coalesced and async.

4. **Tensor core friendly:** The core operations (GEMM0, GEMM1) still map to WGMMA instructions. The pipelining only reorders their issuance.

5. **Predictable benefit:** The 16% speedup is consistent across sequence lengths and head dimensions, as it comes from overlapping fixed-cost SFU operations rather than reducing memory traffic.

## References

- Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. NeurIPS 2024. arXiv:2407.08608.
- Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). Tiled Flash Linear Attention. arXiv:2503.14376 (TFLA, trick 158).
- Soi, R., Yadav, R., Kjolstad, F., Aiken, A., Dehnavi, M. M., Garland, M., & Bauer, M. (2025). Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs. arXiv:2512.18134.
- NVIDIA CUTLASS 3.x Ping-Pong GEMM Kernel. https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/
- Spector, B. F., et al. (2024). ThunderKittens: Simple, Fast, and Adorable AI Kernels. arXiv:2410.20399 (trick 202).
