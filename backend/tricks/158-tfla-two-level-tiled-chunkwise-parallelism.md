# 158: TFLA — Two-Level Tiled Chunkwise Parallelism for Linear RNNs

**Category**: kernel
**Gain type**: efficiency
**Source**: Beck, Pöppel, Lippe & Hochreiter (2025) — NeurIPS 2025
**Paper**: [papers/tfla-tiled-flash-linear-attention.pdf]
**Documented**: 2026-02-15

## Description

Standard Flash Linear Attention (FLA) uses a single level of sequence parallelism: the input sequence is divided into chunks of size $L$, chunk boundary states $C_k \in \mathbb{R}^{d_q \times d_v}$ are computed recurrently, and intra-chunk outputs $H^{(k)}$ are computed in parallel via matmuls. However, the chunk size $L$ is limited by SRAM capacity (typically $L = 64$), forcing $\lceil T/L \rceil$ intermediate states to be materialized in HBM. For long sequences ($T \gg L$), this creates excessive memory traffic and **low arithmetic intensity** — the ratio of compute to memory access that determines whether a kernel is compute-bound (fast) or memory-bound (slow).

**Tiled Flash Linear Attention (TFLA)** introduces a **second level of sequence parallelism** within each chunk by tiling the intra-chunk matrix multiplications along the sequence dimension. This decouples the chunk size $L$ from SRAM capacity: the outer chunk size can be arbitrarily large (e.g., $L = 4096$), while inner tiles of size $B_{Lhq} \times B_{Lkv}$ fit in SRAM. The result is:

1. **Fewer intermediate states:** Only $\lceil T/L \rceil$ boundary states in HBM (vs. $\lceil T/64 \rceil$ for FLA)
2. **Higher arithmetic intensity:** Larger chunks mean more compute per byte of HBM access
3. **Tunable memory-compute trade-off:** $L$ controls the balance between memory consumption and quadratic intra-chunk cost

On H100 GPUs, TFLA mLSTM kernels are **faster than FlashAttention 3** for longer sequences and **2× faster than Mamba 2** across all sequence lengths, achieving state-of-the-art kernel throughput for sequence modeling.

## Mathematical Form

**Linear RNN Chunkwise-Parallel Formulation:**

For a linear RNN with scalar forget gate $f_t \in \mathbb{R}$ and input gate $i_t \in \mathbb{R}$:

$$
C_t = f_t \, C_{t-1} + i_t \, k_t \, v_t^\top
$$

The sequence of length $T$ is split into $N_c = \lceil T/L \rceil$ chunks. The chunkwise formulation has two parts:

**1. Inter-chunk recurrence** (sequential across chunks, $O(d_q \cdot d_v)$ state):

$$
C_k = \bar{g}_k \, C_{k-1} + \left(\bar{a}_k \odot K^{(k)}\right)^\top V^{(k)}
$$

where $\bar{g}_k$ is the cumulative forget gate and $\bar{a}_k$ is the cumulative input gate vector for chunk $k$.

**2. Intra-chunk parallel computation** (independent across chunks):

$$
H^{(k)} = H^{(k)}_{\text{intra}} + H^{(k)}_{\text{inter}}
$$

$$
H^{(k)}_{\text{intra}} = S^{(k)} V^{(k)}, \quad H^{(k)}_{\text{inter}} = \bar{Q}^{(k)} C_{k-1}
$$

where $S^{(k)} \in \mathbb{R}^{L \times L}$ is the intra-chunk attention matrix (gated):

$$
S^{(k)} = \frac{1}{\sqrt{d_q}} Q^{(k)} K^{(k)\top} \odot D^{(k)}
$$

and $D^{(k)}$ is the causal gate matrix, $\bar{Q}^{(k)}$ includes gate contributions from the beginning of the chunk.

**The TFLA Key Insight — Fused Tiled Computation:**

The intra-chunk parallel output can be written as a single fused expression with three matmuls:

$$
H^{(k)} = \underbrace{\left(\frac{Q^{(k)} K^{(k)\top}}{\sqrt{d_q}} \odot D^{(k)}\right) V^{(k)}}_{H^{(k)}_{\text{intra}}} + \underbrace{\frac{Q^{(k)}}{\sqrt{d_q}} C_{k-1}}_{H^{(k)}_{\text{inter}}}
$$

**Standard FLA limitation:** All three matmuls involve the sequence dimension $L$. Since $Q^{(k)}, K^{(k)} \in \mathbb{R}^{L \times d_q}$ and $V^{(k)} \in \mathbb{R}^{L \times d_v}$, the intermediate $S^{(k)} \in \mathbb{R}^{L \times L}$ must fit in SRAM, capping $L \leq \sqrt{\text{SRAM}_{\text{size}}}$.

**TFLA solution:** Tile along $L$ in both dimensions:

$$
H^{(k)}[i \cdot B_{Lhq} : (i+1) \cdot B_{Lhq}, :] = \sum_{j} \underbrace{S^{(k)}_{ij}}_{B_{Lhq} \times B_{Lkv}} \underbrace{V^{(k)}_j}_{B_{Lkv} \times d_v} + \underbrace{\bar{Q}^{(k)}_i}_{B_{Lhq} \times d_q} \underbrace{C_{k-1}}_{d_q \times d_v}
$$

where:
- $B_{Lhq}$ = tile size for output/query sequence dimension
- $B_{Lkv}$ = tile size for key/value sequence dimension
- $B_{d_q}$, $B_{d_{hv}}$ = tile sizes for head dimensions

The inner tiles $S^{(k)}_{ij} \in \mathbb{R}^{B_{Lhq} \times B_{Lkv}}$ are computed and consumed in SRAM without being written to HBM.

**Tiling Strategy (Figure 3 in the paper):**

For the forward pass $H^{(k)}$ kernel:
- **Parallelize** over $B_{Lhq}$ blocks (outer sequence) and $B_{d_{hv}}$ blocks (embedding dimension)
- **Loop** over $B_{Lkv}$ blocks (inner sequence) and $B_{d_q}$ blocks (query dimension)
- **Accumulate** block results within SRAM using block-wise addition

**Key Definitions:**

- $T$ — total sequence length
- $L$ — chunk size (now arbitrarily large, e.g., 128–4096)
- $N_c = \lceil T/L \rceil$ — number of chunks
- $B_{Lhq}$ — tile size for query/output sequence dimension (fits in SRAM)
- $B_{Lkv}$ — tile size for key/value sequence dimension (fits in SRAM)
- $B_{d_q}$, $B_{d_{hv}}$ — tile sizes for head dimensions
- $C_k \in \mathbb{R}^{d_q \times d_v}$ — inter-chunk memory state
- $S^{(k)} \in \mathbb{R}^{L \times L}$ — intra-chunk attention matrix (never fully materialized)

## Complexity

**FLOPs (Appendix F of the paper):**

| Component | FLA (single-level) | TFLA (two-level) |
|-----------|-------------------|------------------|
| Recurrent kernel (states) | $O(N_c \cdot d_q \cdot d_v \cdot L)$ | $O(N_c \cdot d_q \cdot d_v \cdot L)$ |
| Parallel kernel (outputs) | $O(N_c \cdot L^2 \cdot (d_q + d_v))$ | $O(N_c \cdot L^2 \cdot (d_q + d_v))$ |
| **Total FLOPs** | same | same |

The total FLOPs are **identical** — TFLA does not reduce arithmetic work.

**Memory (the real gain):**

| Metric | FLA ($L = 64$) | TFLA ($L = 1024$) |
|--------|---------------|-------------------|
| \# boundary states in HBM | $T / 64$ | $T / 1024$ |
| HBM for states ($d = 256$) | $T/64 \times 256^2 \times 2$ B | $T/1024 \times 256^2 \times 2$ B |
| Memory savings factor | baseline | **16×** fewer states |
| Arithmetic intensity | low | **high** |

**Arithmetic Intensity Analysis (Appendix G):**

The arithmetic intensity $I$ (FLOPs per byte of HBM access) for TFLA:

$$
I_{\text{TFLA}} \propto \frac{L^2 (d_q + d_v)}{L (d_q + d_v + d_{hv}) + d_q \cdot d_{hv}}
$$

For FLA with small $L$: $I \approx L$, which is compute-limited only when $L$ exceeds the machine balance point. Increasing $L$ via TFLA directly increases arithmetic intensity.

**Optimal chunk size (empirical):** $L^* \in [128, 256]$ on H100, scaling proportionally with $\sqrt{d_{hv}}$.

**Wall-clock performance (H100, dim=4096, Figure 5):**

| Method | Seq 8K (Train) | Seq 32K (Train) | Seq 65K (Train) |
|--------|---------------|----------------|----------------|
| FlashAttention 3 | ~8 ms | ~35 ms | ~50 ms |
| Mamba 2 | ~15 ms | ~40 ms | ~55 ms |
| FLA (limit_chunk) | ~10 ms | ~25 ms | ~40 ms |
| **TFLA mLSTMsig** | **~6 ms** | **~18 ms** | **~30 ms** |

TFLA is faster than FlashAttention 3 for sequences $\geq 4K$ and consistently 2× faster than Mamba 2.

## Applicability

- **mLSTM / xLSTM (primary application):** The paper implements TFLA for the mLSTM cell with matrix memory, scalar exponential/sigmoid gating, and normalization. Achieves state-of-the-art kernel throughput on H100.

- **DeltaNet / DeltaProduct:** The chunkwise-parallel formulation of DeltaNet (Yang et al., 2024) has the same structure: inter-chunk state recurrence + intra-chunk parallel matmuls. TFLA's two-level tiling can be applied to DeltaNet's parallel kernel, enabling larger chunk sizes. The UT transform computation (which converts sequential WY recurrence to matmuls) happens at the outer chunk level, while TFLA tiles the intra-chunk attention matmuls at the inner level.

- **GLA (Gated Linear Attention):** Explicitly mentioned as applicable in Appendix A.2 of the paper.

- **RetNet:** Linear attention with exponential decay gates — directly compatible with TFLA's formulation.

- **Any linear RNN with matrix state:** The generic formulation $C_t = f_t C_{t-1} + i_t k_t v_t^\top$ covers all linear attention variants. TFLA applies whenever the chunkwise-parallel decomposition exists.

- **PaTH Attention (potential):** PaTH's blockwise algorithm already tiles along the sequence dimension for cross-block attention. TFLA's inner tiling strategy could be applied to PaTH's intra-block computation for larger block sizes.

## Limitations

- **Does not reduce FLOPs:** TFLA is a pure memory-efficiency and arithmetic-intensity optimization. The total compute is the same as standard FLA. The speedup comes from better hardware utilization (fewer HBM round-trips, higher tensor core occupancy).

- **mLSTM-specific complications:** The exponential input gate requires max-state tracking ($m_t$) for numerical stability, which prevents efficient fusing of loops within TFLA's tiled computation (Appendix C.3). The mLSTMsig variant (sigmoid gate) avoids this issue.

- **DeltaNet adaptation is non-trivial:** DeltaNet's intra-chunk computation involves the UT transform (forward substitution + matmuls), which is more complex than the simple gated attention matrix in mLSTM. Applying TFLA to DeltaNet requires decomposing the UT transform across tiles, which may introduce additional complexity.

- **Quadratic cost persists intra-chunk:** The $O(L^2)$ intra-chunk attention cost remains. Larger $L$ increases FLOPs quadratically but reduces memory linearly, so the optimal $L$ balances these.

- **Backward pass is more complex:** Four separate kernels are needed for the backward pass (gradients for $Q$, $K$, $V$, and gates), each with a different parallelization/loop structure over the tiling dimensions (Table 1).

- **Not yet applied to DeltaNet in practice:** The paper implements TFLA only for mLSTM. Extending to DeltaNet/DeltaProduct with Householder product transitions is an open engineering task.

## Implementation Notes

```python
# TFLA forward pass pseudocode (Algorithm 1 from the paper)
# Adapted for generic linear RNN with scalar gating

def tfla_forward(Q, K, V, f_gates, i_gates, L, B_Lhq, B_Lkv, B_dq, B_dhv):
    """
    Two-level tiled flash linear attention forward pass.

    Args:
        Q: (T, d_q)  - queries
        K: (T, d_q)  - keys (same dim as Q for this formulation)
        V: (T, d_hv) - values
        f_gates: (T,) - forget gate pre-activations
        i_gates: (T,) - input gate pre-activations
        L: int - chunk size (can be very large, e.g., 1024-4096)
        B_Lhq: int - tile size for query sequence dim (fits in SRAM)
        B_Lkv: int - tile size for key/value sequence dim (fits in SRAM)
        B_dq: int - tile size for query head dim
        B_dhv: int - tile size for value head dim

    Returns:
        H: (T, d_hv) - output hidden states
    """
    T, d_q = Q.shape
    d_hv = V.shape[1]
    N_c = T // L  # number of chunks

    # Reshape into chunks
    Q_c = Q.reshape(N_c, L, d_q)
    K_c = K.reshape(N_c, L, d_q)
    V_c = V.reshape(N_c, L, d_hv)

    C = torch.zeros(d_q, d_hv)  # inter-chunk state (in HBM)
    H = torch.empty(N_c, L, d_hv)

    for k in range(N_c):
        # === RECURRENT KERNEL (1st level) ===
        # Compute chunk boundary state C_k from C_{k-1}
        # This materializes C_k in HBM
        C = update_chunk_state(C, K_c[k], V_c[k], f_gates, i_gates, k)

        # === PARALLEL KERNEL (2nd level: TFLA tiling) ===
        # Parallelize over (i, dhv_block) — these are GPU thread blocks
        N_Lhq = L // B_Lhq
        N_dhv = d_hv // B_dhv

        for i in range(N_Lhq):  # PARALLEL (different thread blocks)
            for dhv_b in range(N_dhv):  # PARALLEL (different thread blocks)

                # Accumulator in SRAM (registers)
                h_acc = torch.zeros(B_Lhq, B_dhv)  # in SRAM

                # --- Inter-chunk contribution ---
                # Loop over d_q tiles
                for dq_b in range(d_q // B_dq):  # LOOP (sequential)
                    q_tile = Q_c[k, i*B_Lhq:(i+1)*B_Lhq,
                                   dq_b*B_dq:(dq_b+1)*B_dq]  # B_Lhq × B_dq
                    c_tile = C[dq_b*B_dq:(dq_b+1)*B_dq,
                               dhv_b*B_dhv:(dhv_b+1)*B_dhv]  # B_dq × B_dhv
                    # TENSOR CORE matmul: (B_Lhq × B_dq) @ (B_dq × B_dhv)
                    h_acc += q_tile @ c_tile

                # --- Intra-chunk contribution ---
                # Loop over K/V tiles (causal: j ≤ i for causal mask)
                for j in range(L // B_Lkv):  # LOOP (sequential)
                    # Compute attention tile S_ij in SRAM
                    s_tile = torch.zeros(B_Lhq, B_Lkv)
                    for dq_b in range(d_q // B_dq):  # LOOP
                        q_tile = Q_c[k, i*B_Lhq:(i+1)*B_Lhq,
                                       dq_b*B_dq:(dq_b+1)*B_dq]
                        k_tile = K_c[k, j*B_Lkv:(j+1)*B_Lkv,
                                       dq_b*B_dq:(dq_b+1)*B_dq]
                        # TENSOR CORE: (B_Lhq × B_dq) @ (B_dq × B_Lkv)
                        s_tile += q_tile @ k_tile.T

                    # Apply gate mask D and causal mask
                    s_tile = apply_gate_mask(s_tile, f_gates, i_gates, i, j)

                    # Accumulate S_ij @ V_j
                    for dhv_b2 in range(1):  # already tiled
                        v_tile = V_c[k, j*B_Lkv:(j+1)*B_Lkv,
                                       dhv_b*B_dhv:(dhv_b+1)*B_dhv]
                        # TENSOR CORE: (B_Lhq × B_Lkv) @ (B_Lkv × B_dhv)
                        h_acc += s_tile @ v_tile

                # Write result tile to HBM
                H[k, i*B_Lhq:(i+1)*B_Lhq,
                     dhv_b*B_dhv:(dhv_b+1)*B_dhv] = h_acc

    return H.reshape(T, d_hv)
```

**GPU efficiency analysis:**

1. **Five dimensions of parallelism:** TFLA parallelizes over (a) batch, (b) heads, (c) chunks ($N_c$), (d) query sequence tiles ($L/B_{Lhq}$), and (e) embedding tiles ($d_{hv}/B_{d_{hv}}$) — saturating GPU SMs with independent work.

2. **All dominant operations are matmuls:** $QK^\top$, $S \cdot V$, and $Q \cdot C$ are all matrix-matrix products that map to WGMMA/MMA tensor core instructions.

3. **$S^{(k)}$ never materialized in HBM:** The $L \times L$ attention matrix is computed tile-by-tile in SRAM and immediately consumed, matching FlashAttention's IO-awareness.

4. **Arithmetic intensity scales with $L$:** For embedding dim 4096 on H100, the paper finds optimal $L \in [128, 256]$, at which the kernel transitions from memory-bound to compute-bound.

5. **Code available:** https://github.com/NX-AI/mlstm_kernels — Triton implementation for mLSTM, extensible to other linear RNN variants.

**Application to DeltaNet/DeltaProduct:**

For DeltaNet, the intra-chunk computation has an additional complexity: the UT transform must compute $W = T^{-1}K$ and $U = T^{-1}V$ before the standard linear attention matmuls. TFLA can be applied in two ways:

1. **Outer-level UT + inner-level TFLA:** Compute the UT transform at the outer chunk level (chunk size $L$), producing WY factors $W, U \in \mathbb{R}^{L \times d}$. Then tile the intra-chunk matmuls ($QK^\top$, $S \cdot U$) using TFLA's inner tiling. The UT transform's forward substitution is $O(L^2)$, which becomes more expensive at larger $L$ but can be mitigated by the Neumann-series approximation (trick 157).

2. **Two-level UT + TFLA:** Apply UT transform at a smaller sub-chunk level (e.g., 64), then use TFLA's tiling to combine sub-chunk results at the outer level. This requires a hierarchical WY composition step.

## References

- Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels. NeurIPS 2025. arXiv:2503.14376.
- Yang, S. & Zhang, Y. (2024). FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism. https://github.com/fla-org/flash-linear-attention.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 2024. arXiv:2406.06484.
- Beck, M., Pöppel, K., Spanring, M., et al. (2024). xLSTM: Extended Long Short-Term Memory. NeurIPS 2024. arXiv:2405.04517.
- Dao, T. (2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR 2024.
- Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. arXiv:2407.08691.
- Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024.
