# 176: LASP-2 — AllGather-Based Sequence Parallelism for Linear Attention

**Category**: parallelization
**Gain type**: efficiency
**Source**: Sun, Lan, Zhong, Qu & Cheng (2025) — arXiv:2502.07563
**Paper**: [papers/lasp2-linear-attention-sequence-parallelism.pdf]
**Documented**: 2026-02-15

## Description

When training linear attention models on very long sequences (100K–4M tokens), single-GPU memory is insufficient and the sequence must be distributed across multiple GPUs. Standard sequence parallelism (SP) methods like Megatron-SP or Ring Attention are designed for softmax attention and do not exploit linear attention's key structural advantage: the **right-product kernel trick** $(QK^\top)V = Q(K^\top V)$, which compresses all context into a fixed-size $d \times d$ memory state $M$.

The predecessor LASP-1 uses a **ring-style P2P** communication pattern to pass memory states sequentially between GPUs. This creates $2(W-1)$ communication steps per iteration (where $W$ is the number of devices), each transferring the $d \times d$ state. The many small P2P operations are hard to overlap with computation and create a sequential bottleneck.

**LASP-2** replaces the ring with a **single AllGather** collective communication per forward/backward pass. Each GPU computes its local memory state $M_t = K_t^\top V_t$, then all states are gathered simultaneously via `AllGather`. Since $M_t \in \mathbb{R}^{d \times d}$ regardless of sequence or chunk length, the communication volume is independent of the sequence length — it scales only with the number of devices $W$ and model dimension $d$.

Key advantages over LASP-1:
1. **Fewer communication steps**: 2 AllGather ops per iteration vs. $2(W-1)$ P2P ops
2. **Better overlap**: A single AllGather can overlap with intra-chunk computation
3. **Higher computation parallelism**: After gathering all states, the inter-chunk output $O_{t,\text{inter}} = Q_t M_{1:t-1}$ can be computed in parallel across chunks via a prefix sum on the gathered states
4. **Unified hybrid support**: LASP-2H extends the same AllGather pattern to standard softmax attention layers, enabling efficient SP for hybrid linear+softmax architectures

## Mathematical Form

**Setup:** Sequence $X$ of length $N$ is divided into $T$ chunks, distributed across $W$ GPUs (one chunk per GPU when $T = W$).

**Per-chunk local computation (parallel across all GPUs):**

$$
Q_t, K_t, V_t = X_t W_Q, \; X_t W_K, \; X_t W_V
$$

$$
M_t = K_t^\top V_t \in \mathbb{R}^{d \times d}
$$

**Communication (single AllGather):**

$$
[M_t]_1^T = \texttt{AllGather}([M_t]_1^T)
$$

After this, every GPU holds all $T$ memory states. The communication volume per device is $BHd^2$ bytes (batch $\times$ heads $\times d \times d$), independent of sequence length.

**Prefix sum of states (computed recursively on each GPU):**

$$
M_{1:T} = \texttt{Sum}([M_t]_1^T), \quad \text{i.e.,} \quad M_{1:t-1} = M_{1:t-2} + M_{t-1}
$$

This accumulation is done recursively: $M_{1:t} = M_{1:t-1} + M_t$, and cached in HBM for the backward pass.

**Output computation:**

$$
O_t = Q_t M_{1:T} \quad \text{(without masking)}
$$

**With causal masking (autoregressive):**

$$
O_{t,\text{intra}} = \left[(Q_t K_t^\top) \odot \Psi\right] V_t \quad \text{(local quadratic attention)}
$$

$$
O_{t,\text{inter}} = Q_t M_{1:t-1} \quad \text{(contribution from all previous chunks)}
$$

$$
O_t = O_{t,\text{intra}} + O_{t,\text{inter}}
$$

where $\Psi$ is the causal mask ($\Psi_{ij} = 1$ if $i \geq j$, $-\infty$ otherwise).

The prefix sum $M_{1:t-1} = \texttt{PrefixSum}([M_t]_1^{t-1})$ is computed after the AllGather.

**Backward pass (single AllGather on gradient states):**

$$
dM_t = Q_t^\top \, dO_t \in \mathbb{R}^{d \times d}
$$

$$
[dM_t]_1^T = \texttt{AllGather}([dM_t]_1^T)
$$

$$
dM_{1:T} = \texttt{Sum}([dM_t]_1^T) \quad \text{(or SuffixSum for causal)}
$$

$$
dQ_t = dO_t \, M_{1:T}^\top, \quad dK_t = V_t \, dM_{1:T}, \quad dV_t = K_t \, dM_{1:T}
$$

**Key Definitions:**

- $W$ — distributed world size (number of GPUs)
- $T$ — number of sequence chunks (often $T = W$)
- $C = N / T$ — chunk length (local sequence per GPU)
- $d$ — hidden / head dimension
- $M_t \in \mathbb{R}^{d \times d}$ — per-chunk memory state (KV activation)
- $M_{1:t} \in \mathbb{R}^{d \times d}$ — cumulative state from chunks 1 to $t$

## Complexity

**Communication cost per iteration:**

| Method | Steps | Volume per step | Total volume |
|--------|-------|-----------------|--------------|
| LASP-1 (ring P2P) | $2(W-1)$ | $BHd^2$ | $2(W-1) \cdot IBHd^2$ |
| LASP-2 (AllGather) | $2$ | $BHd^2$ | $2I \cdot BHd^2$ |
| Ring Attention | $2(W-1)$ | $BH \cdot C \cdot d$ | $2(W-1) \cdot BHCd$ |

where $I$ = number of transformer layers, $B$ = batch size, $H$ = number of heads.

LASP-2 reduces total communication by a factor of up to $W - 1$ compared to LASP-1. Ring Attention has per-step volume proportional to chunk length $C$ (sending K/V), while both LASP methods send only the $d \times d$ state.

**Computation cost (per device):**

| Component | Cost |
|-----------|------|
| Local Q, K, V projection | $O(C \cdot d^2)$ |
| Local state $M_t = K_t^\top V_t$ | $O(C \cdot d^2)$ |
| Prefix sum on $M$ | $O(W \cdot d^2)$ |
| Intra-chunk attention $O_{t,\text{intra}}$ | $O(C^2 \cdot d)$ |
| Inter-chunk output $O_{t,\text{inter}}$ | $O(C \cdot d^2)$ |

**Memory:** $O(d^2)$ per cached state, $W$ states total = $O(W \cdot d^2)$ for the prefix sum cache (independent of sequence length).

**Wall-clock performance (A100 GPUs, Linear-Llama3-1B, 64 GPUs):**

| Method | Seq 512K | Seq 1024K | Seq 2048K |
|--------|---------|----------|----------|
| Megatron-SP | ~2.5 Ktok/s | ~1.2 Ktok/s | OOM |
| Ring Attention | ~4.0 Ktok/s | ~3.0 Ktok/s | ~2.5 Ktok/s |
| LASP-1 | ~5.5 Ktok/s | ~4.5 Ktok/s | ~3.5 Ktok/s |
| **LASP-2** | **~6.5 Ktok/s** | **~5.5 Ktok/s** | **~4.5 Ktok/s** |

LASP-2 achieves 36.6% speedup over Ring Attention and 15.2% over LASP-1 at 2048K sequence length.

**Scalability:** Scales to 4096K tokens on 128 GPUs (8 DGX-A100 nodes). Memory per GPU scales linearly with $N/W$ (local chunk), not total sequence length.

## Applicability

- **All linear attention variants:** Basic linear attention, Lightning Attention, Retention/RetNet, GLA, Based, Rebased — all tested with LASP-2 and show competitive convergence.

- **Hybrid linear+softmax models (LASP-2H):** Extends to models mixing linear and standard attention layers (e.g., 1/4 hybrid with every 4th layer as softmax). Uses AllGather on K/V for standard attention layers.

- **Combines with TFLA on each GPU:** LASP-2 handles inter-GPU distribution while TFLA (trick 158) handles intra-GPU kernel optimization. These are complementary and can be stacked: LASP-2 distributes chunks across GPUs, and TFLA's two-level tiling optimizes the local per-chunk computation on each GPU.

- **GQA (Grouped Query Attention):** Particularly beneficial because the AllGather latency for standard attention layers is hidden when $K_t, V_t$ tensors are small relative to $Q_t$ (as in GQA).

- **Compatible with FSDP/ZeRO:** Orthogonal to data parallelism and parameter sharding — LASP-2 handles sequence dimension, FSDP handles model parameters.

## Limitations

- **Requires AllGather collectives:** AllGather has higher latency than P2P for small messages on fast interconnects (NVLink). LASP-2's advantage grows with slower interconnects or larger $d$.

- **Communication still $O(W)$:** While volume per step is constant ($d^2$), the AllGather latency scales with $W$ due to the collective nature. For very large $W$ with fast intra-node links, LASP-1's P2P may be competitive.

- **Quadratic intra-chunk cost remains:** Each GPU still computes $O(C^2 d)$ for intra-chunk attention. For very long local chunks, this dominates. Can be mitigated by combining with TFLA's tiling (trick 158).

- **Linear attention convergence gap:** The paper uses unnormalized linear attention ($\phi = \text{identity}$), which may underperform softmax attention on recall-intensive tasks. The hybrid 1/4 model partially addresses this.

- **Causal prefix sum is sequential:** The $\texttt{PrefixSum}([M_t]_1^{t-1})$ after AllGather is sequential over $W$ states. For large $W$, this introduces $O(W \cdot d^2)$ sequential work, though it is small relative to the $O(C^2 d)$ intra-chunk computation.

## Implementation Notes

```python
# LASP-2 Forward Pass (without masking) — Algorithm 1
# Each GPU t executes this in parallel

def lasp2_forward_no_mask(X_local, W_Q, W_K, W_V, world_size, rank):
    """
    LASP-2 forward pass for linear attention (no causal mask).
    X_local: (C, d) — local chunk on this GPU
    Returns: O_local: (C, d) — local output
    """
    # Step 1: Local projection (parallel, all GPUs)
    Q = X_local @ W_Q  # (C, d)
    K = X_local @ W_K  # (C, d)
    V = X_local @ W_V  # (C, d)

    # Step 2: Local memory state (parallel, all GPUs)
    M_local = K.T @ V   # (d, d) — the key insight: fixed-size state

    # Step 3: Single AllGather — O(d^2) per device, independent of C
    M_all = all_gather(M_local)  # list of T memory states, each (d, d)

    # Step 4: Sum all states (can be done as prefix sum for causal)
    M_total = sum(M_all)  # (d, d)

    # Step 5: Compute output
    O_local = Q @ M_total  # (C, d) — single matmul

    return O_local

def lasp2_forward_causal(X_local, W_Q, W_K, W_V, world_size, rank):
    """
    LASP-2 forward pass with causal masking — Algorithm 2
    """
    Q = X_local @ W_Q
    K = X_local @ W_K
    V = X_local @ W_V

    M_local = K.T @ V  # (d, d)

    # AllGather all memory states
    M_all = all_gather(M_local)  # [M_1, ..., M_T]

    # Prefix sum: M_{1:t-1} for this rank's chunk
    M_prefix = sum(M_all[:rank])  # (d, d) — states from chunks before this one

    # Intra-chunk: local quadratic attention with causal mask
    # This is the O(C^2 d) part — can use TFLA tiling here!
    Psi = causal_mask(C, C)
    O_intra = ((Q @ K.T) * Psi) @ V  # (C, d)

    # Inter-chunk: contribution from all previous chunks
    O_inter = Q @ M_prefix  # (C, d) — single matmul

    O_local = O_intra + O_inter

    # Cache M_prefix for backward pass (avoids recomputation)
    return O_local

# Key GPU efficiency properties:
# 1. Only 1 AllGather per fwd + 1 per bwd (2 total)
# 2. AllGather volume = B * H * d^2 bytes (independent of seq length!)
# 3. AllGather overlaps with O_intra computation (concurrent streams)
# 4. All dominant ops are matmuls (tensor core friendly)
# 5. Prefix sum is O(W * d^2) — negligible vs O(C^2 * d) intra-chunk
```

**GPU efficiency analysis:**

1. **Communication-computation overlap:** The AllGather on $M_t$ (line 7 in Algorithm 2) can run concurrently with $O_{t,\text{intra}}$ computation (line 8), since they are independent. This hides communication latency behind the quadratic intra-chunk matmuls.

2. **Sequence-length-independent communication:** The $d \times d$ state tensor is tiny compared to the $C \times d$ K/V tensors that Ring Attention must transfer. For a 1B model with $d = 2048$, $H = 16$: each state is $\sim$1 MB (FP16), vs. $\sim C/d$ MB for K/V at each step. At $C = 128K$, K/V is 64$\times$ larger.

3. **AllGather is well-optimized:** NCCL provides highly optimized AllGather with bandwidth-optimal algorithms. A single large collective is more efficient than many small P2P transfers.

4. **Backward pass mirrors forward:** The backward pass requires one AllGather on $dM_t = Q_t^\top dO_t$ (same $d \times d$ size), then computes $dQ$, $dK$, $dV$ as matmuls — all tensor-core friendly.

## References

- Sun, W., Lan, D., Zhong, Y., Qu, X., & Cheng, Y. (2025). LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid. arXiv:2502.07563.
- Sun, W., Qin, Z., Li, D., Shen, X., Qiao, Y., & Zhong, Y. (2024). Linear Attention Sequence Parallelism. arXiv:2404.02882.
- Liu, H., Zaharia, M., & Abbeel, P. (2023). Ring Attention with Blockwise Transformers for Near-Infinite Context. arXiv:2310.01889.
- Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024.
- Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). Tiled Flash Linear Attention. NeurIPS 2025. arXiv:2503.14376.
