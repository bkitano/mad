# 192: ZeCO All-Scan — Fused Gather + Prefix Sum for Linear Attention SP

**Category**: parallelization
**Gain type**: efficiency
**Source**: Chou, Liu, Zhu, Wan, Li, Chu, Liu, Wu & Ma (2025) — arXiv:2507.01004
**Paper**: [papers/zeco-all-scan-sequence-parallelism.pdf]
**Documented**: 2026-02-15

## Description

LASP-2 (trick 176) achieves sequence-length-independent communication for linear attention by AllGathering the $d \times d$ memory states $M_t$ from all $P$ devices, then computing a prefix sum locally. While this is a major improvement over ring-style P2P (LASP-1), the AllGather still transmits $(P-1) \times d_k \times d_v$ bytes per device — communication volume that **grows linearly with device count $P$**.

ZeCO (Zero Communication Overhead) introduces **All-Scan**, a new collective communication primitive that **fuses the gather and prefix-sum reduction into a single pipelined P2P chain**. Instead of every device receiving all $P$ states and redundantly computing the same prefix sum, each device sends its final local state $S_{pL}$ only to the **next** device ($p+1$), which immediately integrates it into its own cumulative state via the recurrence $S_{(p-1)L+nC} = \hat{\gamma}_{[n]}^\top \mathbf{1} \odot S_{(p-1)L} + S_{[n]}$. The update is further pipelined by splitting the $d_k \times d_v$ state tensor into $K$ blocks along the $d_k$ dimension, allowing device $p+1$ to begin updating as soon as the first block of $S_{pL}$ arrives.

**Key advantages over LASP-2:**
1. **$P\times$ less communication volume**: Each device sends/receives exactly one state $S \in \mathbb{R}^{d_k \times d_v}$ (volume $= d_k \times d_v$), vs. LASP-2's AllGather of $P \times d_k \times d_v$
2. **Pipelined overlap**: The $K$-block partitioning enables fine-grained overlap of All-Scan communication with intra-chunk diagonal attention computation
3. **Negligible extra compute**: Only $N$ elementwise multiplications by cumulative decay $\hat{\gamma}_{[n]}$ and $N$ additions per device — both $O(Nd)$, negligible vs. the $O(Cd^2)$ attention compute
4. **Provably optimal**: ZeCO achieves the theoretical minimum communication volume for any linear attention SP method

The result is **near-linear throughput scaling**: training a 1B-GLA model on 1M tokens across 64 GPUs takes roughly the same time as training on 16K tokens on a single GPU.

## Mathematical Form

**Setup:** Sequence of length $L$ distributed across $P$ devices, each with local length $L/P$. Each device further partitions into $N = L/(PC)$ chunks of size $C$.

**Gated Linear Attention (GLA) recurrence** (the general form ZeCO targets):

$$
S_t = (\alpha_t^\top \mathbf{1}) \odot S_{t-1} + K_t^\top V_t, \quad O_t = Q_t S_t
$$

where $\alpha_t \in (0,1)^{d_k}$ is a per-token decay factor. In chunk-wise form:

$$
S_{[i]} = (\gamma_{[i]}^\top \mathbf{1}) \odot S_{[i-1]} + (K_{[i]} \odot \Gamma_{[i]})^\top V_{[i]}
$$

where $\gamma_{[i]} = \prod_{j=1}^{C} \alpha_{iC+j}$ is the chunk-level cumulative decay, and $\Gamma_{[i]}$ is the token-level decay within the chunk.

**Local state computation (all devices in parallel):**

$$
S_{[n]} = (\gamma_{[n]}^\top \mathbf{1}) \odot S_{[n-1]} + \tilde{K}_{[n]}^\top V_{[n]}, \quad n = 1, \ldots, N
$$

where $\tilde{K}_{[n]} = K_{[n]} \odot \Gamma_{[n]}$ and initial state $S_{[0]} = \mathbf{0}$.

**Cumulative decay vector:**

$$
\hat{\gamma}_{[n]} = \prod_{i=0}^{n} \gamma_{[i]}, \quad n = 1, \ldots, N
$$

**Global state update** (the key correction formula):

$$
S_{(p-1)L + nC} = (\hat{\gamma}_{[n]}^\top \mathbf{1}) \odot S_{(p-1)L} + S_{[n]}
$$

This decomposes the global state into: (1) the attenuated contribution of the previous device's final state $S_{(p-1)L}$, and (2) the purely local contribution $S_{[n]}$. The linearity of the recurrence enables this decomposition.

**All-Scan primitive — Pipelined State Scan (Algorithm 2):**

Partition the state $S_{(p-1)L} \in \mathbb{R}^{d_k \times d_v}$ into $K$ blocks along $d_k$:

$$
S_{(p-1)L} = \left[S_{(p-1)L}^{(1)}, S_{(p-1)L}^{(2)}, \ldots, S_{(p-1)L}^{(K)}\right], \quad S_{(p-1)L}^{(k)} \in \mathbb{R}^{\frac{d_k}{K} \times d_v}
$$

Each block is transmitted pipelined from device $p-1$ to device $p$, and immediately updated:

$$
S_{\text{send}}^{(k)} = S_{\text{local}}^{(k)} + (\hat{\gamma}_{[N]}^{(k)\top} \mathbf{1}) \times S_{\text{recv}}^{(k)}, \quad k = 0, \ldots, K-1
$$

Device $p+1$ begins processing block $k=0$ as soon as it arrives, while blocks $k=1, \ldots, K-1$ are still in transit.

**Output computation (after All-Scan completes):**

$$
O_{[n]}^{\text{inter}} = \tilde{Q}_{[n]} \left( S_{[n-1]} + (\hat{\gamma}_{[n-1]}^\top \mathbf{1}) \odot S_{(p-1)L} \right)
$$

$$
O_{[n]}^{\text{intra}} = P V_{[n]}, \quad P = (\tilde{Q}_{[n]} \tilde{K}_{[n]}^\top) \odot M \in \mathbb{R}^{C \times C}
$$

$$
O_{[n]} = O_{[n]}^{\text{inter}} + O_{[n]}^{\text{intra}}
$$

**Key Definitions:**

- $P$ — number of devices (GPUs)
- $L$ — total sequence length
- $N = L/(PC)$ — number of chunks per device
- $C$ — chunk size
- $d_k, d_v$ — key and value head dimensions
- $S_{[n]} \in \mathbb{R}^{d_k \times d_v}$ — local chunk state (residual from zero initial state)
- $S_{(p-1)L} \in \mathbb{R}^{d_k \times d_v}$ — global initial state for device $p$ (received via All-Scan)
- $\hat{\gamma}_{[n]} \in \mathbb{R}^{d_k}$ — cumulative decay from chunk 0 to chunk $n$
- $K$ — number of pipeline blocks for All-Scan

## Complexity

**Communication volume per device:**

| Method | Volume per device | Scales with $P$? |
|--------|-------------------|------------------|
| LASP-1 (ring P2P) | $P \cdot d_k \cdot d_v$ | Yes (serial) |
| LASP-2 (AllGather) | $P \cdot d_k \cdot d_v$ | Yes (parallel) |
| **ZeCO (All-Scan)** | $d_k \cdot d_v$ | **No** |

ZeCO achieves $P\times$ less communication volume than LASP-2. This is the theoretical minimum: each device must receive at least the state information from its predecessor.

**Communication latency:**

$$
T_{\text{All-Scan}} = \tau(d_k \times d_v) + \frac{(P-1) \cdot \tau(d_k \times d_v)}{K}
$$

As $K \to \infty$, the pipeline overhead vanishes and $T_{\text{All-Scan}} \to \tau(d_k \times d_v)$ — the time to transfer a single state, independent of $P$.

**Total SP runtime:**

$$
T_{\text{ZeCO}}^P(PL) = T_{\text{ideal-SP}}^1(L) - T_{\text{overlapped\_comp}} + \tau(d_k \times d_v) + \epsilon
$$

where $\epsilon$ is negligible extra computation/I/O. Compare with LASP-2:

$$
T_{\text{LASP-2}}^P(PL) = T_{\text{ideal-SP}}^1(L) + P \times \tau(d_k \times d_v)
$$

**Extra computation overhead:**

| Component | Cost | Notes |
|-----------|------|-------|
| Cumulative decay $\hat{\gamma}$ | $O(Nd_k)$ | Elementwise, $\frac{1}{d_v}$ of state tensor |
| Global state update | $O(Nd_k d_v)$ | Elementwise multiply + add per chunk |
| Total extra vs. base | $< 1\%$ | For $L = 8192$, $C = 64$: $N = 128$ |

**Memory:** Same as LASP-2 — $O(d_k \times d_v)$ per cached state. ZeCO stores $N$ local states $S_{[n]}$ plus one global initial state $S_{(p-1)L}$, totaling $(N+1) \times d_k \times d_v$ per device.

## Wall-Clock Performance (H100 80GB GPUs)

**Communication primitive speed (Table 2, 8K seq/GPU, $d=4096$, $H=32$):**

| GPUs | AllGather (LASP-2) | All-Scan (ZeCO) | Speedup |
|------|-------------------|-----------------|---------|
| 8 | 0.375 ms | 0.226 ms | 1.7× |
| 16 | 0.658 ms | 0.297 ms | 2.2× |
| 32 | 1.616 ms | 0.448 ms | 3.6× |
| 64 | 2.543 ms | 0.739 ms | 3.4× |
| 128 | 4.350 ms | 1.272 ms | 3.4× |
| 256 | 8.514 ms | 2.165 ms | **3.9×** |

**SP Algorithm runtime (1B-GLA, 16K seq/GPU, $H=16$):**

| GPUs | LASP-1 | LASP-2 | ZeCO | GLA baseline (DP) |
|------|--------|--------|------|-------------------|
| 8 | 22.59 ms | 19.39 ms | 7.32 ms | 6.39 ms |
| 32 | 39.03 ms | 25.72 ms | 7.65 ms | 6.44 ms |
| 128 | 113.71 ms | 35.72 ms | 9.88 ms | 6.12 ms |

ZeCO is only ~3 ms slower than the DP baseline at 128 GPUs, while LASP-2 is 30 ms slower.

**Model throughput (1B-GLA, 32K seq/GPU, tokens/sec per GPU):**

| GPUs | LASP-1 | LASP-2 | ZeCO | DP baseline |
|------|--------|--------|------|-------------|
| 8 | 27,014 | 42,946 | 47,369 | 49,633 |
| 64 | 12,268 | 37,485 | 44,468 | 48,230 |
| 128 | — | 33,327 | 43,278 | 47,848 |
| 256 | — | 25,402 | 40,967 | 46,588 |

At 256 GPUs: ZeCO retains **88%** of DP throughput; LASP-2 retains only 55%. ZeCO achieves **1.6×** higher throughput than LASP-2 at 256 GPUs.

## Applicability

- **All linear attention variants with diagonal decay**: GLA, RetNet/Retention, Mamba-2/SSD, Lightning Attention — any model where the inter-chunk recurrence has the form $S_{[i]} = \gamma \odot S_{[i-1]} + \Delta S$. The decay factor $\gamma$ is essential for the decomposition property.

- **Direct successor to LASP-2**: Drop-in replacement for LASP-2's AllGather + prefix-sum pattern. Uses the same chunkwise computation structure; only the inter-device communication primitive changes.

- **Complements TFLA (trick 158)**: ZeCO handles inter-device communication optimally; TFLA handles intra-device tiling. Together they provide end-to-end optimization.

- **Scalable to 8M+ tokens**: Near-linear scaling demonstrated from 8 to 256 GPUs. The $P$-independent communication volume means adding more GPUs does not increase communication cost.

## Limitations

- **Sequential pipeline dependency**: All-Scan is fundamentally a P2P chain (device $p$ must wait for device $p-1$). Unlike LASP-2's AllGather which completes in $O(\log P)$ network hops, All-Scan has $O(P)$ latency steps. The $K$-block pipelining amortizes this, but at very large $P$ (>>256) the pipeline latency could become significant.

- **Requires diagonal decay structure**: The decomposition $S_{(p-1)L+nC} = \hat{\gamma}_{[n]} \odot S_{(p-1)L} + S_{[n]}$ requires the recurrence to have multiplicative decay on the state. Pure linear attention without decay ($\gamma = 1$) works but loses the decay attenuation. Non-diagonal state transitions (e.g., full matrix $A$ in $S_{t+1} = A S_t + \ldots$) would require transmitting a matrix, not just a vector decay.

- **Not applicable to softmax attention**: ZeCO is specific to linear attention models. For hybrid models (linear + softmax), the softmax layers still need Ring Attention, DeepSpeed-Ulysses, or LASP-2H's AllGather-on-KV approach.

- **P2P communication pattern**: All-Scan uses single-direction P2P (device $p \to p+1$), which on NVSwitch topologies underutilizes the full-mesh bandwidth (only $1/(n-1)$ link utilization). TASP's multi-ring approach (trick 193) could potentially be combined with All-Scan to improve link utilization.

- **Backward pass requires recomputation**: In the backward pass, local states $S_{[n]}$ must be recomputed from the cached global initial state $S_{(p-1)L}$. However, no additional All-Scan communication is needed for the backward recomputation since the global state was cached during forward.

## Implementation Notes

```python
# ZeCO Forward Pass with All-Scan — Algorithm 1
# Each GPU p executes this in parallel

def zeco_forward(Q, K, V, G, C, P, rank):
    """
    Q, K: (L, d_k), V: (L, d_v), G: (L, d_k) — decay factors
    C: chunk size, P: num devices, rank: device index p
    Returns: O: (L, d_v) — output, cached states for backward
    """
    N = L // C  # chunks per device

    # Phase 1: Local state computation (parallel across chunks, no communication)
    S = zeros(d_k, d_v)  # local cumulative state
    gamma_hat = ones(d_k)  # cumulative decay
    local_states = []
    gamma_hats = []

    for n in range(N):
        gamma_n = compute_chunk_decay(G, n, C)  # prod of per-token decays
        Gamma_n = compute_token_decay(G, n, C)   # per-token cumulative decay
        K_tilde = K[n] * Gamma_n                  # decay-scaled keys

        gamma_hat = gamma_hat * gamma_n
        S = (gamma_n[:, None] * ones(1, d_v)) * S + K_tilde.T @ V[n]

        local_states.append(S.clone())
        gamma_hats.append(gamma_hat.clone())

    # Phase 2: All-Scan communication (overlapped with Phase 3 on separate stream)
    # Each device sends its final state S_[N] to device p+1
    # and receives S_(p-1)L from device p-1
    # Pipelined: state split into K blocks, each sent/received incrementally
    S_prev, S_next = all_scan(
        P, rank,
        S_local=local_states[-1],   # S_[N] — this device's final local state
        gamma_hat=gamma_hats[-1],   # cumulative decay for merging
        K_blocks=K_PIPELINE_BLOCKS
    )
    # S_prev = S_{(p-1)L}: the global state at the boundary before this device

    # Phase 3: Intra-chunk diagonal attention (overlapped with Phase 2)
    # This computes P_{[n]} = (Q_tilde @ K_tilde.T) * causal_mask for each chunk
    # Can run on a separate CUDA stream since it doesn't depend on S_prev
    P_matrices = []
    for n in range(N):
        Q_tilde = Q[n] * compute_token_decay(G, n, C)
        K_tilde = K[n] * compute_token_decay(G, n, C)
        P_n = (Q_tilde @ K_tilde.T) * causal_mask(C, C)
        P_matrices.append(P_n)

    # Stream barrier: wait for both Phase 2 and Phase 3

    # Phase 4: Final output computation (uses both S_prev and P matrices)
    for n in range(N):
        # Inter-chunk: contribution from all previous chunks + previous devices
        S_global = local_states[n-1] + gamma_hats[n-1][:, None] * S_prev \
                   if n > 0 else gamma_hats[0][:, None] * S_prev
        O_inter = Q_tilde_n @ S_global  # (C, d_v)

        # Intra-chunk: local causal attention
        O_intra = P_matrices[n] @ V[n]  # (C, d_v)

        O[n] = O_inter + O_intra

    return O

def all_scan(P, rank, S_local, gamma_hat, K_blocks):
    """
    All-Scan: Pipelined receive-scan-send pattern.
    Each device receives predecessor's state, applies decay, adds local, sends.

    Communication volume per device: d_k * d_v (ONE state, not P states!)
    """
    send_rank = (rank + 1) % P
    recv_rank = (rank - 1) % P

    # Split state into K blocks for pipelining
    S_send = S_local.chunk(K_blocks, dim=0)
    gamma_blocks = gamma_hat.chunk(K_blocks, dim=0)

    if rank != 0:
        S_recv = recv_from(recv_rank)  # receive S_{(p-1)L} in K blocks

    for k in range(K_blocks):
        if rank == 0:
            # First device: just send local state
            send_to(S_send[k], send_rank)
        elif rank < P - 1:
            # Middle devices: update and forward
            S_send[k] = S_send[k] + gamma_blocks[k][:, None] * S_recv[k]
            send_to(S_send[k], send_rank)
        else:
            # Last device: update only (no send)
            S_send[k] = S_send[k] + gamma_blocks[k][:, None] * S_recv[k]

    S_prev = S_recv if rank != 0 else zeros_like(S_local)
    S_next = cat(S_send)  # = S_{pL} (this device's global final state)

    return S_prev, S_next

# Key GPU efficiency properties:
# 1. Communication volume = d_k * d_v per device (P-independent!)
# 2. All-Scan overlaps with diagonal attention on separate CUDA stream
# 3. K-block pipelining hides pipeline latency behind block-level updates
# 4. All dominant ops are matmuls (tensor core friendly)
# 5. Extra compute overhead < 1% (elementwise gamma scaling)
# 6. Backward pass reuses cached S_prev — no additional All-Scan needed
```

**GPU efficiency analysis:**

1. **Communication volume**: At 256 GPUs with $d=4096$, LASP-2's AllGather transfers $256 \times d_k \times d_v$ per device. ZeCO transfers just $d_k \times d_v$ — a 256× reduction. For $d_k = d_v = 128$ (per head), this is 32KB vs 8MB in FP16.

2. **Overlap scheduling**: ZeCO runs two CUDA streams in parallel: (a) All-Scan communication + global state update, (b) diagonal attention computation $P_{[n]} = (\tilde{Q}_{[n]}\tilde{K}_{[n]}^\top) \odot M$. Since diagonal attention is the most expensive local operation ($O(C^2 d)$), and it has no dependency on inter-device states, the overlap is perfect.

3. **Pipeline amortization**: The $K$-block splitting ensures that the All-Scan pipeline boundary overhead $\frac{(P-1) \cdot \tau(d_k \times d_v)}{K}$ vanishes for large $K$. In practice, $K = 4$–$8$ blocks suffice.

4. **Tensor core utilization**: All matmuls ($\tilde{Q}\tilde{K}^\top$, $PV$, $QS$, $K^\top V$) map directly to tensor cores. The only non-matmul operations are elementwise decay scaling ($\gamma \odot S$), which are memory-bound but negligible in cost.

## References

- Chou, Y., Liu, Z., Zhu, R., Wan, X., Li, T., Chu, C., Liu, Q., Wu, J., & Ma, Z. (2025). ZeCO: Zero Communication Overhead Sequence Parallelism for Linear Attention. arXiv:2507.01004.
- Sun, W., Lan, D., Zhong, Y., Qu, X., & Cheng, Y. (2025). LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid. arXiv:2502.07563.
- Sun, W., Qin, Z., Li, D., Shen, X., Qiao, Y., & Zhong, Y. (2024). Linear Attention Sequence Parallelism. arXiv:2404.02882.
- Yang, S., Wang, B., Shen, Y., et al. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. arXiv:2312.06635.
- Yang, S. & Zhang, Y. (2024). FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism. github.com/fla-org/flash-linear-attention.
