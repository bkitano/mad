# 204: BurstAttention — Double-Buffer GAO+LAO Distributed Attention

**Category**: parallelization
**Gain type**: efficiency
**Source**: Sun, Zhao, Han, Yang, Liu, Shi & Sun (2024) — arXiv:2403.09347
**Paper**: [papers/burst-attention-distributed.pdf]
**Documented**: 2026-02-15

## Description

BurstAttention is a distributed attention framework that bridges the gap between single-GPU memory optimization (FlashAttention) and multi-GPU sequence parallelism (Ring Attention) — two approaches that were previously **incompatible**. Standard Ring Attention separates K and V into independent communication rounds, which prevents fusing the attention computation into a FlashAttention-style IO-aware kernel. BurstAttention fixes this by co-designing the inter-device communication pattern (GAO) and intra-device tiling (LAO) so that both optimizations compose cleanly.

The framework has three key components:

1. **Global Attention Optimization (GAO):** Devices are arranged in a ring. Each round, K and V partitions are shifted together (not separately). The local attention result $O_{i,j}$ is accumulated into the global output $O_i$ using **online softmax** — maintaining running row-wise max $m_i$ and sum $l_i$ statistics. This eliminates storage of the $O(N^2/G^2)$ intermediate score matrices $S_{i,j}$ and $P_{i,j}$, and enables a cheaper backward pass that recomputes S and P from cached log-sum-exp values.

2. **Local Attention Optimization (LAO):** Within each GPU, the local Q, K, V partitions are further tiled into blocks of size $M/(4d)$ (where $M$ is SRAM capacity). Tiles are loaded into SRAM, and $S_{i,j}$, $P_{i,j}$, $O_{i,j}$ are computed entirely in fast on-chip memory — identical to FlashAttention when running on a single device. This is the key architectural insight: because GAO passes K and V together, the local computation at each ring step is a standard attention between $Q_i$ and $(K_j, V_j)$, which can be tiled exactly like FlashAttention.

3. **Double-buffer communication-computation overlap:** Each device maintains two buffers. One feeds the current LAO computation while the other asynchronously receives the next K/V partition via P2P on a separate CUDA stream. Buffers swap each round, achieving zero-stall pipelining.

## Mathematical Form

**Setup:** Sequence of length $N$ distributed across $G$ devices. Device $i$ holds $Q_i, K_i, V_i \in \mathbb{R}^{N/G \times d}$.

**Local attention at each ring step (device $i$ receives $K_j, V_j$):**

$$
S_{i,j} = \frac{Q_i K_j^T}{\sqrt{d}}, \quad P_{i,j} = \text{softmax}(S_{i,j}), \quad O_{i,j} = P_{i,j} V_j
$$

**Online softmax accumulation (GAO forward — Algorithm 1):**

After computing local $O_{i,j}$, accumulate into running output:

$$
m_{i,j} = \text{rowmax}(S_{i,j})
$$

$$
l_{i,j} = \text{rowsum}(\exp(S_{i,j} - m_{i,j}))
$$

$$
m_{\text{new}} \leftarrow \max\{m_i, \; m_{i,j}\}
$$

$$
l_i \leftarrow e^{m_i - m_{\text{new}}} l_i + e^{m_{i,j} - m_{\text{new}}} l_{i,j}
$$

$$
O_i \leftarrow e^{m_i - m_{\text{new}}} O_i + e^{m_{i,j} - m_{\text{new}}} O_{i,j}
$$

$$
m_i \leftarrow m_{\text{new}}
$$

After all $G$ ring steps:

$$
O_i = \text{diag}(l_i)^{-1} O_i, \quad lse_i = m_i + \log l_i
$$

**Key Definitions:**

- $G$ — number of devices (GPUs)
- $N$ — total sequence length
- $N/G$ — local sequence length per device
- $d$ — head dimension
- $Z$ — number of attention heads per device
- $m_i \in \mathbb{R}^{N/G}$ — running row-wise max of attention scores
- $l_i \in \mathbb{R}^{N/G}$ — running row-wise sum of exponentiated scores
- $lse_i \in \mathbb{R}^{N/G}$ — log-sum-exp statistics (cached for backward)
- $M$ — SRAM capacity (bytes)

**LAO tile size:** Each Q, K, V block has sequence length $M/(4d)$, ensuring the tile of $S_{i,j}$ fits entirely in SRAM.

**Backward pass (Algorithm 2):** Communicates $Q_j, dQ_j, dO_j, D_j, lse_j$ in the ring (5 tensors). Recomputes $S_{j,i}$ and $P_{j,i}$ from $lse_j$ instead of storing them:

$$
S_{j,i} = Q_j K_i^T, \quad P_{j,i} = \exp(S_{j,i} - lse_j)
$$

$$
dV_i \mathrel{+}= P_{j,i}^T \, dO_j
$$

$$
dS_{j,i} = P_{j,i} \circ (dP_{j,i} - D_j)
$$

$$
dK_i \mathrel{+}= dS_{j,i}^T Q_j, \quad dQ_j \mathrel{+}= dS_{j,i} K_i
$$

## Complexity

**Communication overhead comparison (one Transformer block):**

| Method | Forward | Backward | Total |
|--------|---------|----------|-------|
| Ring Attention | $\Theta(2BZNd)$ | $\Theta(6BZNd)$ | $\Theta(8BZNd)$ |
| Tensor Parallelism (Megatron-V3) | $\Theta(4BZNd)$ | $\Theta(4BZNd)$ | $\Theta(8BZNd)$ |
| **BurstAttention** | $\Theta(2BZNd)$ | $\Theta(3BZNd + 2BNZ/G)$ | $\approx \Theta(5BZNd)$ |

BurstAttention reduces backward communication by ~2x vs Ring Attention by not communicating S and P (recomputed from $lse$).

**Memory overhead (activation memory per device):**

| Method | Activation Memory |
|--------|------------------|
| Ring Attention (no FlashATT) | $4\frac{BZNd}{G} + \frac{BZN^2}{G} + \frac{BNH}{G}$ |
| Tensor Parallelism + FlashATT | $4\frac{BZNd}{G} + \frac{BZN^2}{(M/4d)^2 G} + \frac{BNH}{G}$ |
| **BurstAttention (GAO+LAO)** | $4\frac{BZNd}{G} + \frac{BZN^2}{(M/4d)^2 G^2} + \frac{BNH}{G}$ |

The quadratic term shrinks by $G^2$ in BurstAttention due to both inter-device splitting (GAO) and intra-device tiling (LAO).

**I/O complexity (HBM accesses per device):**

$$
\Theta\left(\frac{BZN^2}{(M/d) \cdot G}\right)
$$

vs $\Theta\left(\frac{BZN^2}{G}\right)$ for Ring Attention without FlashAttention.

## Applicability

- **Softmax attention in long-context LLMs:** Directly applicable to GPT/LLaMA-style models. Tested on LLaMA-2-7B and LLaMA-2-13B at sequences up to 262K tokens.

- **Hybrid linear+softmax architectures (LASP-2H):** BurstAttention handles the softmax attention layers while LASP-2 (trick 176) handles linear attention layers via AllGather. The two approaches are complementary and composable.

- **Inference (first-token latency):** Particularly effective for long-context inference where the first token latency is dominated by the attention encoding pass. Achieves the lowest first-token latency across all methods at 262K tokens.

- **Compatible with ZeRO/FSDP:** Orthogonal to parameter sharding — combining BurstAttention+ZeRO achieves memory comparable to Megatron-V3 with superior speed.

- **Sparse attention compatible:** Because the ring communication pattern naturally partitions Q vs K/V, sparse attention methods that skip certain (Q, K/V) pairs can skip entire ring communication rounds.

## Limitations

- **Softmax attention only:** The online softmax accumulation (GAO) is specific to softmax attention. Does not help linear attention, where LASP-2's AllGather approach is superior.

- **P2P ring communication:** Still requires $G-1$ ring steps, each with P2P latency. The volume per step is $O(BZNd/G)$ — proportional to chunk length, unlike LASP-2 which sends only $d \times d$ states.

- **Double buffering requires 2x KV memory:** Each device must hold two sets of K/V buffers simultaneously (one computing, one receiving). This doubles the activation memory for the KV cache during the ring phase.

- **Backward pass sends 5 tensors:** The backward ring communicates Q, dQ, dO, D, lse per step (vs 2 tensors in forward), making backward communication heavier per step (though total volume is still reduced vs Ring Attention due to S/P recomputation).

- **Communication not fully hidden at short sequences:** When the local attention computation ($O(N^2/G^2)$ FLOPs) is small relative to P2P transfer time, the overlap becomes incomplete. The paper notes the communication bottleneck is the primary factor at shorter sequences.

## Implementation Notes

```python
# BurstAttention Forward with Double-Buffer (Algorithm 3)
# Each GPU i executes this

def burst_attention_forward(Q_i, K_i, V_i, world_size, rank):
    """
    Q_i, K_i, V_i: (N/G, d) — local partitions on device i
    Returns: O_i: (N/G, d), lse_i: (N/G,)
    """
    G = world_size
    C = Q_i.shape[0]  # N/G

    # Initialize accumulators
    O_i = zeros(C, d)
    l_i = zeros(C)
    m_i = full(C, -inf)

    # Double buffers: K_buf/V_buf for async recv
    K_buf, V_buf = K_i.clone(), V_i.clone()

    for j in range(G):
        if j != 0:
            # Wait for async recv from previous step
            K_j, V_j = K_buf, V_buf  # swap from buffer
        else:
            K_j, V_j = K_i, V_i

        # Launch async P2P for NEXT round (on separate CUDA stream)
        if j < G - 1:
            async_send(K_j, V_j, dst=next_rank)
            async_recv(K_buf, V_buf, src=prev_rank)

        # === LAO: FlashAttention-style tiled computation ===
        # Tile Q_i, K_j, V_j into blocks of size M/(4d)
        # Compute S_{i,j}, P_{i,j}, O_{i,j} entirely in SRAM
        S_ij = Q_i @ K_j.T / sqrt(d)
        m_ij = S_ij.max(dim=-1)      # row-wise max
        P_ij = exp(S_ij - m_ij)
        l_ij = P_ij.sum(dim=-1)      # row-wise sum
        O_ij = P_ij @ V_j

        # === GAO: Online softmax accumulation ===
        m_new = max(m_i, m_ij)
        l_i = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij
        O_i = exp(m_i - m_new) * O_i + exp(m_ij - m_new) * O_ij
        m_i = m_new

    # Final normalization
    O_i = diag(1/l_i) @ O_i
    lse_i = m_i + log(l_i)

    return O_i, lse_i  # cache lse_i for backward

# Key GPU efficiency properties:
# 1. K and V travel together — enables FlashAttention tiling (LAO)
# 2. Online softmax avoids storing N/G x N/G score matrices
# 3. Double-buffer: P2P recv overlaps with LAO compute on separate stream
# 4. Backward recomputes S,P from lse — halves backward communication
# 5. All dominant ops are matmuls (tensor core friendly)
# 6. SRAM tile size M/(4d) ensures entire tile fits in shared memory
```

**GPU efficiency analysis:**

1. **Communication-computation overlap:** The P2P send/recv on buffer stream runs concurrently with FlashAttention-style compute on the main stream. At 128K+ sequences, compute time dominates and communication is fully hidden.

2. **FlashAttention integration (key differentiator):** Ring Attention sends K and V in separate rounds, breaking the $(Q, K, V) \rightarrow O$ fusion boundary. BurstAttention sends K and V together, so each ring step performs a standard attention that can be fused into a single FlashAttention kernel. This achieves the same HBM access pattern as single-GPU FlashAttention.

3. **Backward efficiency:** By caching only the scalar $lse_i$ per row (not the full $S, P$ matrices), the backward pass recomputes scores on-the-fly. This saves $O(N^2/G^2)$ memory and reduces backward communication from $\Theta(6BZNd)$ to $\Theta(3BZNd + 2BNZ/G)$.

4. **Measured performance (8xA100, LLaMA-7B):**
   - 2x speedup over Megatron-V3+FlashAttention at 128K+ sequences
   - 40% communication overhead reduction vs Ring Attention
   - Handles 262K tokens on 32xA100 where Ring Attention and Megatron-V3 OOM

## References

- Sun, A., Zhao, W., Han, X., Yang, C., Liu, Z., Shi, C., & Sun, M. (2024). BurstAttention: An Efficient Distributed Attention Framework for Extremely Long Sequences. arXiv:2403.09347.
- Milakov, M. & Gimelshein, N. (2018). Online normalizer calculation for softmax. arXiv:1805.02867.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691.
- Liu, H., Zaharia, M., & Abbeel, P. (2023). Ring Attention with Blockwise Transformers for Near-Infinite Context. arXiv:2310.01889.
- Sun, W., et al. (2025). LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid. arXiv:2502.07563.
