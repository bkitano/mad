# 217: Lightning Attention-2 — Intra-Inter Block Causal Linear Attention Tiling

**Category**: kernel
**Gain type**: efficiency
**Source**: Qin, Sun, Li, Shen, Sun & Zhong (2024) — "Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models"
**Paper**: [papers/lightning-attention-2.pdf]
**Documented**: 2026-02-15

## Description

Linear attention replaces softmax with a kernel feature map $\phi$, enabling $O(n)$ complexity via the "right-product trick": compute $\phi(K)^\top V$ first (a $d \times d$ matrix), then multiply by $\phi(Q)$. However, in the **causal** setting, the cumulative summation (`cumsum`) required to maintain the running KV state $\text{kv}_t = \lambda \cdot \text{kv}_{t-1} + k_t^\top v_t$ breaks parallelism and creates a sequential bottleneck that prevents linear attention from realizing its theoretical speedup on GPU hardware.

Lightning Attention-2 solves this with a **divide-and-conquer tiling strategy** that separates the computation into two distinct components per block:

1. **Intra-block** (diagonal): Use conventional **quadratic** (left-product) attention $O_\text{intra} = (Q_i K_i^\top \odot M) V_i$ within each tile of size $B$. This is a standard masked matmul — fully parallel, tensor-core friendly, fits in SRAM.

2. **Inter-block** (off-diagonal): Use **linear** (right-product) attention $O_\text{inter} = \Lambda Q_i(\text{KV})$ where $\text{KV}$ is a running cumulative $d \times d$ state accumulated across all previous blocks. This avoids materializing the full $n \times n$ attention matrix.

The critical insight is that the **cumsum bottleneck only exists within blocks** (handled by the causal mask $M$ in the intra-block term), while **between blocks** the state update is a simple matrix addition that can be done incrementally. By choosing the block size $B$ appropriately, the intra-block quadratic cost $O(B^2 d)$ is bounded while the inter-block linear cost $O(B d^2)$ scales favorably.

This IO-aware tiling (load blocks from HBM to SRAM, compute both terms on-chip, write back) achieves **constant training speed** regardless of sequence length and is significantly faster than FlashAttention-2 at long sequences, while using linear memory.

## Mathematical Form

**Linear attention with decay (NormAttention / TransNormerLLM):**

$$
o_t = q_t \sum_{s \leq t} \lambda^{t-s} k_s^\top v_s
$$

**Recurrent form:**

$$
\text{kv}_t = \lambda \cdot \text{kv}_{t-1} + k_t^\top v_t, \quad o_t = q_t(\text{kv}_t)
$$

where $\text{kv}_t = \sum_{s \leq t} \lambda^{t-s} k_s^\top v_s \in \mathbb{R}^{d \times d}$ is the cumulative KV state.

**Block formulation:**

Divide the sequence of length $n$ into $T = n/B$ blocks $\{X_1, \ldots, X_T\}$ of size $B \times d$ each, for $X \in \{Q, K, V, O\}$.

Define the inter-block cumulative state:

$$
\text{KV}_0 = 0 \in \mathbb{R}^{d \times d}, \quad \text{KV}_t = \sum_{s \leq tB} \lambda^{tB - s} k_s^\top v_s
$$

**Core Decomposition — Output of block $t+1$, position $tB + r$ (for $1 \leq r \leq B$):**

$$
o_{tB+r} = q_{tB+r} \left(\sum_{s=tB+1}^{tB+r} \lambda^{tB+r-s} k_s^\top v_s + \lambda^r \sum_{s \leq tB} \lambda^{tB-s} k_s^\top v_s\right)
$$

In matrix form:

$$
O_{t+1} = \underbrace{(Q_{t+1} K_{t+1}^\top) \odot M) V_{t+1}}_{\text{Intra Block}} + \underbrace{\Lambda Q_{t+1}(\text{KV}_t)}_{\text{Inter Block}}
$$

**Key Definitions:**

- $M \in \mathbb{R}^{B \times B}$ — causal decay mask, $M_{st} = \lambda^{s-t}$ for $s \geq t$, $0$ otherwise
- $\Lambda = \text{diag}(\lambda, \lambda^2, \ldots, \lambda^B) \in \mathbb{R}^{B \times B}$ — per-position decay diagonal
- $\text{KV}_t \in \mathbb{R}^{d \times d}$ — accumulated KV state up to block $t$

**State update rule:**

$$
\text{KV}_{t+1} = \lambda^B \cdot \text{KV}_t + (\lambda^B \Lambda^{-1} K_{t+1})^\top V_{t+1}
$$

This is a simple matrix addition performed on-chip after the intra and inter block outputs are computed.

**Backward pass — Intra/Inter decomposition for gradients:**

$$
dQ_{t+1} = \underbrace{((dO_{t+1} V_{t+1}^\top) \odot M) K_{t+1}}_{\text{Intra Block}} + \underbrace{\Lambda dO_{t+1}(\text{KV}_t^\top)}_{\text{Inter Block}}
$$

$$
dK_{t-1} = \underbrace{((dO_{t-1} V_{t-1}^\top) \odot M)^\top Q_{t-1}}_{\text{Intra Block}} + \underbrace{\lambda^B \Lambda^{-1} V_{t-1}(\text{dKV}_t^\top)}_{\text{Inter Block}}
$$

$$
dV_{t-1} = \underbrace{((Q_{t-1} K_{t-1}^\top) \odot M)^\top dO_{t-1}}_{\text{Intra Block}} + \underbrace{(\Lambda Q_{t-1})^\top \text{dKV}_t}_{\text{Inter Block}}
$$

where $\text{dKV}$ is the backward running state, accumulated in reverse order.

## Complexity

| Operation | Naive Linear Attn (causal) | FlashAttention-2 | Lightning Attention-2 |
|-----------|---------------------------|------------------|----------------------|
| Forward FLOPs | $O(n d^2)$ | $O(n^2 d)$ | $O(n d^2)$ (inter) + $O(n B d)$ (intra) |
| Sequential steps | $O(n)$ (cumsum) | $O(n/B)$ tiles | $O(n/B)$ blocks |
| Memory | $O(n d)$ | $O(n d)$ | $O(n d)$ |
| HBM reads/writes | $O(n d)$ per step (cumsum) | $O(n^2 d / M)$ | $O(n d)$ total |
| Training speed scaling | Degrades with $n$ | $O(n^2)$ | **Constant** w.r.t. $n$ |

**Memory:** $O(Bd + d^2)$ SRAM per block — the $B \times d$ tiles for $Q_i, K_i, V_i$ plus the $d \times d$ running KV state. Total HBM usage is $O(nd)$ for inputs/outputs (linear in sequence length).

**Wall-clock performance (A100 80GB, from Table 1):**

| Seq Length | LLaMA-FA2 (TGS) | TNL-LA1 (TGS) | TNL-LA2 (TGS) |
|-----------|------------------|----------------|----------------|
| 1K | 35,931 | 38,615 | 38,172 |
| 8K | 21,996 | 28,627 | 37,755 |
| 32K | 9,715 | 13,852 | 37,364 |
| 64K | 5,643 | 8,247 | 38,278 |
| 128K | 4,078 | 6,012 | **38,596** |

Lightning Attention-2 maintains **~38K tokens/sec** regardless of sequence length, while FlashAttention-2 drops from 36K to 4K as length increases from 1K to 128K.

## Applicability

- **All causal linear attention variants:** TransNormerLLM, RetNet/Retention, RWKV, Linear Transformer — any model using $O = Q(K^\top V)$ with causal masking can adopt this tiling
- **Gated linear attention (GLA):** Models with data-dependent decay $\lambda_t$ fit the same intra/inter decomposition, though the mask matrix $M$ becomes input-dependent (see trick 177)
- **Long-context LLM pretraining:** The constant-speed property makes this ideal for training on very long sequences (64K–128K+) where FlashAttention becomes prohibitively slow
- **Linear RNNs and SSMs in quadratic mode:** When Mamba-2's SSD or other SSMs use the quadratic (attention-like) form within chunks, the intra/inter decomposition is essentially the same pattern (see trick 109)
- **Inference with large batch sizes:** During prefill, Lightning Attention-2 provides significant speedup over FlashAttention-2 due to the linear complexity in sequence length

## Limitations

- **Requires a kernel feature map:** The inter-block computation relies on the associative "right-product" trick $Q(K^\top V)$, which requires linear (not softmax) attention. Cannot be applied to standard softmax attention
- **Block size tuning:** $B$ must balance intra-block quadratic cost ($O(B^2 d)$) vs inter-block state size ($O(d^2)$). Too large $B$ wastes SRAM on the causal mask; too small $B$ increases the relative cost of KV state updates
- **Quality gap vs softmax attention:** Linear attention models (TransNormerLLM) have a small but consistent quality gap compared to softmax Transformers (LLaMA) on language modeling benchmarks. Lightning Attention-2 closes the *speed* gap but not the *quality* gap
- **The $d^2$ state is the real bottleneck:** The KV state $\in \mathbb{R}^{d \times d}$ must fit in SRAM. For large head dimensions (e.g., $d = 128$), this is $128 \times 128 \times 2 = 32$KB in BF16 — manageable but limits the head dimension
- **Decay rate $\lambda$ is global:** The paper assumes a single scalar decay $\lambda$ shared across all positions. Models with per-position or per-head decay (GLA, RetNet) require a more general mask $M$, though the same intra/inter principle applies

## Implementation Notes

```python
# Lightning Attention-2 Forward Pass (Algorithm 1 from paper)
# Triton kernel pseudocode

def lightning_attention_2_forward(Q, K, V, lam, B):
    """
    Q, K, V: (n, d) — query, key, value matrices
    lam: scalar decay rate
    B: block size
    Returns O: (n, d) — output
    """
    n, d = Q.shape
    T = n // B  # number of blocks

    # Precompute causal decay mask (B x B), fits in SRAM
    M = torch.zeros(B, B)
    for s in range(B):
        for t in range(B):
            M[s, t] = lam ** (s - t) if s >= t else 0.0

    # Diagonal decay vector
    Lambda = torch.diag(torch.tensor([lam ** (i+1) for i in range(B)]))

    # Running KV state (d x d) — lives in SRAM
    KV = torch.zeros(d, d)
    O = torch.zeros(n, d)

    for i in range(T):
        # Load block i from HBM to SRAM
        Qi = Q[i*B:(i+1)*B]   # (B, d)
        Ki = K[i*B:(i+1)*B]   # (B, d)
        Vi = V[i*B:(i+1)*B]   # (B, d)

        # INTRA-BLOCK: quadratic attention with causal mask
        # Uses tensor cores — standard attention on small B×B tile
        O_intra = (Qi @ Ki.T * M) @ Vi          # (B, d)

        # INTER-BLOCK: linear attention using accumulated KV state
        O_inter = (Lambda @ Qi) @ KV              # (B, d)

        # Output is sum of both terms
        O[i*B:(i+1)*B] = O_intra + O_inter

        # UPDATE KV STATE on chip (no HBM round-trip)
        KV = lam**B * KV + (lam**B * torch.inverse(Lambda) @ Ki).T @ Vi

        # Write output block back to HBM
    return O

# KEY INSIGHT: Why this is fast
#
# 1. Intra-block: (B,d) @ (d,B) -> (B,B) @ (B,d) -> (B,d)
#    This is two matmuls on small tiles — TENSOR CORE friendly
#    The B×B causal mask M is applied via Hadamard product
#
# 2. Inter-block: (B,d) @ (d,d) -> (B,d)
#    This is one matmul with the d×d KV state — TENSOR CORE friendly
#
# 3. State update: (d,B) @ (B,d) -> (d,d)
#    One matmul to update KV state — stays in SRAM
#
# 4. No cumsum needed! The causal ordering within a block is handled
#    by the mask M (dense matmul), and between blocks by the sequential
#    KV state accumulation (just one matrix addition per block)
#
# Compare to naive causal linear attention:
#   for t in range(n):  # O(n) sequential steps!
#       kv = lam * kv + k[t][:,None] @ v[t][None,:]  # rank-1 update
#       o[t] = q[t] @ kv
#
# Lightning Attn-2 reduces sequential steps from O(n) to O(n/B)
# and converts all compute to tensor-core matmuls
```

**GPU Efficiency Analysis:**

- **Memory access pattern:** Fully coalesced — blocks of $Q, K, V$ are loaded contiguously from HBM to SRAM, processed, and output is written back contiguously. The KV state stays in SRAM across all iterations (no HBM round-trip for the state)
- **Tensor core utilization:** All three compute-heavy operations (intra-block QK^T matmul, inter-block Q·KV matmul, KV state update K^T·V matmul) are standard dense matrix multiplications that map directly to tensor cores (WMMA/MMA instructions)
- **Arithmetic intensity:** For block size $B$ and head dim $d$: intra-block is $O(B^2 d)$ FLOPs over $O(Bd)$ bytes loaded = $O(B)$ FLOPs/byte. Inter-block is $O(Bd^2)$ FLOPs over $O(Bd)$ bytes = $O(d)$ FLOPs/byte. Both are compute-bound for reasonable $B$ and $d$
- **Parallelism:** Each block's intra-block computation is independent across heads and batch elements. The inter-block sequential dependency is only $O(n/B)$ steps, and each step is a large matmul that saturates GPU SMs
- **SRAM budget:** Per block: $3 \times B \times d$ (Q,K,V tiles) + $d^2$ (KV state) + $B^2$ (mask M) + $B \times d$ (output). For $B = 256$, $d = 128$: $3 \times 64$KB + $32$KB + $128$KB + $64$KB ≈ $416$KB in BF16 — fits in H100 shared memory (228KB per SM with 2 blocks, or use register file)
- **Implemented in Triton:** The actual kernel is written in Triton, enabling efficient autotuning of block sizes and leveraging Triton's automatic memory management for the tiling

## References

- Qin, Z., Sun, W., Li, D., Shen, X., Sun, W., & Zhong, Y. (2024). Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models. arXiv:2401.04658.
- Qin, Z., et al. (2023). Scaling TransNormer to 175 Billion Parameters (TransNormerLLM). arXiv:2307.14995.
- Qin, Z., et al. (2023). Lightning Attention-1. (FlashAttention-1/2 style IO-aware tiling for the left product in linear attention.)
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. (Related chunkwise-parallel approach for GLA.)
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. (The IO-aware tiling paradigm this work extends to linear attention.)
