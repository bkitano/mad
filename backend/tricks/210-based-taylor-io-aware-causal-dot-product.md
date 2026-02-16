# 210: BASED Taylor IO-Aware Causal Dot Product

**Category**: kernel
**Gain type**: efficiency
**Source**: Arora, Eyuboglu, Zhang et al. (2024) — ICML 2024
**Paper**: [papers/based-taylor-linear-attention.pdf]
**Documented**: 2026-02-15

## Description

Linear attention replaces softmax with a feature map $\phi$ so that attention can be computed via the associative property of matrix products, yielding $O(Nd^2)$ time and $O(d^2)$ recurrent state. However, naïve CUDA implementations of linear attention are **slower** than highly optimized FlashAttention-2 due to excessive memory traffic: the large KV-state $\sum_j \phi(\boldsymbol{k}_j)\boldsymbol{v}_j^\top \in \mathbb{R}^{D \times d}$ must be repeatedly read from and written to slow GPU memory (HBM/SRAM).

BASED introduces an **IO-aware kernel** for the 2nd-order Taylor feature map that fuses the feature map computation and the causal dot product into a single kernel, exploiting three levels of the GPU memory hierarchy (HBM → SRAM → registers). The key insights are:

1. **Split causal dot product into quadratic + linear terms**: For each tile of 16 output tokens, decompose the output into a local quadratic attention term (within the tile, using the causal mask) and a running cumulative KV-state term (linear attention across all prior tiles).

2. **Fuse feature map into the kernel**: The Taylor expansion $\phi(\boldsymbol{q})^\top\phi(\boldsymbol{k}) = 1 + \boldsymbol{q}^\top\boldsymbol{k} + \frac{(\boldsymbol{q}^\top\boldsymbol{k})^2}{2}$ is computed in-register rather than materializing the expanded $d^2$-dimensional feature vectors in HBM.

3. **Partition KV-state across warps in registers**: The large KV-state $\in \mathbb{R}^{D \times d}$ is partitioned across warps and stored in the fastest memory (thread registers), avoiding $O(BHNDd)$ bytes of SRAM↔register traffic per layer.

The result: BASED achieves **24× higher throughput** than FlashAttention-2 on generation (1024 tokens, 1.3B parameters, H100), and 56% faster prefill at 4K sequence length — while matching Transformer++ quality on language modeling.

## Mathematical Form

**Softmax attention (causal):**

$$
y_i = \sum_{j=1}^{i} \frac{\exp(\boldsymbol{q}_i^\top \boldsymbol{k}_j / \sqrt{d})}{\sum_{m=1}^{i} \exp(\boldsymbol{q}_i^\top \boldsymbol{k}_m / \sqrt{d})} \boldsymbol{v}_j
$$

**Linear attention with feature map $\phi$:**

$$
y_i = \frac{\phi(\boldsymbol{q}_i)^\top \sum_{j=1}^{i} \phi(\boldsymbol{k}_j) \boldsymbol{v}_j^\top}{\phi(\boldsymbol{q}_i)^\top \sum_{j=1}^{i} \phi(\boldsymbol{k}_j)}
$$

This admits a recurrent form with KV-state $\boldsymbol{s}_i \in \mathbb{R}^{D \times d}$ and K-state $\boldsymbol{z}_i \in \mathbb{R}^{D}$:

$$
\boldsymbol{s}_i = \boldsymbol{s}_{i-1} + \phi(\boldsymbol{k}_i) \boldsymbol{v}_i^\top, \quad \boldsymbol{z}_i = \boldsymbol{z}_{i-1} + \phi(\boldsymbol{k}_i)
$$

$$
\boldsymbol{y}_i = \frac{\phi(\boldsymbol{q}_i)^\top \boldsymbol{s}_i}{\phi(\boldsymbol{q}_i)^\top \boldsymbol{z}_i}
$$

**2nd-order Taylor feature map:**

The key feature map choice in BASED approximates $\exp(\boldsymbol{q}^\top \boldsymbol{k} / \sqrt{d})$ via a 2nd-order Taylor expansion. With $\boldsymbol{q}, \boldsymbol{k} \in \mathbb{R}^d$ projected to feature dimension $d'$ (typically $d' = 16$):

$$
\phi(\boldsymbol{q})^\top \phi(\boldsymbol{k}) = 1 + \boldsymbol{q}^\top \boldsymbol{k} + \frac{(\boldsymbol{q}^\top \boldsymbol{k})^2}{2}
$$

The implicit feature map $\phi: \mathbb{R}^{d'} \to \mathbb{R}^{D}$ where $D = 1 + d' + d'^2$ maps to the space of all monomials up to degree 2. With $d' = 16$: $D = 1 + 16 + 256 = 273$.

**IO-aware causal dot product decomposition:**

For each tile of $T_{\text{tile}} = 16$ tokens (chosen to match tensor core tile size), the output $\boldsymbol{y}_i$ for token $i$ in tile $t$ is split:

$$
\boldsymbol{y}_i = \underbrace{\text{Causal}(\boldsymbol{q}_i^T \boldsymbol{k}_i) \boldsymbol{v}_i}_{\text{quadratic (local, within tile)}} + \underbrace{\boldsymbol{q}_i \sum_{j=0}^{i-1}(\boldsymbol{k}_j \boldsymbol{v}_j)}_{\text{linear (cumulative KV-state)}}
$$

where the first term computes $\boldsymbol{q}^\top \boldsymbol{k}$ with causal masking within the 16-token tile (handled by the quadratic/linear view of the Taylor expansion), and the second term accumulates contributions from all prior tiles via the running KV-state.

**Key Definitions:**

- $N$ — sequence length
- $d$ — model/head dimension
- $d'$ — feature dimension for linear attention (typically 16)
- $D = 1 + d' + d'^2$ — expanded feature dimension after Taylor map
- $B$ — batch size, $H$ — number of heads
- $\boldsymbol{s}_i \in \mathbb{R}^{D \times d}$ — cumulative KV-state
- $\boldsymbol{z}_i \in \mathbb{R}^{D}$ — cumulative K-state (for denominator normalization)

## Complexity

| Operation | Naïve Linear Attn | BASED IO-Aware |
|-----------|------------------|----------------|
| HBM→SRAM data movement | $O(BHNDd)$ | **Avoided** (fused) |
| SRAM→Register data movement | $O(BHNDd)$ | **Avoided** (KV-state in registers) |
| Total HBM reads (prefill) | $2BHND + 2BHNd$ bytes | $2BHNd' + 2BHNd$ bytes |
| Feature map materialization | $O(BHNd'^2)$ in HBM | **In-register** (never materialized) |

**Memory:** KV-state is $D \times d = (1 + d' + d'^2) \times d$. With $d' = 16, d = 64$: KV-state = $273 \times 64 \approx 17K$ floats per head. Partitioned across warps, this fits entirely in registers.

**Wall-clock performance (H100 GPU):**

| Operation | FlashAttention-2 | BASED (IO-aware) | Speedup |
|-----------|-----------------|------------------|---------|
| Prefill (4K seq, 1.3B) | baseline | 56% faster | 1.56× |
| Generation (1024 tok, 1.3B, batch 128) | baseline | **24× faster** | 24× |
| Generation (1024 tok, 360M, batch 128) | baseline | **~118% faster** | 2.18× |

The generation speedup comes from BASED's $O(1)$ per-token cost (recurrent state update) vs. attention's $O(N)$ per-token cost (reading growing KV-cache).

## Applicability

- **Any linear attention model**: The IO-aware kernel applies to any feature map that can be expressed as a polynomial (Taylor, polynomial sketches), not just the 2nd-order Taylor. Higher-order Taylor maps would increase $D$ but the same register-partitioning strategy applies.

- **BASED architecture (hybrid)**: BASED combines Taylor linear attention (~20% of layers) with small sliding-window softmax attention (~20%) and gated convolutions (~60%). The IO-aware kernel is critical for making the linear attention component competitive.

- **Recurrent generation for any linear RNN**: The register-based KV-state accumulation strategy extends to GLA, RetNet, Mamba-style models — any model with a matrix-valued recurrent state that must be updated per-token during generation.

- **Prefill/training for sequence models**: The fused causal dot product kernel with tiled quadratic+linear decomposition accelerates the forward pass during training and prompt processing.

## Limitations

- **Feature dimension tradeoff**: The Taylor map expands dimension from $d'$ to $D = 1 + d' + d'^2$. With $d' = 16$, $D = 273$; with $d' = 32$, $D = 1057$. Larger $d'$ improves recall quality but quadratically increases KV-state size, potentially overflowing register capacity.

- **Approximation quality**: The 2nd-order Taylor expansion $1 + x + x^2/2$ is a truncated approximation of $e^x$. For large $|\boldsymbol{q}^\top\boldsymbol{k}|$, the approximation degrades. This limits expressivity compared to full softmax, which is why BASED supplements with sliding window attention.

- **Register pressure**: Storing the KV-state in registers limits the number of concurrent thread blocks per SM, reducing occupancy. This is acceptable when the kernel is memory-bound (generation) but may reduce throughput for compute-bound scenarios.

- **Tile size fixed at 16**: The 16-token tile size is chosen to match tensor core MMA instruction tile dimensions. Different GPU architectures may benefit from different tile sizes.

- **No decay/gating**: The basic BASED linear attention has no forget gate or decay mechanism. The recurrent state grows unboundedly, leading to interference over very long sequences. This is mitigated architecturally by the sliding window component, not by the kernel.

## Implementation Notes

```python
# BASED IO-aware Taylor linear attention — forward pass (prefill)
# Pseudocode matching Algorithm 1 from the paper

def based_taylor_forward_io_aware(q, k, v, d_prime=16):
    """
    IO-aware causal dot product with 2nd-order Taylor feature map.

    Args:
        q: (B, H, N, d_prime) - queries (projected to feature dim d')
        k: (B, H, N, d_prime) - keys (projected to feature dim d')
        v: (B, H, N, d) - values

    Key insight: Never materialize the expanded phi(q), phi(k) in HBM.
    Instead, compute phi(q)^T @ phi(k) = 1 + q^Tk + (q^Tk)^2/2
    directly in-register.
    """
    B, H, N, d = v.shape
    TILE = 16  # matches tensor core tile size

    # KV-state: partitioned across warps, stored in REGISTERS
    # Shape: (D, d) where D = 1 + d' + d'^2
    # Never written to HBM or SRAM during the inner loop
    kv_state = zeros(1 + d_prime + d_prime**2, d)  # in registers
    k_state = zeros(1 + d_prime + d_prime**2)       # in registers

    output = empty(B, H, N, d)

    for tile_idx in range(N // TILE):  # iterate over 16-token tiles
        # Load tile from HBM to SRAM (coalesced reads)
        q_tile = q[:, :, tile_idx*TILE:(tile_idx+1)*TILE]  # (B,H,16,d')
        k_tile = k[:, :, tile_idx*TILE:(tile_idx+1)*TILE]  # (B,H,16,d')
        v_tile = v[:, :, tile_idx*TILE:(tile_idx+1)*TILE]  # (B,H,16,d)

        # === TERM 1: Quadratic local attention (within tile) ===
        # Compute q^T k with causal mask, all in SRAM/registers
        # This is the "quadratic view" for 16 tokens only
        qk = q_tile @ k_tile.T  # (16, 16) — tensor core matmul
        qk = causal_mask(qk)     # zero out future positions
        # Taylor: score = 1 + qk + qk^2/2 (elementwise, in register)
        scores = 1.0 + qk + 0.5 * qk * qk
        local_out = scores @ v_tile  # (16, d) — tensor core matmul

        # === TERM 2: Linear cumulative KV-state (across tiles) ===
        # q_i @ kv_state gives contribution from all prior tiles
        # Feature map applied implicitly: phi(q) = [1, q, vec(qq^T/√2)]
        cumul_out = apply_taylor_feature(q_tile) @ kv_state  # in register

        # Combine
        output[:, :, tile_idx*TILE:(tile_idx+1)*TILE] = local_out + cumul_out

        # === Update KV-state in REGISTERS ===
        # kv_state += phi(k_i) @ v_i^T for each token in tile
        for i in range(TILE):
            phi_k = apply_taylor_feature(k_tile[:, :, i])  # in register
            kv_state += outer(phi_k, v_tile[:, :, i])      # register update
            k_state += phi_k                                # register update

    return output

# GPU Efficiency Analysis:
# 1. HBM reads: Only q, k, v tiles (2BHNd' + 2BHNd bytes)
# 2. HBM writes: Only output (BHNd bytes)
# 3. KV-state NEVER touches HBM — lives in registers across tiles
# 4. Feature map computed in-register, never materialized
# 5. Local attention uses tensor core matmuls (16×16 tiles)
# 6. 16-token tile matches WGMMA/MMA instruction dimensions
```

**GPU memory hierarchy exploitation:**

| Data | Location | Access Pattern |
|------|----------|---------------|
| $\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{v}$ input tiles | HBM → SRAM | Coalesced streaming reads |
| Local $QK^\top$ scores | SRAM (shared memory) | Computed and consumed in-place |
| KV-state $\boldsymbol{s} \in \mathbb{R}^{D \times d}$ | **Thread registers** | Persistent across tiles, never evicted |
| K-state $\boldsymbol{z} \in \mathbb{R}^{D}$ | **Thread registers** | Persistent across tiles |
| Output tiles | SRAM → HBM | Single coalesced write per tile |

**Why this is fast on real GPUs:**

1. **Minimal HBM bandwidth**: Only reads inputs and writes outputs — no intermediate materializations. This is critical since A100/H100 are HBM-bandwidth-limited for generation.
2. **Tensor core utilization**: The 16×16 local attention computation ($QK^\top$ and $S \cdot V$) maps directly to MMA/WGMMA instructions.
3. **Register-resident state**: The KV-state (17K floats for $d'=16, d=64$) fits in registers when partitioned across warps (256 KB register file per SM on H100).
4. **No kernel launch overhead**: Feature map, causal dot product, and state update are all fused into one kernel — no separate launches for $\phi(\boldsymbol{q})$, $\phi(\boldsymbol{k})$, matmul, etc.

## References

- Arora, S., Eyuboglu, S., Zhang, M., Timalsina, A., Alberti, S., Zinsley, D., Zou, J., Rudra, A., & Ré, C. (2024). Simple linear attention language models balance the recall-throughput tradeoff. ICML 2024. arXiv:2402.18668.
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.
- Zhang, M., Bhatia, K., Kumbong, H., & Ré, C. (2024). The hedgehog & the porcupine: Expressive linear attentions with softmax mimicry. ICLR 2024.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR 2024.
- Code: https://github.com/HazyResearch/based
- Kernels: https://github.com/HazyResearch/ThunderKittens
