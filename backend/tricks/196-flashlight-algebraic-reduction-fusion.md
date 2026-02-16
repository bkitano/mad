# 196: Flashlight — Algebraic Reduction Fusion for Compiler-Generated Tiled Attention

**Category**: kernel
**Gain type**: efficiency
**Source**: You, Wang, Mustafaoglu, Jangda, Moreira, Dathathri, Mahajan, Pingali — "Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants" (arXiv 2511.02043, Nov 2025)
**Paper**: [papers/flashlight-compiler-tiled-attention.pdf]
**Documented**: 2026-02-16

## Description

FlashAttention-style tiled and fused kernels achieve near-peak GPU throughput by computing attention tile-by-tile in SRAM, never materializing the full $N \times N$ attention matrix in HBM. However, every new attention variant (differential attention, Evoformer, linear attention, etc.) currently requires a **hand-written fused kernel** or must fit the rigid FlexAttention template. This limits adoption of novel attention mechanisms that don't match the `softmax(score_mod(QK^T/√d))V` pattern.

**Flashlight** solves this by introducing three compiler transformations within PyTorch's `torch.compile` stack (TorchInductor) that **automatically generate** FlashAttention-style tiled and fused kernels from standard PyTorch code. The key innovation is a **unified reduction IR** that models matrix multiplications as generalized reductions, combined with **algebraic transformation rules** based on ring homomorphisms that enable fusing dependent multi-pass reductions (like stable softmax) into single-pass online algorithms.

The three transformations are:

1. **Structural fusion with dimension demotion**: Converts a producer's parallel (p) dimension into a consumer's reduction (r) dimension, enabling fusion of matmul with subsequent operations by trading parallelism for elimination of HBM materialization.

2. **Semantic fusion with algebraic transformation**: Exploits ring homomorphisms (e.g., $e^{x+y} = e^x \cdot e^y$) to automatically convert multi-pass dependent reductions (stable softmax = max-pass + sum-pass) into single-pass online algorithms — generalizing the online softmax trick to arbitrary reductions with homomorphic structure.

3. **Structural fusion with tiling-aware dimension elimination**: When a dimension fits entirely within a single tile ($B_P \geq |P|$), the tile-level loop collapses to a single iteration, enabling fusion of chained matmuls (e.g., $(A \cdot B) \cdot D$) that are otherwise impossible to fuse.

On H100 and A100, Flashlight matches or beats FlexAttention for supported variants, and achieves **5× speedup** for Evoformer and **6–9% end-to-end inference improvement** for AlphaFold2 — all from unmodified PyTorch code compiled with a single flag.

## Mathematical Form

**Unified Reduction IR — Computation Sketches:**

Every tensor operation is classified by its **computation sketch** $[(P_0, P_1, \ldots), (R_0, R_1, \ldots)]$, where $P_i$ are parallel (data-independent) dimensions and $R_j$ are reduction (data-dependent) dimensions.

**Examples:**

- **Element-wise addition:** $C(P_0, P_1) = A(P_0, P_1) + B(P_0, P_1)$ has sketch $[(P_0, P_1), ()]$ — no reduction.
- **GEMM:** $C_{mn} = \sum_k A_{mk} B_{kn}$ has sketch $[(M, N), (K)]$ — the contracted dimension $k$ is a reduction.
- **Softmax:** $\sigma(x)_i = e^{x_i} / \sum_j e^{x_j}$ has sketch $[(), (R)]$ — pure reduction.

**Key insight:** By modeling GEMM as a generalized reduction within the same IR as elementwise and reduction ops, the fusion boundary between GEMM and surrounding operations is dissolved.

**Dimension Demotion Rule:**

A producer kernel $K_0$ with sketch $[(P_{\text{common}}, P_{\text{producer}}), (\ldots)]$ can be fused with a consumer kernel $K_1$ with sketch $[(P_{\text{common}}), (P_{\text{producer}}, \ldots)]$ to produce:

$$
K_{\text{fused}} : [(P_{\text{common}}), (P_{\text{producer}}, \ldots)]
$$

The dimension $P_{\text{producer}}$ is **demoted** from parallel to reduction. The producer's output is generated and consumed within SRAM of the fused kernel, eliminating HBM materialization of the intermediate tensor.

**Algebraic Transformation — Ring Homomorphism:**

For fusing the stable softmax (two passes: max then sum-exp) into one pass:

**Definition:** A function $f: A \to B$ is a **homomorphism** if $f(a_1 \oplus a_2) = f(a_1) \otimes f(a_2)$ where $(A, \oplus)$ and $(B, \otimes)$ are groups.

For softmax, $f(x) = e^x$ is a homomorphism from $(\mathbb{R}, +)$ to $(\mathbb{R}_{>0}, \times)$ because $e^{a+b} = e^a \cdot e^b$.

**Stable softmax** (two-pass) computes:

$$
ds[j] = ds[j-1] \oplus \left(E(x[j]) \otimes E(\ominus m[N])\right) \quad \text{for } N \geq j \geq 1
$$

where $E = e^{(\cdot)}$, $\oplus = +$, $\otimes = \times$, $\ominus m[N] = -\max(x_1, \ldots, x_N)$.

**Online softmax** (one-pass) computes:

$$
do[j] = \left(do[j-1] \otimes E(m[j-1] \oplus (\ominus m[j]))\right) \oplus \left(E(x[j]) \otimes E(\ominus m[j])\right)
$$

Flashlight proves by induction that $ds[N] = do[N]$, enabling the compiler to automatically transform the two-pass algorithm into the single-pass version when the algebraic homomorphism structure is detected.

**Correction factor for running accumulation:**

$$
S_{\text{new}} = S_{\text{old}} \times \exp(m_{\text{old}} - m_{\text{new}}) + \exp(x_j - m_{\text{new}})
$$

where $m_{\text{old}}$ and $m_{\text{new}}$ are the running maximums before and after incorporating $x_j$.

**Tiling-Aware Dimension Elimination:**

For chained matmul $E = (A \cdot B) \cdot D$:

- Producer: $C[M, N] = A[M, K] \otimes B[K, N]$ with sketch $[(M, N), (K)]$
- Consumer: $E[M, P] = C[M, N] \otimes D[N, P]$ with sketch $[(M, P), (N)]$

Standard fusion is impossible because the producer's p-dimension $N$ doesn't match the consumer's structure. With tiling where $B_N \geq |N|$ (dimension $N$ fits in a single tile), the tile-level loop for $N$ becomes $\lceil |N|/B_N \rceil = 1$, collapsing it. The tiled sketches become:

$$
\text{Producer: } [(\tfrac{M}{B_M}, \underbrace{\tfrac{N}{B_N}}_{=1}), (\tfrac{K}{B_K})] \implies [(\tfrac{M}{B_M}), (\tfrac{K}{B_K})]
$$

$$
\text{Consumer: } [(\tfrac{M}{B_M}), (\underbrace{\tfrac{N}{B_N}}_{=1}, \tfrac{K_{\text{tile}}}{B_{K'}})]
$$

Now dimension demotion can fuse these into a single kernel.

**Logical Grid Dimensions:**

Flashlight decouples logical tiling from physical GPU grid mapping. Instead of flattening multiple p-dimensions into CUDA's asymmetric grid (X up to $2^{31}-1$, Y/Z up to 65,535), Flashlight defines a logical multi-dimensional tile grid, maps it to a single linear sequence, and recovers tile coordinates via inverse affine maps inside the kernel:

$$
\text{tile\_id} = p_0 \cdot |P_1| + p_1 \quad \longleftrightarrow \quad p_0 = \text{tile\_id} \mathbin{//} |P_1|, \; p_1 = \text{tile\_id} \bmod |P_1|
$$

This enables independent per-dimension tile-size tuning.

## Complexity

| Operation | torch.compile (baseline) | Flashlight |
|-----------|-------------------------|------------|
| Vanilla attention | 2 kernel launches (GEMM + softmax+GEMM) | 1 fused kernel |
| DiffAttn | Multiple unfused kernels | 1 fused tiled kernel |
| Evoformer gated self-attn | Multiple unfused kernels | 1 fused tiled kernel (**5× faster**) |

**Memory:** Flashlight eliminates materialization of the $N \times N$ attention matrix and intermediate GEMM results to HBM, matching FlashAttention's $O(N)$ memory.

**HBM accesses:** For vanilla attention:
- Unfused: $O(N^2 d)$ bytes (materialize $S = QK^T$) + $O(N^2)$ (softmax) + $O(N^2 d)$ (output)
- Flashlight fused: $O(N d)$ bytes — same as hand-tuned FlashAttention

**Arithmetic intensity increase:** By fusing GEMM + softmax + GEMM into one kernel, the arithmetic intensity rises from memory-bound (each op separately has low reuse) to compute-bound (data loaded once, used for all three operations).

**Performance (H100, forward pass, representative configs):**

| Variant | FlexAttention | Flashlight | Speedup |
|---------|--------------|------------|---------|
| Sliding Window (bs=8, seq=2048) | 3.31 ms | 2.50 ms | 1.32× |
| Causal (bs=8, seq=2048) | 1.74 ms | 1.31 ms | 1.33× |
| Document Mask (bs=32, seq=512) | 4.04 ms | 2.70 ms | 1.50× |
| DiffAttn (nhead=16, dim=128) | N/A (unsupported) | 0.48× of torch.compile | ~2× |
| Evoformer (nhead=4, dim=64) | N/A (unsupported) | 0.37× of torch.compile | ~2.7× |

## Applicability

- **All softmax attention variants:** Vanilla, ALiBi, softcap, causal, sliding window, PrefixLM, document mask — Flashlight generates fused kernels competitive with FlexAttention.

- **Attention variants beyond FlexAttention:** Differential Attention (Ye et al., 2024), Evoformer row/column-wise gated self-attention (Jumper et al., 2021), Invariant Point Attention (IPA), Rectified Sparse Attention (RSA) — these cannot be expressed in FlexAttention's `score_mod`/`block_mask` template but are handled natively by Flashlight.

- **Linear attention (potential):** The unified reduction IR can model linear attention's outer-product state accumulation as a reduction. The dimension demotion rule would fuse the Q·(K^T·V) computation, though the lack of softmax simplifies the algebraic transformation step. This is not evaluated in the paper but is architecturally supported.

- **TFLA-style chunkwise kernels:** Flashlight's tiling-aware dimension elimination could potentially automate TFLA's two-level tiling for linear RNN chunkwise kernels, since the inner tile dimension can be collapsed when it fits in SRAM. The structural fusion rules would handle the intra-chunk matmul chain ($S = QK^T \odot D$, $H = SV + QC$).

- **Any new attention variant written in PyTorch:** The key value proposition — researchers write standard PyTorch attention code, compile with `enable_flashlight=True`, and get FlashAttention-level performance automatically.

## Limitations

- **Floating-point non-associativity:** Semantic fusion via algebraic transformation assumes real-number arithmetic. In practice, floating-point addition/multiplication is not associative, so the online softmax transformation may introduce small numerical differences. The paper reports no observable accuracy loss for tested variants, matching prior FlashAttention results.

- **No block-sparsity exploitation:** Unlike FlexAttention, Flashlight does not inspect attention masks to skip fully-masked blocks. FlexAttention's `block_mask` approach can skip entire tiles of computation for sparse patterns (e.g., sliding window), which can be faster for very sparse attention patterns despite the overhead of mask creation and inspection.

- **Compile-time cost:** Flashlight adds compilation overhead to `torch.compile`. For models compiled once and run many times (standard training), this is amortized. For dynamic shapes or frequent recompilation, the overhead may matter.

- **Backward pass not explicitly discussed:** The paper focuses on forward-pass fusion. Backward-pass fusion for attention (which requires recomputation of attention weights) would need additional compiler support for the checkpointing/recomputation pattern.

- **Limited to attention-like patterns:** The fusion rules are general but the evaluation focuses on attention. Applicability to other fused operator patterns (e.g., fused FFN, fused normalization + attention) is not explored.

## Implementation Notes

```python
# Using Flashlight is trivial — just standard PyTorch + compile flag:

def diff_attn(q, k, v, lambda_full):
    """Differential Attention (Ye et al., 2024) — NOT supported by FlexAttention."""
    q0, q1 = q.chunk(2, dim=1)
    k0, k1 = k.chunk(2, dim=1)
    attn0 = attention(q0, k0, v)
    attn1 = attention(q1, k1, v)
    output = attn0 - lambda_full * attn1
    return output

# Compile with Flashlight — generates a single fused tiled kernel automatically
diff_attn_compiled = torch.compile(
    diff_attn,
    dynamic=False,
    enable_flashlight=True  # <-- the only change needed
)
output = diff_attn_compiled(q, k, v, lambda_full=0.2)

# Under the hood, Flashlight applies:
# 1. Unified Reduction IR: Models QK^T as reduction over K dimension
# 2. Dimension Demotion: Fuses QK^T (producer) with softmax (consumer)
#    by demoting K's output dimension N into softmax's reduction dimension
# 3. Algebraic Transformation: Converts stable softmax (2-pass) to online
#    softmax (1-pass) via exponential homomorphism
# 4. Dimension Elimination: Collapses small dimensions that fit in one tile
# 5. Logical Grid: Maps independent tile dimensions to GPU thread blocks
# Result: Single Triton kernel matching FlashAttention performance
```

**Key GPU efficiency properties:**

1. **Single kernel launch:** Eliminates all intermediate HBM materialization and kernel launch overhead for the entire attention computation.

2. **All dominant operations are matmuls:** The fused kernel contains tiled GEMM operations that map directly to tensor cores (WGMMA on H100, MMA on A100).

3. **IO-aware tiling:** Like FlashAttention, the fused kernel computes attention tile-by-tile in SRAM. The tiling strategy is determined by the compiler's autotuner, not hand-coded.

4. **Composable with existing compiler passes:** Flashlight's rewrites are applied within TorchInductor and compose with existing optimizations (vectorization, auto-tuning, ahead-of-time compilation).

5. **Reduction fusion is the key enabler:** The ability to fuse matmul + reduction (softmax, gating, normalization) into a single tiled loop is what makes FlashAttention-style performance possible. Without this, each GEMM and each reduction would be a separate kernel with separate HBM round-trips.

## References

- You, B., Wang, I., Mustafaoglu, Z. S., Jangda, A., Moreira, A., Dathathri, R., Mahajan, D., & Pingali, K. (2025). Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants. arXiv:2511.02043.
- Milakov, M. & Gimelshein, N. (2018). Online normalizer calculation for softmax. arXiv:1805.02867.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.
- Dong, Z., et al. (2024). FlexAttention. PyTorch Blog.
- Ye, Z., et al. (2024). Differential Transformer. arXiv:2410.05258.
- Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature.
