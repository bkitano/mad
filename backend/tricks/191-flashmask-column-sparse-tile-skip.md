# 191: FlashMask — Column-Wise Sparse Mask with Tile-Level Block Skipping

**Category**: kernel
**Gain type**: efficiency
**Source**: Wang, Zeng, Xiao, Wu, Yang, Zheng, Chen, Bian, Yu & Wang (2025) — ICLR 2025
**Paper**: [papers/flashmask-column-sparse-tile-skipping.pdf]
**Documented**: 2026-02-16

## Description

FlashAttention natively supports only a fixed set of attention masks (causal, sliding window, document mask). To handle arbitrary mask patterns, previous approaches resort to dense $N \times N$ mask matrices with $O(N^2)$ memory, which is prohibitive for long sequences. FlexAttention (PyTorch) uses compiler-generated block masks but still requires $O(N^2 / B_r B_c)$ memory for block-level storage.

**FlashMask** introduces a **column-wise sparse representation** of attention masks using just four $O(N)$-sized vectors, enabling $O(N)$ memory complexity for mask storage. The key observation is that for all practical attention mask types (causal, sliding window, document, prefix, blockwise, etc.), the set of masked rows for each key column $j$ forms at most two contiguous intervals — one in the lower-left triangle and one in the upper-right triangle. This column-wise interval representation is then converted to **tile-level min/max bounds** during a preprocessing step, enabling the kernel to classify each tile as:

- **Fully masked** → skip entirely (no HBM load, no compute)
- **Partially masked** → load mask bounds from SRAM, apply element-wise
- **Unmasked** → compute normally

This tile-level block skipping reduces computational complexity from $O(N^2)$ to $O((1-\rho) \cdot T_r \cdot T_c)$ where $\rho$ is the fraction of fully masked blocks. For causal attention, $\rho \approx 0.5$, giving roughly 2× speedup over dense FlashAttention.

**Relevance to TFLA/chunkwise linear attention**: FlashMask's tile-level skip optimization is directly applicable to TFLA's intra-chunk computation. In TFLA, the intra-chunk attention matrix $S^{(k)} \in \mathbb{R}^{L \times L}$ has a causal structure (the gate matrix $D^{(k)}$ is lower-triangular). When TFLA tiles this matrix into $B_{Lhq} \times B_{Lkv}$ blocks, upper-triangular tiles are zero and can be skipped entirely. FlashMask's column-wise representation can encode this causal structure plus any additional sparsity (e.g., sliding window within chunks, document boundaries) with $O(L)$ memory per chunk. Additionally, for non-causal masks in post-training tasks (SFT, DPO, RM), FlashMask enables TFLA-style kernels to skip computation for packed sequences where different documents should not attend to each other.

## Mathematical Form

**Attention with mask:**

$$
S = \frac{QK^\top}{\sqrt{d_k}}, \quad P = \text{Softmax}(S + M), \quad O = PV
$$

where $M \in \mathbb{R}^{N \times N}$ with $M_{ij} \in \{0, -\infty\}$ controls token visibility.

**Column-wise sparse representation:**

For each key column $j \in \{1, \ldots, N\}$, the masked rows are described by two intervals:

$$
M_j = [LTS_j, LTE_j) \cup [UTS_j, UTE_j)
$$

where:
- $LTS_j$ — **L**ower **T**riangular **S**tart: first masked row in lower-left triangle
- $LTE_j$ — **L**ower **T**riangular **E**nd: last masked row (exclusive) in lower-left triangle
- $UTS_j$ — **U**pper **T**riangular **S**tart: first masked row in upper-right triangle
- $UTE_j$ — **U**pper **T**riangular **E**nd: last masked row (exclusive) in upper-right triangle

For query row $i$ attending to key column $j$:

$$
M_{ij} = \begin{cases} -\infty & \text{if } i \in [LTS_j, LTE_j) \cup [UTS_j, UTE_j) \\ 0 & \text{otherwise} \end{cases}
$$

**Storage:** Four vectors $LTS, LTE, UTS, UTE \in \mathbb{R}^N$ — total $O(N)$ memory.

**Tile-level preprocessing:**

Given block sizes $B_r$ (query tile rows) and $B_c$ (key tile columns), precompute min/max bounds for each key tile $j \in \{1, \ldots, T_c\}$ where $T_c = \lceil N/B_c \rceil$:

$$
LTStart_j^{min} = \min_{j' \in \text{tile}_j} LTS_{j'}, \quad LTStart_j^{max} = \max_{j' \in \text{tile}_j} LTS_{j'}
$$

$$
LTEnd_j^{min} = \min_{j' \in \text{tile}_j} LTE_{j'}, \quad LTEnd_j^{max} = \max_{j' \in \text{tile}_j} LTE_{j'}
$$

(Analogously for $UTStart$ and $UTEnd$.)

**Block classification rule** for query tile $i$ and key tile $j$:

$$
T_{block} = \begin{cases}
\text{Fully masked} & \text{if } BlockRow_{min} \geq Start^{max} \text{ and } BlockRow_{max} \leq End^{min} \\
\text{Partially masked} & \text{if } BlockRow_{min} < End^{max} \text{ and } BlockRow_{max} > Start^{min} \\
\text{Unmasked} & \text{otherwise}
\end{cases}
$$

where $BlockRow_{min} = (i-1) \times B_r$ and $BlockRow_{max} = i \times B_r$.

**Key Definitions:**

- $N$ — sequence length
- $B_r, B_c$ — block sizes for query and key tiles (e.g., 128, 64)
- $T_r = \lceil N/B_r \rceil$ — number of query tile rows
- $T_c = \lceil N/B_c \rceil$ — number of key tile columns
- $\rho = \alpha / (T_r \times T_c)$ — block sparsity ratio ($\alpha$ = number of fully masked blocks)
- $LTS, LTE, UTS, UTE \in \mathbb{R}^N$ — column-wise mask interval vectors

## Complexity

| Metric | Dense Mask | FlexAttention | FlashMask |
|--------|-----------|---------------|-----------|
| Mask memory | $O(N^2)$ | $O(N^2 / B_r B_c)$ | $O(N)$ |
| HBM reads for mask | $O(N^2)$ | $O(N^2 / B_r B_c)$ | $4 \times T_r \times N$ |
| Compute | $O(N^2)$ | $O((1-\rho) N^2)$ | $O((1-\rho) T_r T_c)$ |

**Memory reduction factor:** For $N = 128K$ with $B_c = 64$:
- Dense mask: $128K \times 128K \times 2$ bytes $= 32$ GB
- FlashMask: $4 \times 128K \times 4$ bytes $= 2$ MB (**16,000× reduction**)

**End-to-end training throughput (Llama-2, A100 80GB):**

| Model | Task | Seq Length | FlashAttn Dense | FlashMask | Speedup |
|-------|------|-----------|----------------|-----------|---------|
| 7B | SFT | 64K | baseline | 1.65× | 65% |
| 7B | LoRA | 128K | OOM | 544K max | $\infty$ |
| 13B | DPO | 64K | baseline | 2.03× | 103% |
| 70B | RM | 64K | baseline | 2.06× | 106% |
| 7B | LoRA | 128K | baseline | 3.22× | 222% |

**Kernel-level performance (A100, BF16, headdim=128):**

FlashMask achieves **37.8% to 62.3%** of theoretical maximum FLOPs/s on A100, outperforming FlexAttention by **12.1% to 60.7%** in kernel TFLOPs/s across 12 mask types at sequence lengths 8K–128K.

## Applicability

- **FlashAttention with complex masks (primary):** Drop-in replacement for dense mask FlashAttention in SFT, LoRA, DPO, RM training. Supports causal, sliding window, document, prefix, blockwise, shared question, QK-sparse, random eviction, and combination masks.

- **TFLA intra-chunk tiling:** TFLA's inner tiles of the causal gate matrix $D^{(k)}$ can be classified as fully masked/unmasked/partial using FlashMask's column-wise representation. Upper-triangular tiles (future positions) are fully masked and can be skipped, reducing TFLA's intra-chunk compute by ~50% for causal models.

- **Packed sequence training:** When multiple documents are packed into a single sequence for efficient training, FlashMask's document mask support prevents cross-document attention without padding, directly applicable to TFLA-based linear RNN training.

- **Sparse attention patterns:** Any attention pattern expressible as column-wise intervals (which covers most practical patterns) benefits from tile-level skipping.

- **Long-context pretraining:** The $O(N)$ mask memory enables training with sequences up to 544K tokens on 80GB GPUs (vs 64K with dense masks), critical for long-context linear attention models.

## Limitations

- **Cannot represent arbitrary masks:** The column-wise interval representation requires each column's masked region to be at most two contiguous intervals. Completely random masks (e.g., dropout-style) cannot be efficiently represented. However, this covers the vast majority of practical attention patterns.

- **Preprocessing overhead:** Computing the min/max tile bounds requires a reduction over each tile's columns. This is $O(T_c)$ work, negligible compared to the attention computation but adds an extra kernel launch.

- **Currently PaddlePaddle only:** The open-source implementation is in PaddlePaddle/PaddleNLP. Porting to PyTorch/Triton would be needed for broader adoption and integration with TFLA.

- **Block sparsity depends on mask pattern:** For fully dense masks (no sparsity), FlashMask has no computational advantage over FlashAttention — only the memory savings remain. The speedup is proportional to $\rho$, the fraction of skippable blocks.

- **Not yet extended to Hopper:** The implementation targets A100 (Ampere). Extending to H100 with TMA async loads and warp specialization could further improve performance.

- **Backward pass has different parallelization:** The $dK$ and $dV$ gradients are column-parallel, which naturally aligns with FlashMask's column-wise representation. But the $dQ$ gradient accumulation order can introduce non-determinism (line 27 of Algorithm 2).

## Implementation Notes

```python
# FlashMask forward pass pseudocode (Algorithm 1 from the paper)
def flashmask_forward(Q, K, V, LTS, LTE, UTS, UTE, B_r, B_c):
    """
    FlashAttention-2 extended with column-wise sparse mask.

    Args:
        Q, K, V: (N, d) input matrices in HBM
        LTS, LTE: (N,) lower triangular mask start/end indices
        UTS, UTE: (N,) upper triangular mask start/end indices
        B_r, B_c: block sizes for query rows and key columns
    """
    N, d = Q.shape
    T_r = ceil(N / B_r)
    T_c = ceil(N / B_c)

    # === PREPROCESSING (once, in HBM) ===
    # Compute tile-level min/max bounds for each key tile
    for j in range(T_c):
        cols = range(j * B_c, (j+1) * B_c)
        LTStart_min[j] = min(LTS[cols])
        LTStart_max[j] = max(LTS[cols])
        LTEnd_min[j]   = min(LTE[cols])
        LTEnd_max[j]   = max(LTE[cols])
        # Similarly for UTStart, UTEnd

    # === MAIN KERNEL (FlashAttention-2 with block skipping) ===
    for i in range(T_r):  # PARALLEL over SMs
        Q_i = load_sram(Q[i*B_r:(i+1)*B_r])  # B_r x d
        O_i = zeros(B_r, d)   # accumulator in SRAM
        m_i = full(B_r, -inf) # row max in SRAM
        l_i = zeros(B_r)      # row sum in SRAM

        row_min = (i - 1) * B_r
        row_max = i * B_r

        for j in range(T_c):  # LOOP over KV tiles
            # === TILE-LEVEL BLOCK CLASSIFICATION ===
            # Lower triangle skip check
            if row_min >= LTStart_max[j] and row_max <= LTEnd_min[j]:
                continue  # FULLY MASKED — skip entirely!

            # Upper triangle skip check
            if row_min >= UTStart_max[j] and row_max <= UTEnd_min[j]:
                continue  # FULLY MASKED — skip entirely!

            # Load K, V tile from HBM -> SRAM
            K_j = load_sram(K[j*B_c:(j+1)*B_c])  # B_c x d
            V_j = load_sram(V[j*B_c:(j+1)*B_c])  # B_c x d

            # Compute attention tile
            S_ij = Q_i @ K_j.T  # B_r x B_c (in SRAM)

            # === PARTIAL MASK APPLICATION ===
            if row_max > LTStart_min[j] and row_min < LTEnd_max[j]:
                # Load column-wise mask vectors for this tile
                lts_j = load_sram(LTS[j*B_c:(j+1)*B_c])
                lte_j = load_sram(LTE[j*B_c:(j+1)*B_c])
                # Apply element-wise: S_ij[x,y] = -inf
                # where (i-1)*B_r + x >= lts_j[y] and (i-1)*B_r + x < lte_j[y]
                apply_mask_lower(S_ij, lts_j, lte_j, i, B_r)

            if row_max > UTStart_min[j] and row_min < UTEnd_max[j]:
                uts_j = load_sram(UTS[j*B_c:(j+1)*B_c])
                ute_j = load_sram(UTE[j*B_c:(j+1)*B_c])
                apply_mask_upper(S_ij, uts_j, ute_j, i, B_r)

            # Online softmax update
            m_new = max(m_i, rowmax(S_ij))
            P_ij = exp(S_ij - m_new)
            l_i = exp(m_i - m_new) * l_i + rowsum(P_ij)
            O_i = diag(exp(m_i - m_new)) @ O_i + P_ij @ V_j
            m_i = m_new

        # Normalize and write to HBM
        O_i = diag(l_i) ** -1 @ O_i
        write_hbm(O[i*B_r:(i+1)*B_r], O_i)

    return O
```

**GPU efficiency analysis:**

1. **Tile skip eliminates both compute AND HBM loads:** When a tile is fully masked, neither $K_j$, $V_j$ nor the mask vectors are loaded from HBM. This saves both memory bandwidth and compute cycles.

2. **Column-wise mask fits in SRAM:** The mask vectors $LTS_j, LTE_j$ for a single key tile are only $B_c$ elements each — easily preloaded into shared memory alongside $K_j$.

3. **No warp divergence for fully masked/unmasked tiles:** The classification check is per-tile (not per-element), so all threads in a warp take the same branch. Element-wise masking only occurs for partially masked tiles at the boundary.

4. **Linear relationship between sparsity and latency:** Figure 4(a) confirms that kernel execution time scales linearly with $(1-\rho)$, demonstrating that the block skip has near-zero overhead.

5. **Backward pass benefits from column-parallel structure:** The $dK$ and $dV$ computations are naturally column-parallel, matching the column-wise mask representation for efficient skip in the backward pass.

## References

- Wang, G., Zeng, J., Xiao, X., Wu, S., Yang, J., Zheng, L., Chen, Z., Bian, J., Yu, D., & Wang, H. (2025). FlashMask: Efficient and Rich Mask Extension of FlashAttention. ICLR 2025. arXiv:2410.01359.
- Dao, T. (2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR 2024.
- He, B., et al. (2024). FlexAttention: A flexible mask description method based on deep learning compiler techniques. PyTorch.
- Paglhardini, A., et al. (2023). Faster Causal Attention Over Large Sequences Through Sparse Flash Attention.
- Milakov, M. & Gimelshein, N. (2018). Online Normalizer Calculation for Softmax. arXiv:1805.02867.
