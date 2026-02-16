# 218: Sliding Window Recurrence — Phalanx Block Two-Pass (B2P)

**Category**: parallelization
**Gain type**: efficiency
**Source**: Secrieru, Brixi, Bengio, Suzuki, Poli & Massaroli, "Sliding Window Recurrences for Sequence Models" (2025)
**Paper**: papers/sliding-window-recurrences-phalanx.pdf
**Documented**: 2026-02-15

## Description

Sliding Window Recurrences (SWR) truncate the global range of linear recurrences to hardware-aligned local windows, eliminating global synchronization while preserving the essential local dynamics. The key insight is that stable linear recurrences (with contraction factor $|a_i| \leq \rho < 1$) have transfer operators whose entries decay exponentially with distance, so long-range dependencies contribute negligibly — and can be safely dropped without quality loss.

The paper introduces a **hierarchical decomposition framework** that views the transfer operator $\boldsymbol{L}$ of a linear recurrence as a block lower-triangular matrix, factoring it into independent **intra-block solves** (diagonal blocks $\boldsymbol{L}_t$) and **inter-block coupling** mediated by a scalar carrier. The SWR approximation aggressively truncates the carrier transfer operator $\boldsymbol{T} \approx \boldsymbol{I}_b$, retaining only nearest-neighbor block interactions. This yields a **block-bidiagonal** structure (the "jagged window") that is computed via the **Block Two-Pass (B2P)** algorithm: (1) parallel local GEMM solves within warp-sized blocks, then (2) a single nearest-neighbor rank-1 update — achieving $O(1)$ depth, $O(n)$ work, and purely local communication.

The resulting **Phalanx layer** serves as a drop-in replacement for sliding window attention or linear recurrences in multi-hybrid architectures, training **24% faster** than Transformer++ and **10% faster** than SWA hybrids at 8K context length on H100 GPUs, while matching perplexity at 1.3B parameters / 100B tokens.

## Mathematical Form

**Scalar linear recurrence:**

$$
x_i = a_i x_{i-1} + u_i, \quad i \in [n]
$$

with coefficients $a_i \in \mathbb{R}$, inputs $u_i \in \mathbb{R}$, initial condition $x_0 \in \mathbb{R}$.

**Transfer operator (matrix form):**

The recurrence is equivalent to solving $(I - AZ)x = u$, where $A = \text{diag}(a_1, \ldots, a_n)$ and $Z$ is the down-shift operator ($Z_{i,j} = \delta_{i,j+1}$). The transfer operator is:

$$
\boldsymbol{L} = (\boldsymbol{I} - \boldsymbol{A}\boldsymbol{Z})^{-1} = \boldsymbol{I} + \boldsymbol{A}\boldsymbol{Z} + (\boldsymbol{A}\boldsymbol{Z})^2 + \cdots + (\boldsymbol{A}\boldsymbol{Z})^{n-1}
$$

with entry $\boldsymbol{L}_{ij} = a_{i:j+1}$ for $i \geq j$ (product of coefficients), zero otherwise.

**Hierarchical decomposition (Theorem 3.3):**

Partition into $b$ blocks of size $\ell$ ($n = b\ell$). The transfer operator admits:

$$
\boldsymbol{L} = \boldsymbol{\mathcal{L}} + \boldsymbol{G}\boldsymbol{Z}_b\boldsymbol{T}\boldsymbol{R}
$$

where:
- $\boldsymbol{\mathcal{L}} = \text{diag}(\boldsymbol{L}_1, \ldots, \boldsymbol{L}_b)$ — block-diagonal local solves, $\boldsymbol{L}_t = (\boldsymbol{I}_\ell - \boldsymbol{A}_t\boldsymbol{Z}_\ell)^{-1}$
- $\boldsymbol{T} = (\boldsymbol{I}_b - \boldsymbol{C}\boldsymbol{Z}_b)^{-1}$ — carrier transfer operator, $\boldsymbol{C} = \text{diag}(c_1, \ldots, c_b)$, $c_t = a_{t,\ell:1}$ (compound attenuation)
- $\boldsymbol{G}$ — propagation factors broadcasting carrier into blocks
- $\boldsymbol{R}$ — extraction factors reading local contribution to carrier

**Jagged window (SWR) approximation:**

Truncate $\boldsymbol{T} \approx \boldsymbol{I}_b$, retaining only nearest-neighbor coupling:

$$
\tilde{\boldsymbol{L}} = \boldsymbol{\mathcal{L}} + \boldsymbol{G}\boldsymbol{Z}_b\boldsymbol{R}
$$

This is a block-bidiagonal matrix with diagonal blocks $\boldsymbol{L}_t \in \mathbb{R}^{\ell \times \ell}$ and rank-one off-diagonal blocks $\boldsymbol{F}_{t,t-1} = \boldsymbol{g}_t \boldsymbol{r}_{t-1}^\top$.

**Block Two-Pass (B2P) algorithm:**

- **Pass 1** (parallel over all blocks): Materialize $\boldsymbol{L}_t$, compute $\boldsymbol{w}_t = \boldsymbol{L}_t \boldsymbol{u}_t$ via GEMM, extract carrier $v_t = \boldsymbol{w}_{t,\ell}$
- **Pass 2** (parallel with neighbor shift): Reconstruct $\hat{\boldsymbol{x}}_t = \boldsymbol{w}_t + \boldsymbol{g}_t v_{t-1}$ via rank-1 update

**Linear-space materialization of $\boldsymbol{L}_t$:**

$$
\boldsymbol{L}_t = \text{tril}(\boldsymbol{g}_t \boldsymbol{g}_t^{-\top})
$$

Computed stably via column-wise cumulative product in linear space (avoiding log/exp):

$$
A \leftarrow \boldsymbol{a}_t \otimes \boldsymbol{1}, \quad A_{ij} \leftarrow 1 \text{ for } i \leq j, \quad P \leftarrow \text{CumProd}_j(A), \quad \boldsymbol{L}_t \leftarrow \text{tril}(P)
$$

**Computational horizon (error bound):**

For contraction factor $\rho$, the truncation error's tail sum satisfies:

$$
\sum_{\ell=k+1}^{\infty} \rho^\ell = \frac{\rho^{k+1}}{1-\rho} < \varepsilon
$$

yielding effective bandwidth $k = \lceil \log(\varepsilon(1-\rho)) / \log\rho \rceil - 1$.

## Complexity

| Operation | Full Recurrence (scan) | SWR with B2P |
|-----------|----------------------|--------------|
| Work | $O(nd)$ (sequential scan) | $O(b\ell^2 d + b\ell d) = O(nd)$ |
| Depth | $O(n)$ sequential or $O(\log n)$ scan | $O(1)$ (constant) |
| Communication | Global (all blocks) | Local (nearest neighbor only) |
| Kernel launches | $O(1)$ scan or $O(\log n)$ flat | $O(1)$ (single persistent kernel) |

**Memory:** $O(n d)$ for inputs/outputs + $O(b \ell^2)$ for materialized $\boldsymbol{L}_t$ tiles (in SRAM).

**Arithmetic intensity:** High — local solves are $\ell \times \ell$ by $\ell \times d$ GEMMs via tensor cores (wmma), amortizing $O(\ell^2)$ tile materialization across $d$ feature channels.

**Hardware mapping ($\ell = 16$, H100):**
- 1 warp (32 threads) per block → wmma $16 \times 16 \times 16$ tiles
- 1 CTA (32 warps) → $32 \times 16 = 512$ time steps, no global sync
- 1 cluster (16 CTAs via DSMEM) → up to 8192 time steps with purely local communication

## Applicability

- **Multi-hybrid architectures**: Designed as local token mixer paired with global attention layers. Phalanx–Attention hybrids match SWA–Attention quality.
- **Drop-in replacement for**: Sliding window attention, gated short convolutions, or linear recurrences in the "local" slots of hybrid models.
- **Long-context training**: Linear scaling in sequence length. Faster than Flash Attention 3 and Mamba-2 SSD at all sequence lengths from 1K to 512K (forward pass).
- **Inference**: Can decode in $O(1)$ per token using recurrence mode with a single scalar carry state per head.
- **1B+ parameter scale**: Validated at 1.3B parameters, 100B tokens on FineWeb-Edu.

## Limitations

- **No global context**: By design, SWR truncates long-range dependencies. Requires pairing with global attention for tasks needing full-context access (e.g., retrieval, long-range reasoning).
- **Scalar recurrence only**: Current B2P kernel handles scalar coefficients $a_i$ per head. Matrix-valued transitions (as in GLA, DeltaNet) would need adaptation to the rank-1 inter-block coupling structure.
- **Quality depends on decay rate**: If the learned decay rates approach 1 (slow decay), the jagged window approximation error grows. The paper bounds decay to $[0, 0.8]$ via sigmoid scaling.
- **Block size fixed at 16**: Tightly coupled to wmma tile size. Different hardware (e.g., AMD, TPU) would need different block sizes.
- **Not a standalone model**: Best results require hybrid architecture with attention; Phalanx alone would lack global reasoning ability.

## Implementation Notes

```python
# Block Two-Pass (B2P) — GPU Kernel Pseudocode
# Each warp processes one time block of size ℓ=16

def b2p_kernel(u, a, output):
    """
    u: [n, d] inputs (n = b*ℓ)
    a: [n] recurrence coefficients (sigmoid-bounded)
    output: [n, d] states
    """
    block_id = warp_id  # one warp per time block

    # === Pass 1: Parallel local solves ===
    # Materialize L_t in registers via cumulative product
    a_block = a[block_id * ℓ : (block_id+1) * ℓ]  # [ℓ]
    L_t = linear_space_cumprod_tril(a_block)  # [ℓ, ℓ] in registers

    # Compute propagation vector g_t (first column of L_t)
    g_t = L_t[:, 0]  # [ℓ]

    # Load input block
    u_t = load_smem(u[block_id * ℓ : (block_id+1) * ℓ])  # [ℓ, d]

    # Local solve via wmma tensor core GEMM
    w_t = wmma_matmul(L_t, u_t)  # [ℓ, d], using 16x16x16 tiles

    # Extract carrier: last element of local solve
    v_t = w_t[ℓ-1, :]  # [d] — effective input for next block

    # Write carrier to SMEM/DSMEM for neighbor
    write_carrier(v_t)

    # === Pass 2: Rank-1 reconstruction ===
    sync_warps()  # single thread-block sync

    if block_id > 0:
        v_prev = read_carrier(block_id - 1)  # [d] from neighbor
        # Rank-1 update: inject neighbor contribution
        output_t = w_t + outer(g_t, v_prev)  # g_t * v_prev^T
    else:
        output_t = w_t

    write_global(output_t)
```

## References

- Secrieru, Brixi, Bengio, Suzuki, Poli & Massaroli. "Sliding Window Recurrences for Sequence Models." arXiv:2512.13921, 2025.
- Blog: https://www.radicalnumerics.ai/blog/phalanx
- Dao & Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024.
- Yang, Song, Bakhtiari, Rajbhandari & Wang. "Gated Linear Attention Transformers with Hardware-Efficient Training." ICML 2024.
- Merrill & Garland. "Single-pass Parallel Prefix Scan with Decoupled Look-back." 2016.
- Blelloch. "Prefix Sums and Their Applications." 1990.
