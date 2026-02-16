# 203: Gated DeltaNet — Chunkwise WY Kernel with Data-Dependent Gating

**Category**: parallelization
**Gain type**: efficiency
**Source**: Yang, Kautz & Hatamizadeh (2025) — MIT CSAIL / NVIDIA (ICLR 2025)
**Paper**: [papers/gated-deltanet-chunkwise-kernel.pdf]
**Documented**: 2026-02-15

## Description

Gated DeltaNet introduces the **gated delta rule** — a recurrence that unifies scalar gating (for memory erasure, as in Mamba2) with the delta update rule (for targeted key-value association replacement, as in DeltaNet). The key algorithmic contribution is a **hardware-efficient chunkwise training algorithm** that extends DeltaNet's WY-based parallelization to incorporate data-dependent gating while preserving the chunkwise-parallel structure.

**Why gating + delta rule matters:**

- **Mamba2** ($\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + v_t k_t^\top$): Uniform decay erases all associations equally. Cannot selectively forget specific key-value pairs.
- **DeltaNet** ($\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$): Targeted replacement of individual associations, but cannot rapidly clear the entire state when context changes.
- **Gated DeltaNet** ($\mathbf{S}_t = \alpha_t(\mathbf{S}_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$): Combines both — $\alpha_t \to 0$ clears memory; $\alpha_t \to 1$ preserves the delta rule behavior.

The chunkwise algorithm is the critical enabler: it converts the sequential recurrence into matmul-rich parallel computation within each chunk of size $C$, making it amenable to tensor core acceleration. The key challenge solved is incorporating the gating factor $\alpha_t$ into the WY representation for cumulative Householder-like products.

On H100, Gated DeltaNet at 1.3B parameters achieves ~45 Kt/s training throughput (comparable to DeltaNet), outperforms Mamba2 on language modeling (12.17 vs 12.56 ppl on LMB), and achieves the best performance among recurrent models on 8 of 9 evaluation benchmarks.

## Mathematical Form

**Gated Delta Rule (the recurrence):**

$$
\mathbf{S}_t = \alpha_t \left(\mathbf{S}_{t-1} \left(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top\right)\right) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top
$$

where:
- $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$ — the matrix-valued hidden state (fast weight matrix)
- $\alpha_t \in (0, 1)$ — scalar data-dependent gating term (controls global decay)
- $\beta_t \in (0, 1)$ — scalar writing strength (controls delta update intensity)
- $\mathbf{k}_t \in \mathbb{R}^{d_k}$ — key vector (L2-normalized)
- $\mathbf{v}_t \in \mathbb{R}^{d_v}$ — value vector
- $\mathbf{q}_t \in \mathbb{R}^{d_k}$ — query vector

The output is $\mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t$.

**Online learning interpretation:** The hidden state $\mathbf{S}$ is a fast weight matrix optimizing:

$$
\mathcal{L}(\mathbf{S}_t) = \frac{1}{2}\|\mathbf{S}_t - \alpha_t \mathbf{S}_{t-1}\|_F^2 - 2\langle \mathbf{S}_t \mathbf{k}_t, \beta_t(\mathbf{v}_t - \alpha_t \mathbf{S}_{t-1} \mathbf{k}_t) \rangle
$$

The update is one step of SGD on this objective with the gating term $\alpha_t$ acting as adaptive weight decay.

**Chunkwise Parallel Expansion:**

Splitting the sequence into chunks of size $C$, within chunk $[t]$, position $r$:

$$
\mathbf{S}_{[t]}^r = \mathbf{S}_{[t]} \underbrace{\left(\prod_{i=1}^{r} \alpha_{[t]}^i \left(\mathbf{I} - \beta_{[t]}^i \mathbf{k}_{[t]}^i \mathbf{k}_{[t]}^{i\top}\right)\right)}_{:= \mathbf{F}_{[t]}^r} + \underbrace{\sum_{i=1}^{r} \left(\beta_{[t]}^i \mathbf{v}_{[t]}^i \mathbf{k}_{[t]}^{i\top} \prod_{j=i+1}^{r} \alpha_{[t]}^j \left(\mathbf{I} - \beta_{[t]}^j \mathbf{k}_{[t]}^j \mathbf{k}_{[t]}^{j\top}\right)\right)}_{:= \mathbf{G}_{[t]}^r}
$$

where $\mathbf{F}_{[t]}^r$ is the cumulative transition product and $\mathbf{G}_{[t]}^r$ is the cumulative gated input.

**Extended WY Representation with Gating:**

The cumulative transition $\mathbf{F}_{[t]}^r$ involves products of generalized Householder matrices scaled by $\alpha_t$. Factoring out the cumulative decay $\gamma_{[t]}^r = \prod_{j=1}^{r} \alpha_{[t]}^j$:

$$
\mathbf{F}_{[t]}^r = \gamma_{[t]}^r \tilde{\mathbf{P}}_{[t]}^r, \quad \tilde{\mathbf{P}}_{[t]}^r = \mathbf{I} - \tilde{\mathbf{W}}_{[t]}^\top \mathbf{K}_{[t]}
$$

where $\tilde{\mathbf{W}}_{[t]}$ are the modified WY factors incorporating gating. Specifically:

$$
\tilde{\mathbf{w}}_{[t]}^r = \beta_{[t]}^r \left(\mathbf{k}_{[t]}^r - \sum_{i=1}^{r-1} \tilde{\mathbf{w}}_{[t]}^i \left(\frac{\gamma_{[t]}^i}{\gamma_{[t]}^r} \mathbf{k}_{[t]}^{i\top} \mathbf{k}_{[t]}^r\right)\right) \in \mathbb{R}^{d_k}
$$

Note the gating ratio $\gamma_{[t]}^i / \gamma_{[t]}^r$ that modifies the standard WY recurrence.

For the input accumulation $\mathbf{G}_{[t]}^r$:

$$
\tilde{\mathbf{u}}_{[t]}^r = \beta_{[t]}^r \left(\mathbf{v}_{[t]}^r - \sum_{i=1}^{r-1} \tilde{\mathbf{u}}_{[t]}^i \left(\frac{\gamma_{[t]}^i}{\gamma_{[t]}^r} \mathbf{k}_{[t]}^{i\top} \mathbf{k}_{[t]}^r\right)\right) \in \mathbb{R}^{d_v}
$$

**UT Transform (matrix form):**

$$
\widetilde{\mathbf{U}}_{[t]} = \left[\mathbf{I} + \text{strictLower}\left(\text{diag}(\beta_{[t]}) \left(\Gamma_{[t]} \odot \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top\right)\right)\right]^{-1} \text{diag}(\beta_{[t]}) \mathbf{V}_{[t]}
$$

where $\Gamma_{[t]}$ is the decay-aware causal mask with $(\Gamma_{[t]})_{ij} = \gamma_{[t]}^i / \gamma_{[t]}^j$ for $i \geq j$.

**Hardware-Efficient Chunkwise Algorithm:**

The inter-chunk state update and intra-chunk output computation become:

$$
\mathbf{S}_{[t+1]} = \overrightarrow{\mathbf{S}_{[t]}} + \left(\widetilde{\mathbf{U}_{[t]}} - \overleftarrow{\widetilde{\mathbf{W}}_{[t]}} \mathbf{S}_{[t]}^\top\right)^\top \overrightarrow{\mathbf{K}_{[t]}} \quad \in \mathbb{R}^{d_v \times d_k}
$$

$$
\mathbf{O}_{[t]} = \overleftarrow{\mathbf{Q}_{[t]}} \mathbf{S}_{[t]}^\top + \left(\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}\right) \left(\widetilde{\mathbf{U}_{[t]}} - \overleftarrow{\widetilde{\mathbf{W}}_{[t]}} \mathbf{S}_{[t]}^\top\right) \quad \in \mathbb{R}^{C \times d_v}
$$

where the arrows denote decay-to-start ($\leftarrow$) or decay-to-end ($\rightarrow$) operations:
- $\overleftarrow{\mathbf{q}_{[t]}^r} = \gamma_{[t]}^r \mathbf{q}_{[t]}^r$ (decay each query to chunk start)
- $\overrightarrow{\mathbf{k}_{[t]}^r} = \frac{\gamma_{[t]}^C}{\gamma_{[t]}^r} \mathbf{k}_{[t]}^r$ (decay each key to chunk end)
- $\overrightarrow{\mathbf{S}_{[t]}} = \gamma_{[t]}^C \mathbf{S}_{[t]}$ (decay state over entire chunk)

**Key Definitions:**

- $C$ — chunk size (typically 64)
- $d_k$ — key/query head dimension
- $d_v$ — value head dimension
- $\mathbf{S}_{[t]} \in \mathbb{R}^{d_v \times d_k}$ — inter-chunk hidden state at boundary $t$
- $\widetilde{\mathbf{W}}_{[t]} \in \mathbb{R}^{C \times d_k}$ — gated WY factors for transition
- $\widetilde{\mathbf{U}}_{[t]} \in \mathbb{R}^{C \times d_v}$ — gated WY factors for input
- $\Gamma_{[t]} \in \mathbb{R}^{C \times C}$ — decay-aware causal mask
- $\gamma_{[t]}^r = \prod_{j=1}^{r} \alpha_{[t]}^j$ — cumulative gating product within chunk

## Complexity

**Per-chunk computational cost:**

| Operation | Complexity | Dominant Matmul |
|-----------|-----------|----------------|
| UT transform (solve for $\widetilde{\mathbf{W}}, \widetilde{\mathbf{U}}$) | $O(C^2 d_k + C^2 d_v)$ | $\mathbf{K}\mathbf{K}^\top$ ($C \times d_k$ @ $d_k \times C$) |
| Inter-chunk output ($\overleftarrow{\mathbf{Q}} \mathbf{S}^\top$) | $O(C \cdot d_k \cdot d_v)$ | $C \times d_k$ @ $d_k \times d_v$ |
| Intra-chunk scores ($\mathbf{Q}\mathbf{K}^\top$) | $O(C^2 d_k)$ | $C \times d_k$ @ $d_k \times C$ |
| Intra-chunk output (scores @ $\widetilde{\mathbf{U}}$) | $O(C^2 d_v)$ | $C \times C$ @ $C \times d_v$ |
| State update ($\widetilde{\mathbf{U}}^\top \overrightarrow{\mathbf{K}}$) | $O(C \cdot d_k \cdot d_v)$ | $d_v \times C$ @ $C \times d_k$ |
| **Total per chunk** | $O(C^2(d_k + d_v) + C \cdot d_k \cdot d_v)$ | |
| **Total for sequence $T$** | $O\left(\frac{T}{C}\left(C^2(d_k+d_v) + C \cdot d_k \cdot d_v\right)\right)$ | |

Simplified: $O(T \cdot C \cdot (d_k + d_v) + T \cdot d_k \cdot d_v)$ where the first term is quadratic intra-chunk and the second is the linear recurrence cost.

**Comparison with Mamba2/GLA at same chunk size $C$:**

| Method | Intra-chunk | Transition structure | Extra cost vs GLA |
|--------|------------|---------------------|-------------------|
| GLA / Mamba2 | $O(C^2(d_k+d_v))$ | Diagonal ($\alpha_t$) | — |
| DeltaNet | $O(C^2(d_k+d_v))$ | Householder ($\mathbf{I}-\beta_t kk^\top$) | UT transform: $+O(C^2 d_k)$ |
| **Gated DeltaNet** | $O(C^2(d_k+d_v))$ | Gated Householder ($\alpha_t(\mathbf{I}-\beta_t kk^\top)$) | UT transform: $+O(C^2 d_k)$ |

The gating adds **negligible overhead** over DeltaNet — only scaling operations in the UT transform by decay ratios $\gamma_i/\gamma_j$, which are elementwise multiplies.

**Wall-clock throughput (H100, 1.3B models, Fig. 3):**

| Seq × Batch | DeltaNet (Kt/s) | Gated DeltaNet (Kt/s) | Mamba2 (Kt/s) | Transformer++ (Kt/s) |
|-------------|----------------|----------------------|--------------|---------------------|
| 2K × 16 | ~42 | ~42 | ~40 | ~55 |
| 4K × 8 | ~48 | ~48 | ~53 | ~52 |
| 8K × 4 | ~47 | ~47 | ~50 | ~45 |
| 16K × 2 | ~45 | ~45 | ~50 | ~35 |

Gated DeltaNet matches DeltaNet throughput exactly (the gating adds no measurable overhead). Both are slightly slower than Mamba2 due to the more complex Householder transition.

## Applicability

- **Gated DeltaNet (primary):** The target architecture. The chunkwise algorithm is implemented in the FLA library at `fla/ops/generalized_delta_rule`, enabling direct use with the GatedDeltaNet model.

- **DeltaNet (subset):** Setting $\alpha_t = 1$ recovers standard DeltaNet. The algorithm generalizes the original DeltaNet chunkwise kernel (Yang et al., 2024b).

- **DeltaProduct (extension):** For multi-step Householder transitions $\prod_h (\mathbf{I} - \beta_t^h k_t^h k_t^{h\top})$, the WY+gating approach can be extended by applying the gated WY representation to each Householder factor.

- **Hybrid architectures:** The paper demonstrates GatedDeltaNet-H1 (Gated DeltaNet + sliding window attention) and GatedDeltaNet-H2 (Mamba2 + Gated DeltaNet + SWA), achieving the best overall performance. The chunkwise kernel enables efficient interleaving with attention layers.

- **TFLA integration (potential):** TFLA's two-level tiling could be applied to the Gated DeltaNet chunkwise kernel. The outer chunk size $L$ would be large (e.g., 1024), with inner tiles of size $B_{Lhq} \times B_{Lkv}$ for the intra-chunk matmuls. The UT transform would need to be computed at the outer chunk level, which becomes more expensive at larger $L$ ($O(L^2 d_k)$). The gating ratios $\gamma_i/\gamma_j$ are computed in log-space for numerical stability.

## Limitations

- **UT transform is sequential within a chunk:** The forward substitution to solve the lower-triangular system for $\widetilde{\mathbf{W}}$ and $\widetilde{\mathbf{U}}$ is inherently sequential along the chunk dimension. For chunk size $C = 64$ this is not a bottleneck, but scaling to larger chunks (as TFLA enables) increases the cost quadratically.

- **Slightly slower than Mamba2:** The Householder transition $(\mathbf{I} - \beta_t k_t k_t^\top)$ is more expensive than Mamba2's diagonal transition ($\alpha_t \mathbf{I}$). The UT transform adds ~$O(C^2 d_k)$ FLOPs per chunk that Mamba2 doesn't need.

- **Head dimension constraint:** The current FLA implementation uses $d_k = d_v = d_{\text{model}} / n_{\text{heads}}$ (e.g., 128 for 4096-dim model with 32 heads). The matrix state $\mathbf{S} \in \mathbb{R}^{d_v \times d_k}$ consumes $d_v \times d_k$ memory per head per chunk, which limits scaling to very large head dimensions.

- **L2-normalized keys required:** The delta rule requires $\|\mathbf{k}\|_2 = 1$ for the Householder structure $(\mathbf{I} - \beta k k^\top)$ to be well-conditioned. This constrains key representations and requires a normalization step.

- **Retrieval still inferior to attention:** Despite improvement over Mamba2, Gated DeltaNet still underperforms Transformer++ on retrieval-heavy tasks (SWDE: 25.4 vs 29.5, FDA: 23.7 vs 52.2). The fixed-size state $\mathbf{S}$ cannot match attention's unbounded context.

## Implementation Notes

```python
# Gated DeltaNet chunkwise forward pass pseudocode
# Based on Equations 8-9 and the gated UT transform from Section 3.3

def gated_deltanet_chunk_forward(Q, K, V, alpha, beta, C):
    """
    Chunkwise parallel forward pass for Gated DeltaNet.

    Args:
        Q: (T, d_k) - queries
        K: (T, d_k) - keys (L2-normalized)
        V: (T, d_v) - values
        alpha: (T,) - gating factors in (0, 1)
        beta: (T,) - writing strengths in (0, 1)
        C: int - chunk size (typically 64)
    """
    T, d_k = Q.shape
    d_v = V.shape[1]
    N_c = T // C

    # Reshape into chunks
    Q_c = Q.reshape(N_c, C, d_k)
    K_c = K.reshape(N_c, C, d_k)
    V_c = V.reshape(N_c, C, d_v)
    alpha_c = alpha.reshape(N_c, C)
    beta_c = beta.reshape(N_c, C)

    S = torch.zeros(d_v, d_k)  # inter-chunk state
    O = torch.empty(N_c, C, d_v)

    for t in range(N_c):
        # --- Compute cumulative gating products (log-space for stability) ---
        log_gamma = torch.cumsum(torch.log(alpha_c[t]), dim=0)  # (C,)
        gamma = torch.exp(log_gamma)

        # Decay-aware causal mask: Gamma[i,j] = gamma[i] / gamma[j] for i >= j
        Gamma = torch.exp(log_gamma.unsqueeze(1) - log_gamma.unsqueeze(0))
        Gamma = torch.tril(Gamma)  # (C, C)

        # --- Gated UT Transform (solve for W_tilde, U_tilde) ---
        # T_mat = [I + strictLower(diag(beta) @ (Gamma ⊙ K K^T))]
        KKt = K_c[t] @ K_c[t].T  # (C, C) — TENSOR CORE matmul
        T_lower = torch.tril(torch.diag(beta_c[t]) @ (Gamma * KKt), diagonal=-1)
        T_mat = torch.eye(C) + T_lower  # (C, C) lower triangular

        # Forward substitution: W_tilde = T_mat^{-1} @ diag(beta) @ K
        # U_tilde = T_mat^{-1} @ diag(beta) @ V
        beta_diag = beta_c[t].unsqueeze(1)  # (C, 1)
        W_tilde = torch.linalg.solve_triangular(
            T_mat, beta_diag * K_c[t], upper=False)  # (C, d_k)
        U_tilde = torch.linalg.solve_triangular(
            T_mat, beta_diag * V_c[t], upper=False)  # (C, d_v)

        # --- Decay operations ---
        gamma_C = gamma[-1]  # total decay over chunk

        # Decay queries to chunk start
        Q_left = gamma.unsqueeze(1) * Q_c[t]  # (C, d_k)
        # Decay keys to chunk end
        K_right = (gamma_C / gamma).unsqueeze(1) * K_c[t]  # (C, d_k)
        # Decay state over chunk
        S_right = gamma_C * S  # (d_v, d_k)

        # --- Inter-chunk output: Q_left @ S^T ---
        # (C, d_k) @ (d_k, d_v) → (C, d_v) — TENSOR CORE matmul
        O_inter = Q_left @ S.T

        # --- Intra-chunk scores: Q @ K^T ⊙ M ---
        # (C, d_k) @ (d_k, C) → (C, C) — TENSOR CORE matmul
        scores = Q_c[t] @ K_c[t].T  # raw scores
        M = torch.tril(Gamma)  # causal + decay mask
        scores = scores * M

        # --- Intra-chunk output: scores @ (U_tilde - W_tilde @ S^T) ---
        # First: W_tilde @ S^T — (C, d_k) @ (d_k, d_v) → (C, d_v)
        WS = W_tilde @ S.T  # correction for inter-chunk state
        # Then: scores @ (U_tilde - WS) — (C, C) @ (C, d_v) → (C, d_v)
        O_intra = scores @ (U_tilde - WS)  # TENSOR CORE matmul

        O[t] = O_inter + O_intra

        # --- State update ---
        # S_{t+1} = gamma_C * S + (U_tilde - W_tilde @ S^T)^T @ K_right
        delta = U_tilde - WS  # (C, d_v) — what's new in this chunk
        # (d_v, C) @ (C, d_k) → (d_v, d_k) — TENSOR CORE matmul
        S = S_right + delta.T @ K_right

    return O.reshape(T, d_v)
```

**GPU efficiency analysis:**

1. **All dominant operations are matmuls:** $KK^\top$, $QK^\top$, $Q\mathbf{S}^\top$, scores$\cdot$$(U-WS^\top)$, $(U-WS)^\top K$ — all map to WGMMA tensor core instructions
2. **Memory access pattern:** Within each chunk, Q, K, V tiles are loaded once from HBM to SMEM, then reused across multiple matmuls. The UT transform's forward substitution is sequential but operates entirely in SMEM/registers
3. **Arithmetic intensity:** At chunk size $C=64$, $d_k=d_v=128$: the dominant matmuls are $64 \times 128$ @ $128 \times 64$, giving $2 \times 64 \times 128 \times 64 = 1M$ FLOPs per tile pair, with data size ~$64 \times 128 \times 2 = 16$KB per operand — arithmetic intensity ~60 FLOPs/byte (compute-bound)
4. **No irregular memory access:** All operations are structured matmuls + elementwise ops; no gather/scatter, no permutations
5. **Code:** https://github.com/NVlabs/GatedDeltaNet (model) and https://github.com/fla-org/flash-linear-attention (kernels at `fla/ops/generalized_delta_rule`)

## References

- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. arXiv:2412.06464.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024b). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 2024. arXiv:2406.06484.
- Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024.
- Bischof, C. H. & Van Loan, C. (1985). The WY Representation for Products of Householder Matrices. SIAM J. Sci. Stat. Comput.
- Joffrain, T., et al. (2006). Accumulating Householder Transformations, Revisited. ACM Trans. Math. Softw.
- Beck, M., et al. (2025). Tiled Flash Linear Attention. arXiv:2503.14376 (TFLA, trick 158).
- Code: https://github.com/NVlabs/GatedDeltaNet, https://github.com/fla-org/flash-linear-attention
