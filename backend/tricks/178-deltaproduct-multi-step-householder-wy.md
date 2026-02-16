# 178: DeltaProduct — Multi-Step Householder Product State Transitions

**Category**: decomposition
**Gain type**: expressivity
**Source**: Siems, Carstensen, Zela, Hutter, Pontil & Grazzi (NeurIPS 2025)
**Paper**: [papers/deltaproduct-householder-products.pdf]
**Documented**: 2026-02-15

## Description

DeltaProduct extends DeltaNet by taking $n_h$ gradient descent steps per token (instead of 1), yielding state-transition matrices that are **products of $n_h$ generalized Householder transformations**. The result is a diagonal-plus-rank-$n_h$ matrix $A(\boldsymbol{x}_i) = I + \text{low-rank}$, which interpolates smoothly between DeltaNet ($n_h = 1$, rank-1 update) and full dense matrices ($n_h = n$). This provides a **tunable expressivity–efficiency knob** without changing the chunkwise parallel training algorithm: each step of DeltaProduct reuses the same UT transform and WY representation that DeltaNet uses, applied $n_h$ times sequentially within the recurrence.

The key insight is that a product of generalized Householder reflections $\prod_{j=1}^{n_h}(I - \beta_j \boldsymbol{k}_j \boldsymbol{k}_j^\top)$ can represent any orthogonal matrix (by the Cartan-Dieudonné theorem) when $n_h = n$ and $\beta_j = 2$, yet maintains spectral norm $\leq 1$ for stability since $\|A(\boldsymbol{x}_i)\| \leq \prod_j \|I - \beta_j \boldsymbol{k}_j \boldsymbol{k}_j^\top\| \leq 1$ when each $\beta_j \in [0, 2]$. In contrast, a naive rank-$n_h$ parameterization $I - \sum_j \beta_j \boldsymbol{k}_j \boldsymbol{k}_j^\top$ would be symmetric and could not represent rotations, and would lack the spectral norm guarantee.

**Why this matters for pretraining:** DeltaProduct achieves better language modeling perplexity and dramatically improved length extrapolation compared to DeltaNet, while using the same Triton kernel infrastructure (flash-linear-attention library). Training throughput scales linearly with $n_h$ (the recurrence is $n_h$ times longer), but parameter-matched comparisons (scaling $n_h$ vs. head dimension) show DeltaProduct scales more favorably.

## Mathematical Form

**DeltaNet Recurrence (Single Step):**

$$
\boldsymbol{H}_i = (I - \beta_i \boldsymbol{k}_i \boldsymbol{k}_i^\top) \boldsymbol{H}_{i-1} + \beta_i \boldsymbol{k}_i \boldsymbol{v}_i^\top
$$

This is one step of online gradient descent on the loss $\mathcal{L}_i(\boldsymbol{H}) = \frac{1}{2}\|\boldsymbol{H}^\top \boldsymbol{k}_i - \boldsymbol{v}_i\|_2^2$.

**DeltaProduct Recurrence ($n_h$ Steps):**

For each input token $\boldsymbol{x}_i$, generate $n_h$ keys $\boldsymbol{k}_{i,j}$, values $\boldsymbol{v}_{i,j}$, and learning rates $\beta_{i,j}$ for $j = 1, \ldots, n_h$. Apply $n_h$ gradient descent steps:

$$
\boldsymbol{H}_{i,j} = (I - \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{k}_{i,j}^\top) \boldsymbol{H}_{i,j-1} + \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{v}_{i,j}^\top
$$

where $\boldsymbol{H}_{i,0} = \boldsymbol{H}_{i-1}$ and $\boldsymbol{H}_{i,n_h} = \boldsymbol{H}_i$.

**State-Transition Matrix:**

$$
A(\boldsymbol{x}_i) = \prod_{j=1}^{n_h} \left(I - \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{k}_{i,j}^\top\right)
$$

**Input Matrix:**

$$
B(\boldsymbol{x}_i) = \sum_{j=1}^{n_h} \left(\prod_{k=j+1}^{n_h} (I - \beta_{i,k} \boldsymbol{k}_{i,k} \boldsymbol{k}_{i,k}^\top)\right) \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{v}_{i,j}^\top
$$

**Gated DeltaProduct:**

$$
A(\boldsymbol{x}_i) = g_i \prod_{j=1}^{n_h} (I - \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{k}_{i,j}^\top)
$$

where $g_i \in [0, 1]$ is a scalar forget gate (adopted from Mamba-2/GLA).

**Key Definitions:**

- $\boldsymbol{H}_i \in \mathbb{R}^{n \times d}$ — hidden state matrix at time $i$
- $n_h$ — number of Householder steps per token (the expressivity parameter)
- $\boldsymbol{k}_{i,j} \in \mathbb{R}^n$, $\|\boldsymbol{k}_{i,j}\| = 1$ — unit key vector for step $j$ of token $i$
- $\boldsymbol{v}_{i,j} \in \mathbb{R}^d$ — value vector for step $j$ of token $i$
- $\beta_{i,j} \in [0, 2]$ — learning rate / reflection coefficient for step $j$

**Spectral Properties of the Householder Product (Proposition 1):**

1. **Identical keys:** If $\boldsymbol{k}_j = \boldsymbol{k}$ for all $j$, then $\prod_j (I - \beta_j \boldsymbol{k}\bk^\top) = I - \beta^* \boldsymbol{k}\bk^\top$ for some scalar $\beta^*$. The product collapses to a single Householder — no added expressivity.

2. **Orthogonal keys:** If $\boldsymbol{k}_j^\top \boldsymbol{k}_l = \delta_{jl}$, the factors commute and $A = I - \sum_j \beta_j \boldsymbol{k}_j \boldsymbol{k}_j^\top$. The matrix is symmetric with real eigenvalues — no rotations possible.

3. **Non-orthogonal keys with $\beta > 1$:** When keys are non-trivially linearly dependent and $\beta_1, \beta_2 > 1$, the product can have **complex eigenvalues**, enabling representation of **rotations**. This is the key expressivity gain over DeltaNet.

**Chunkwise Training:**

DeltaProduct reuses DeltaNet's chunkwise parallel algorithm. The $n_h$ steps per token are "flattened" into the sequence: keys are arranged as $[\boldsymbol{k}_{1,1}, \ldots, \boldsymbol{k}_{1,n_h}, \boldsymbol{k}_{2,1}, \ldots, \boldsymbol{k}_{2,n_h}, \ldots]$, and the gates are constructed as $[g_1, 1, \ldots, 1, g_2, 1, \ldots, 1, \ldots]$ (one gate per token followed by $n_h - 1$ ones). The chunkwise parallel form with UT transform then applies unchanged to this extended sequence of length $n_h \cdot T$.

## Complexity

| Operation | DeltaNet ($n_h = 1$) | DeltaProduct ($n_h$) |
|-----------|---------------------|---------------------|
| Recurrence length | $T$ | $n_h \cdot T$ |
| State rank per token | rank-1 | rank-$n_h$ |
| Per-token FLOPs (recurrence) | $O(nd)$ | $O(n_h \cdot nd)$ |
| UT transform (per chunk) | $O(C^2 d)$ | $O((n_h C)^2 d)$ |
| Inter-chunk state | $O(nd)$ | $O(nd)$ (unchanged) |
| Parameters per head | $3nd + n$ | $(2n_h + 1)nd + n_h \cdot n$ |
| Expressivity | Reflections only ($S_2$) | Rotations + reflections ($O(n_h)$) |

**Training throughput (1.3B parameters, H100, from Figure 4):**

| Config | Tokens/sec |
|--------|-----------|
| DeltaNet (8 heads, $d_h = 256$) | ~40K |
| DeltaProduct$_2$ (8 heads, $d_h = 256$) | ~25K |
| DeltaProduct$_3$ (8 heads, $d_h = 256$) | ~18K |
| DeltaProduct$_2$ (16 heads, $d_h = 128$) | ~35K |
| DeltaProduct$_3$ (24 heads, $d_h = 85$) | ~30K |

When parameter-matching by scaling the number of heads (rather than head dimension), DeltaProduct$_2$ with 16 heads achieves comparable throughput to DeltaNet with 8 heads while providing superior quality.

**Memory:** Same as DeltaNet — $O(nd)$ inter-chunk state. The intra-chunk WY representation grows as $O(n_h C \cdot d)$ but this is within the chunk and doesn't affect peak memory for the $d \times d$ state materialization.

## Applicability

- **Language modeling with linear RNNs:** DeltaProduct directly replaces DeltaNet in the Gated DeltaNet architecture (Mamba-2 hybrid). At 1.3B parameters trained on FineWeb, DeltaProduct$_3[-1,1]$ achieves comparable performance to Gated DeltaNet$[-1,1]$ **without a forget gate**, showing that the multi-step Householder product subsumes some of the gate's forgetting functionality.

- **State-tracking tasks:** DeltaProduct solves permutation group word problems ($S_3, S_4, A_5, S_5$) that DeltaNet cannot solve in a single layer. With $n_h = 2$ (two Householder steps), a single-layer model can learn the $S_3$ group; with $n_h = 4$, a single layer learns $S_5$. DeltaNet requires multiple layers for these tasks.

- **Length extrapolation:** DeltaProduct shows significantly better length extrapolation on code generation (CodeParrot) and math (OpenThoughts-114K). The effective rank of the hidden state stabilizes with higher $n_h$, preventing the rank explosion that causes DeltaNet's extrapolation failure.

- **Scaling:** DeltaProduct scales favorably — increasing $n_h$ provides more consistent quality improvements than increasing head dimension. This makes it an attractive knob for scaling linear RNN architectures.

## Limitations

- **Linear cost scaling with $n_h$:** Training throughput decreases linearly with $n_h$ since the recurrence is $n_h$ times longer. At $n_h = 3$, throughput is roughly 45% of DeltaNet.

- **Head dimension interaction:** When matching parameters by scaling head dimension down (rather than adding heads), non-power-of-2 head dimensions waste GPU resources due to padding. Scaling by number of heads avoids this issue.

- **No custom kernel:** DeltaProduct reuses DeltaNet's Triton kernel by flattening the multi-step sequence. A custom kernel that fuses the $n_h$ steps within a single chunk could potentially be faster (e.g., by avoiding the overhead of constructing the extended key/value/gate sequences).

- **Stability constraint limits expressivity:** The spectral norm bound $\|A\| \leq 1$ ensures stability but prevents representing matrices with spectral radius $> 1$. RWKV-7 relaxes this constraint (allowing $\|A\| = \sqrt{2}$ for copy matrices) at the cost of potential instability.

- **Autoregressive generation not parallelized:** While $n_h \geq 1$ could parallelize the per-token recurrence steps during autoregressive generation (since each Householder step is independent), this is not yet implemented.

## Implementation Notes

```python
import torch

def deltaproduct_recurrence(H_prev, keys, values, betas, gate=None):
    """
    DeltaProduct: n_h gradient descent steps per token.

    Args:
        H_prev: (n, d) - previous hidden state matrix
        keys: (n_h, n) - unit key vectors for this token
        values: (n_h, d) - value vectors for this token
        betas: (n_h,) - learning rates in [0, 2]
        gate: scalar in [0, 1] or None - forget gate

    Returns:
        H: (n, d) - updated hidden state
    """
    n_h = keys.shape[0]
    H = H_prev.clone()

    for j in range(n_h):
        k_j = keys[j]   # (n,)
        v_j = values[j]  # (d,)
        beta_j = betas[j] # scalar

        # Generalized Householder update (one gradient step)
        # H = (I - beta * k k^T) H + beta * k v^T
        kTH = k_j @ H  # (d,)  — k^T H
        H = H - beta_j * torch.outer(k_j, kTH) + beta_j * torch.outer(k_j, v_j)

    if gate is not None:
        # Gated version: apply forget gate to transition only
        # H = g * A(x) * H_prev + B(x)
        # But since we've already computed the full update,
        # we interpolate: H_gated = gate * H + (1 - gate) * B_contribution
        # In practice, gate is folded into the flattened sequence
        pass

    return H


def deltaproduct_flatten_for_chunkwise(keys, values, betas, gates, n_h):
    """
    Flatten n_h steps per token into an extended sequence
    for reuse with DeltaNet's chunkwise parallel algorithm.

    Args:
        keys: (T, n_h, n) - keys for all tokens
        values: (T, n_h, d) - values for all tokens
        betas: (T, n_h) - learning rates
        gates: (T,) - per-token forget gates

    Returns:
        flat_keys: (T * n_h, n) - flattened keys
        flat_values: (T * n_h, d) - flattened values
        flat_betas: (T * n_h,) - flattened betas
        flat_gates: (T * n_h,) - flattened gates
    """
    T, _, n = keys.shape
    d = values.shape[2]

    # Flatten keys, values, betas
    flat_keys = keys.reshape(T * n_h, n)      # (T*n_h, n)
    flat_values = values.reshape(T * n_h, d)   # (T*n_h, d)
    flat_betas = betas.reshape(T * n_h)        # (T*n_h,)

    # Gate pattern: [g_1, 1, ..., 1, g_2, 1, ..., 1, ...]
    # Only the first step of each token gets the actual gate
    flat_gates = torch.ones(T * n_h, device=gates.device)
    flat_gates[::n_h] = gates  # Gate at positions 0, n_h, 2*n_h, ...

    return flat_keys, flat_values, flat_betas, flat_gates


def deltaproduct_chunkwise_training(Q, K, V, beta, gates, n_h, chunk_size):
    """
    Full DeltaProduct training via flattened chunkwise parallel form.

    1. Generate n_h keys/values/betas per token
    2. Flatten into extended sequence of length T * n_h
    3. Run DeltaNet's chunkwise parallel algorithm (UT transform + WY)
    4. Keep every n_h-th output

    This reuses the flash-linear-attention Triton kernel unchanged.
    """
    T, d = Q.shape

    # Step 1: Flatten (T, n_h, d) -> (T*n_h, d)
    flat_K, flat_V, flat_beta, flat_gates = deltaproduct_flatten_for_chunkwise(
        K, V, beta, gates, n_h
    )

    # Step 2: Run standard DeltaNet chunkwise parallel
    # (uses UT transform for intra-chunk, WY for inter-chunk)
    flat_output = deltanet_chunkwise_parallel(
        Q=Q.repeat_interleave(n_h, dim=0),  # Repeat queries n_h times
        K=flat_K, V=flat_V, beta=flat_beta,
        gates=flat_gates, chunk_size=chunk_size
    )

    # Step 3: Keep only every n_h-th output (one per token)
    output = flat_output[n_h-1::n_h]  # (T, d)

    return output
```

**GPU efficiency analysis:**

1. **Same kernel, longer sequence:** DeltaProduct reuses DeltaNet's optimized Triton kernel from flash-linear-attention. The only overhead is the $n_h \times$ longer sequence, which increases the UT transform cost from $O(C^2 d)$ to $O((n_h C)^2 d)$ per chunk — but chunks can be made smaller to compensate.

2. **Tensor core utilization:** All the dominant operations (UT transform matmuls, inter-chunk state updates) remain tensor-core-friendly. The per-step Householder products within the UT transform are handled by the same forward substitution + matmul pipeline.

3. **Throughput/quality tradeoff is favorable:** DeltaProduct$_2$ with 16 heads (35K tok/sec) matches DeltaNet with 8 heads (40K tok/sec) in throughput while achieving strictly better quality. The 12.5% throughput reduction buys significant expressivity gains (rotations, better length extrapolation).

4. **Memory unchanged:** The $n \times d$ inter-chunk state does not grow with $n_h$. Only the intra-chunk WY representation grows, but this is bounded by shared memory.

5. **HBM bandwidth:** The flattened sequence increases the amount of data loaded from HBM by $n_h \times$, which is the primary throughput bottleneck. A fused kernel that avoids materializing the extended sequence could recover most of this overhead.

## References

- Siems, J., Carstensen, T., Zela, A., Hutter, F., Pontil, M., & Grazzi, R. (2025). DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products. *NeurIPS 2025*. arXiv:2502.10297.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. *NeurIPS 2024*. arXiv:2406.06484.
- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. *ICLR 2025*. arXiv:2412.06464.
- Grazzi, R., Siems, J., Zela, A., Franke, J., Hutter, F., & Pontil, M. (2025). Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues. *ICLR 2025*.
- Bischof, C. & Van Loan, C. (1987). The WY Representation for Products of Householder Matrices. *SIAM J. Sci. Stat. Comput.*, 8(1), 2–13.
- Schreiber, R. & Parlett, B. (1988). Block Reflectors: Theory and Computation. *SIAM J. Numer. Anal.*, 25(1), 189–205.
- Yang, S. & Zhang, Y. (2024). FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism. https://github.com/sustcsonglin/flash-linear-attention
