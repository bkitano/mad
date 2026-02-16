# 243: Residual Linear Attention (RLA/RDN)

**Category**: approximation
**Gain type**: expressivity
**Source**: Lai, Kang, Lu, Lin & Zhao (2025) — Peking University / University of Hong Kong / Minimax
**Paper**: [papers/residual-linear-attention.pdf]
**Documented**: 2026-02-15

## Description

Linear attention models can be decomposed into a **base prediction** from the accumulated state matrix and a **correction term** derived from the current token alone. This single-token correction creates an expressivity bottleneck — the error correction at each step can only depend on the current key-value pair, not on accumulated prediction errors.

**Residual Linear Attention (RLA)** introduces an auxiliary recurrent state $\boldsymbol{R}_t$ that explicitly accumulates past residual errors $\boldsymbol{r}_t = \boldsymbol{v}_t - \boldsymbol{S}_{t-1}\boldsymbol{k}_t$ and their corresponding keys. This second state matrix learns to correct systematic prediction errors of the base state $\boldsymbol{S}_{t-1}$, yielding more expressive output while preserving $O(T)$ time and memory complexity.

The key insight is that the residual $\boldsymbol{r}_t$ is the optimal Newton step for the L2 loss $\frac{1}{2}\|\boldsymbol{S}\boldsymbol{k}_t - \boldsymbol{v}_t\|^2$ — motivating the auxiliary state to accumulate these residuals as a second-order correction. The auxiliary state uses the **same linear recurrence structure** as the base state, so existing optimized chunkwise-parallel Triton kernels (from the FLA library) can be reused with only minor modification: the kernel returns both the attention result and the intermediate residual.

**Residual Delta Net (RDN)** extends this to the delta-rule family, where $\boldsymbol{R}_t$ also uses a Householder-style correction $(\boldsymbol{I} - \gamma_t \boldsymbol{k}_t\boldsymbol{k}_t^\top)$ for fine-grained memory management in the auxiliary state.

At 1.5B parameters / 100B tokens, RDN achieves 14.93 LAMBADA ppl (vs. 15.76 for GDN, 20.80 for Mamba2) and 47.11 average recall accuracy (vs. 44.99 for GDN, 42.54 for Mamba2), narrowing the gap to Transformers (14.93 vs. 19.53) while retaining linear scaling.

## Mathematical Form

**Prediction-correction decomposition of linear attention:**

The standard linear attention output can be decomposed as:

$$
\boldsymbol{o}_t = \underbrace{\boldsymbol{S}_{t-1}\boldsymbol{q}_t}_{\text{Base Prediction}} + \underbrace{(\boldsymbol{v}_t\boldsymbol{k}_t^\top)\boldsymbol{q}_t}_{\text{Single-Token Correction}}
$$

This decomposition generalizes across all linear attention variants via a correction state $\boldsymbol{R}_t$:

$$
\boldsymbol{o}_t = \boldsymbol{S}_{t-1}\boldsymbol{q}_t + \boldsymbol{R}_t\boldsymbol{q}_t
$$

**Residual error computation:**

$$
\boldsymbol{r}_t = \text{Clip}_{[-c, c]}(\boldsymbol{v}_t - \boldsymbol{S}_{t-1}\boldsymbol{k}_t)
$$

where $c > 0$ is a clipping threshold (typically $c = 1$) for numerical stability.

**Residual Linear Attention (RLA) — full formulation with gating:**

$$
\boldsymbol{S}_t = \alpha_t \boldsymbol{S}_{t-1} + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^\top
$$

$$
\boldsymbol{R}_t = \alpha_t \boldsymbol{R}_{t-1} + \gamma_t \boldsymbol{r}_t \boldsymbol{k}_t^\top
$$

$$
\boldsymbol{o}_t = \alpha_t \boldsymbol{S}_{t-1}\boldsymbol{q}_t + \gamma_t \boldsymbol{R}_t \boldsymbol{q}_t
$$

where $\alpha_t \in [0, 1]$ is a decay factor, $\beta_t \in [0, 1]$ is the base update rate, and $\gamma_t \in [0, 1]$ is a **dedicated correction factor** (decoupled from $\beta_t$) that controls the contribution of the residual-fitting term.

**Residual Delta Net (RDN) — delta-rule variant:**

$$
\boldsymbol{S}_t = \alpha_t \boldsymbol{S}_{t-1}(\boldsymbol{I} - \beta_t \boldsymbol{k}_t\boldsymbol{k}_t^\top) + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^\top
$$

$$
\boldsymbol{R}_t = \alpha_t \boldsymbol{R}_{t-1}(\boldsymbol{I} - \gamma_t \boldsymbol{k}_t\boldsymbol{k}_t^\top) + \gamma_t \boldsymbol{r}_t \boldsymbol{k}_t^\top
$$

$$
\boldsymbol{o}_t = \alpha_t \boldsymbol{S}_{t-1}\boldsymbol{q}_t + \gamma_t \boldsymbol{R}_t \boldsymbol{q}_t
$$

**Key Definitions:**

- $\boldsymbol{S}_t \in \mathbb{R}^{d_v \times d_k}$ — primary state matrix (base model)
- $\boldsymbol{R}_t \in \mathbb{R}^{d_v \times d_k}$ — auxiliary residual state matrix (error correction)
- $\boldsymbol{r}_t = \text{Clip}_{[-c,c]}(\boldsymbol{v}_t - \boldsymbol{S}_{t-1}\boldsymbol{k}_t) \in \mathbb{R}^{d_v}$ — clipped residual error
- $\alpha_t = \exp(-a \cdot \text{softplus}(\boldsymbol{W}_\alpha \boldsymbol{x}_t + b))$ — decay gate (Mamba2 reparameterization)
- $\beta_t = \sigma(\boldsymbol{W}_\beta \boldsymbol{x}_t)$ — base update rate (sigmoid)
- $\gamma_t = \sigma(\boldsymbol{W}_\gamma \boldsymbol{x}_t)$ — correction factor (sigmoid, separate from $\beta_t$)
- $c$ — residual clipping threshold (default: 1.0)

**Motivation from second-order analysis:**

Given a prediction $\hat{\boldsymbol{v}}$ and the L2 loss $\mathcal{L}(\hat{\boldsymbol{v}}, \boldsymbol{v}) = \frac{1}{2}\|\hat{\boldsymbol{v}} - \boldsymbol{v}\|^2$, the optimal Newton correction is:

$$
\boldsymbol{\delta}^* = -(\nabla_{\hat{v}}^2 \mathcal{L})^{-1} \nabla_{\hat{v}} \mathcal{L} = \boldsymbol{v} - \hat{\boldsymbol{v}}
$$

For the base prediction $\hat{\boldsymbol{v}}_t = \boldsymbol{S}_{t-1}\boldsymbol{k}_t$, this gives $\boldsymbol{\delta}^* = \boldsymbol{v}_t - \boldsymbol{S}_{t-1}\boldsymbol{k}_t = \boldsymbol{r}_t$, motivating the residual accumulation.

## Complexity

| Operation | Base Linear Attention | With Residual Fitting |
|-----------|----------------------|----------------------|
| State updates per step | 1 matmul ($O(d_k d_v)$) | 2 matmuls ($O(d_k d_v)$) |
| Output computation | 1 matvec ($O(d_k d_v)$) | 2 matvecs ($O(d_k d_v)$) |
| Total per step | $O(d_k d_v)$ | $O(d_k d_v)$ (2× constant) |
| Sequence scaling | $O(T)$ | $O(T)$ |
| Memory (states) | $d_k \times d_v$ per head | $2 \times d_k \times d_v$ per head |

**Key observation:** The auxiliary state $\boldsymbol{R}_t$ uses the **exact same recurrence structure** as $\boldsymbol{S}_t$ (gated linear attention or gated delta rule). This means the same optimized chunkwise-parallel kernel can be called twice — once for the base and once for the residual — with minimal code change.

**Wall-clock throughput (H100, 1.5B models, Fig. 2 from paper):**

| Seq Length | FlashAttention | Gated LinearAttn | Residual LinearAttn | Gated DeltaNet | Residual DeltaNet |
|-----------|----------------|-----------------|-------------------|---------------|-----------------|
| 2K | ~125 Kt/s | ~115 Kt/s | ~115 Kt/s | ~110 Kt/s | ~110 Kt/s |
| 8K | ~130 Kt/s | ~130 Kt/s | ~125 Kt/s | ~120 Kt/s | ~120 Kt/s |
| 32K | ~100 Kt/s | ~135 Kt/s | ~130 Kt/s | ~130 Kt/s | ~125 Kt/s |
| 128K | ~20 Kt/s | ~140 Kt/s | ~130 Kt/s | ~130 Kt/s | ~120 Kt/s |

The residual fitting adds ~5-8% overhead to the base linear attention kernel, while maintaining the same linear scaling characteristic. At 128K, the advantage over FlashAttention is ~6× in throughput.

## Applicability

- **Drop-in enhancement for any linear attention model**: The residual fitting framework is a general wrapper. The paper demonstrates it on both standard gated linear attention (sGLA → RLA) and gated delta rule (GDN → RDN). Any model with recurrence $\boldsymbol{S}_t = f(\boldsymbol{S}_{t-1}, \boldsymbol{k}_t, \boldsymbol{v}_t)$ can be augmented.

- **Recall-intensive tasks**: The auxiliary state particularly helps with recall — RDN achieves 79.2% on NIAH (needle-in-a-haystack) vs. 75.7% for GDN and 67.2% for Mamba2. The residual state provides a second channel for memorizing difficult associations.

- **Language modeling at scale**: At 1.5B / 100B tokens, RDN achieves 16.57 WikiText ppl (vs. 17.27 GDN, 18.42 Mamba2) and competitive with Transformer (17.33). Best among all linear attention methods on 8 of 10 language/reasoning benchmarks.

- **Integration with FLA library**: Implementation builds on `flash-linear-attention` Triton kernels. The same kernel is augmented to return both the attention result and intermediate residual, enabling kernel reuse across both stages.

## Limitations

- **2× state memory**: The auxiliary state $\boldsymbol{R}_t$ doubles the per-head state memory from $d_k \times d_v$ to $2 \times d_k \times d_v$. At $d_k = d_v = 128$, this is 32KB → 64KB per head — still far below softmax attention's KV-cache but non-trivial.

- **~2× compute for the recurrence**: While the chunkwise kernels can be reused, the residual fitting requires a second pass through the recurrence. This roughly doubles the state-update FLOPs (not the intra-chunk attention FLOPs, which dominate at large chunk sizes).

- **Residual clipping is a hyperparameter**: The clipping threshold $c$ affects training stability and performance. The paper uses $c = 1$ throughout, but optimal values may vary across tasks and scales. Without clipping, RLA (but not RDN) exhibits training instability due to exploding activation norms.

- **Coupling between base and auxiliary**: The residual $\boldsymbol{r}_t$ depends on the base state $\boldsymbol{S}_{t-1}$, creating a dependency between the two passes. This prevents fully independent parallelization of base and residual computations within a single kernel launch.

- **Validated at 1.5B scale only**: While results are strong at 1.5B / 100B tokens, behavior at larger scales (7B+, 1T+ tokens) is unknown. The 2× state overhead could interact with other scaling considerations.

## Implementation Notes

```python
# Residual Linear Attention (RLA) — PyTorch pseudocode
# Key insight: reuse the same GLA chunkwise kernel for both base and residual passes

def residual_linear_attention(q, k, v, x, W_alpha, W_beta, W_gamma, c=1.0):
    """
    RLA forward pass.

    The base state S and auxiliary state R use identical recurrence structure,
    so the same chunkwise-parallel Triton kernel handles both.
    """
    B, H, T, D = q.shape

    # Compute gates
    alpha = torch.exp(-F.softplus(W_alpha(x)))  # decay gate
    beta = torch.sigmoid(W_beta(x))              # base update rate
    gamma = torch.sigmoid(W_gamma(x))            # correction factor (separate!)

    # L2-normalize queries and keys, apply SiLU to values
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    # === Pass 1: Base linear attention ===
    # Standard chunkwise GLA kernel, but ALSO returns intermediate residual
    # r_t = v_t - S_{t-1} k_t (computed inside the kernel for free)
    o_base, residuals = chunkwise_gla_with_residual(q, k, v, alpha, beta)
    # o_base[t] = alpha_t * S_{t-1} @ q_t  (base prediction)
    # residuals[t] = v_t - S_{t-1} @ k_t   (prediction error)

    # Clip residuals for stability
    r = torch.clamp(residuals, -c, c)

    # === Pass 2: Residual fitting (reuses same kernel!) ===
    # R_t = alpha_t * R_{t-1} + gamma_t * r_t @ k_t^T
    o_correction, _ = chunkwise_gla(q, k, r, alpha, gamma)
    # o_correction[t] = gamma_t * R_t @ q_t

    # === Combine ===
    o = o_base + o_correction

    return o


# For RDN (Residual Delta Net), same principle but with delta-rule kernel:
def residual_delta_net(q, k, v, x, W_alpha, W_beta, W_gamma, c=1.0):
    """
    RDN forward pass. Uses GDN chunkwise kernel for both passes.
    """
    alpha = torch.exp(-F.softplus(W_alpha(x)))
    beta = torch.sigmoid(W_beta(x))
    gamma = torch.sigmoid(W_gamma(x))

    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    # Pass 1: Base GDN with residual output
    # S_t = alpha_t * S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    o_base, residuals = chunkwise_gdn_with_residual(q, k, v, alpha, beta)

    r = torch.clamp(residuals, -c, c)

    # Pass 2: Residual fitting with delta rule
    # R_t = alpha_t * R_{t-1}(I - gamma_t k_t k_t^T) + gamma_t r_t k_t^T
    o_correction, _ = chunkwise_gdn(q, k, r, alpha, gamma)

    o = o_base + o_correction
    return o
```

**GPU efficiency analysis:**

1. **Kernel reuse**: The critical implementation insight is that both $\boldsymbol{S}_t$ and $\boldsymbol{R}_t$ follow the same recurrence structure. The existing FLA Triton kernels for GLA/GDN are reused with minimal modification (returning the intermediate residual alongside the output). No new kernel development needed.

2. **Memory access pattern**: The residual $\boldsymbol{r}_t = \boldsymbol{v}_t - \boldsymbol{S}_{t-1}\boldsymbol{k}_t$ is computed within the chunkwise kernel during the base pass — it's a byproduct of the inter-chunk output computation $\boldsymbol{S}_{t-1}\boldsymbol{k}_t$ that's already being calculated. This means the residual comes "for free" from a memory access perspective.

3. **Arithmetic intensity preserved**: Each pass individually maintains the same arithmetic intensity as the base kernel. The two passes are sequential (due to the dependency $\boldsymbol{r}_t$ depending on $\boldsymbol{S}_{t-1}$), but each pass is independently compute-bound at typical chunk sizes.

4. **All operations are matmuls + elementwise**: Clipping is an elementwise op. The gate computations are small linear projections. The dominant operations in both passes are the same tensor-core-friendly matmuls as the base model.

## References

- Lai, X., Kang, J., Lu, J., Lin, T., & Zhao, P. (2025). Enhancing Linear Attention with Residual Learning. arXiv:2509.25223.
- Yang, S. & Zhang, B. (2024). FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism. arXiv:2410.10989.
- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. arXiv:2412.06464.
- Dao, T. & Gu, A. (2024). Transformers are SSMs. ICML 2024. arXiv:2405.21060.
- Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML 2021.
- Sun, Y., et al. (2023). Retentive Network: A Successor to Transformer for Large Language Models. arXiv:2307.08621.
