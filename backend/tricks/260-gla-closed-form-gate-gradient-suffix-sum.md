# 260: GLA Closed-Form Gate Gradient via Suffix Sum

**Category**: efficiency
**Gain type**: efficiency
**Source**: Yang, Wang, Shen, Panda & Kim, "Gated Linear Attention Transformers with Hardware-Efficient Training" (2024) — ICML 2024
**Paper**: papers/gla-hardware-efficient-training.pdf
**Documented**: 2026-02-16

## Description

In Gated Linear Attention (GLA), each position has a data-dependent forget gate $\alpha_t \in (0,1)^{d_k}$ that controls how much of the recurrent state is retained. Computing the gradient $d\alpha_t$ during backpropagation was previously believed to require materializing the full $d_k \times d_v$ hidden state $S_t$ at every timestep in HBM (Mao, 2022, §3.1), incurring $O(L \cdot d_k \cdot d_v)$ memory — a prohibitive cost that scales with both sequence length and state size.

The GLA paper derives a **closed-form formula** for $d\log \alpha_t$ that completely avoids hidden state materialization for the gate gradient. The key insight is that the gradient of the log-cumulative-gate $\log b_t = \sum_{i=1}^{t} \log \alpha_i$ decomposes into a simple element-wise product of already-computed quantities:

$$
d\log b_t = q_t \odot dq_t - k_t \odot dk_t
$$

Since $\log b_t$ is a cumulative sum of $\log \alpha_i$, the individual gate gradients are recovered via a **reverse cumulative sum** (suffix sum):

$$
d\log \alpha_t = \sum_{i=t}^{L} d\log b_i = \text{revcumsum}(d\log b)_t
$$

This reduces the gate gradient computation from $O(L \cdot d_k \cdot d_v)$ (materializing all hidden states) to $O(L \cdot d_k)$ (element-wise products + one prefix sum), with **zero additional memory** beyond the already-computed $Q, K, dQ, dK$ gradients. The suffix sum is a single `torch.cumsum` on a flipped tensor — a trivially parallelizable GPU primitive.

## Mathematical Form

**Setup — GLA recurrence with cumulative gates:**

$$
S_t = \text{Diag}(\alpha_t) \, S_{t-1} + k_t^\top v_t, \quad o_t = q_t S_t
$$

Define the cumulative gate product $b_t = \prod_{i=1}^{t} \alpha_i$, so $\log b_t = \sum_{i=1}^{t} \log \alpha_i$.

**Parallel form (unrolled):**

$$
o_t = q_t S_t = \sum_{i=1}^{t} (q_t \odot b_t) \left(\frac{k_i}{b_i}\right)^\top v_i
$$

**Step 1 — Gradients w.r.t. Q and K (from Alg. 4/6 in the paper):**

$$
dq_t = \sum_{i=1}^{t} \langle do_t, v_i \rangle \, b_t \odot \frac{k_i}{b_i}
$$

$$
dk_i = \sum_{t=i}^{L} \langle do_t, v_i \rangle \, q_t \odot \frac{b_t}{b_i}
$$

These are computed as part of the standard chunkwise backward pass via matmuls with the $P$ matrix and the inter-chunk state $dS$.

**Step 2 — Gradient w.r.t. log-cumulative-gate (the key derivation):**

Taking the derivative of the output $o_t$ with respect to $\log b_t$:

$$
d\log b_t = q_t \odot \underbrace{\sum_{i=1}^{t} \langle do_t, v_i \rangle \, b_t \odot k_i / b_i}_{= \, dq_t} \;-\; k_t \odot \underbrace{\sum_{s=t}^{L} \langle do_s, v_t \rangle \, q_s \odot b_s / b_t}_{= \, dk_t}
$$

This simplifies to:

$$
\boxed{d\log b_t = q_t \odot dq_t - k_t \odot dk_t}
$$

**Step 3 — Recover individual gate gradients via suffix sum:**

Since $\log b_t = \sum_{i=1}^{t} \log \alpha_i$ is a cumulative sum, the chain rule gives:

$$
d\log \alpha_t = \sum_{i=t}^{L} d\log b_i
$$

This is a **reverse cumulative sum** (suffix sum), computed as:

$$
d\log \alpha = \text{flip}(\text{cumsum}(\text{flip}(d\log b)))
$$

Or equivalently: $dG = \text{revcumsum}(Q \odot dQ - K \odot dK)$

**Key Definitions:**

- $\alpha_t \in (0,1)^{d_k}$ — per-position, per-dimension forget gate (data-dependent)
- $b_t = \prod_{i=1}^{t} \alpha_i \in (0,1)^{d_k}$ — cumulative gate product
- $dq_t, dk_t \in \mathbb{R}^{d_k}$ — gradients w.r.t. query and key vectors (already computed in backward)
- $d\log b_t \in \mathbb{R}^{d_k}$ — gradient w.r.t. log-cumulative-gate
- $d\log \alpha_t \in \mathbb{R}^{d_k}$ — gradient w.r.t. individual log-gate (the desired output)
- $L$ — sequence length, $d_k$ — key dimension, $d_v$ — value dimension

## Complexity

| Operation | Without Trick (Mao 2022) | With Trick |
|-----------|-------------------------|------------|
| Gate gradient FLOPs | $O(L \cdot d_k \cdot d_v)$ | $O(L \cdot d_k)$ |
| Gate gradient memory | $O(L \cdot d_k \cdot d_v)$ (all hidden states) | $O(L \cdot d_k)$ (reuse $dQ, dK$) |
| Additional HBM reads | $L$ reads of $d_k \times d_v$ state | **Zero** — reuses existing gradients |
| Parallelism | Sequential (depends on $S_t$) | Fully parallel (element-wise + cumsum) |

**Memory:** The trick requires **zero additional memory** for the gate gradient. The quantities $Q, K, dQ, dK$ are already computed and available in the chunkwise backward pass. The only new computation is element-wise Hadamard products and one reverse cumulative sum.

**Savings at scale:** For GLA with $d_k = d/2, d_v = d, L = 2048, d = 2048$: naive approach stores $L \times d_k \times d_v = 2048 \times 1024 \times 2048 \approx 4$GB per head in BF16. The closed-form uses $L \times d_k = 2048 \times 1024 \approx 4$MB — a **1000× reduction**.

## Applicability

- **GLA (primary application):** This trick is the reason GLA can train with data-dependent matrix-valued gates without prohibitive memory costs. Without it, the gate gradient would dominate both memory and compute in the backward pass.

- **Any gated linear recurrence with log-space gates:** The derivation holds for any model of the form $S_t = \text{Diag}(\alpha_t) S_{t-1} + k_t^\top v_t$ where $\alpha_t$ is data-dependent and differentiable. This includes HGRN-2, GateLoop, and RWKV-6.

- **Models with $\alpha_t^\top \beta_t$ (general outer-product gates):** The GLA paper's Appendix C shows the closed form extends to the general case $G_t = \alpha_t^\top \beta_t$ with a similar suffix-sum structure. Though $\beta_t$ is fixed to $\mathbf{1}$ in practice, the trick generalizes.

- **Integration with chunkwise backward:** In the chunkwise backward (Alg. 4/6), $dQ$ and $dK$ are computed per-chunk as matmuls with the intra-chunk $P$ matrix and inter-chunk $dS$ state. The gate gradient is computed as a **post-processing step** after all chunk gradients are assembled: `dA = Q ⊙ dQ - K ⊙ dK`, `dG = revcumsum(dA)`. This adds negligible overhead.

## Limitations

- **Requires $dQ$ and $dK$ to be fully computed first:** The gate gradient is a function of the *final* $dQ$ and $dK$ values (after accumulating contributions from all chunks). This means it cannot be computed incrementally during the backward chunk loop — it must be a separate post-processing step over the full sequence.

- **Only works for multiplicative (diagonal) gates:** The derivation exploits the fact that $\text{Diag}(\alpha_t)$ commutes with element-wise operations. For models with full matrix transitions $A_t \in \mathbb{R}^{d \times d}$ (e.g., general SSMs), this trick does not apply.

- **Numerical precision in log-space:** Since we compute $d\log \alpha_t$ and then convert to $d\alpha_t$ via the chain rule ($d\alpha_t = \alpha_t \odot d\log \alpha_t$), any precision loss in $d\log b_t$ propagates through the suffix sum. For very long sequences, the cumulative sum may accumulate floating-point errors.

- **The suffix sum is sequential in theory:** While `torch.cumsum` is highly optimized on GPU (parallel prefix sum under the hood), it still has $O(\log L)$ depth. For very short sequences, the overhead of launching the cumsum kernel may exceed the cost of a simple element-wise accumulation.

## Implementation Notes

```python
import torch

def compute_gate_gradient(Q, K, dQ, dK, alpha):
    """
    Closed-form gate gradient for GLA.

    Args:
        Q:  (B, H, L, d_k) — query vectors
        K:  (B, H, L, d_k) — key vectors
        dQ: (B, H, L, d_k) — gradient w.r.t. Q (from chunkwise backward)
        dK: (B, H, L, d_k) — gradient w.r.t. K (from chunkwise backward)
        alpha: (B, H, L, d_k) — forget gate values (sigmoid output)

    Returns:
        d_alpha: (B, H, L, d_k) — gradient w.r.t. alpha
    """
    # Step 1: d(log b_t) = q_t ⊙ dq_t - k_t ⊙ dk_t
    dlog_b = Q * dQ - K * dK  # (B, H, L, d_k), element-wise

    # Step 2: d(log alpha_t) = sum_{i=t}^{L} d(log b_i)  [suffix sum]
    # Reverse cumsum = flip -> cumsum -> flip
    dlog_alpha = torch.flip(
        torch.cumsum(torch.flip(dlog_b, dims=[-2]), dim=-2),
        dims=[-2]
    )  # (B, H, L, d_k)

    # Step 3: Convert from log-space gradient to actual gradient
    # d_alpha = alpha * d(log alpha)  [chain rule: d/d_alpha log(alpha) = 1/alpha]
    d_alpha = alpha * dlog_alpha

    return d_alpha

# In the FLA (Flash Linear Attention) library, this appears at the end
# of the backward pass in Algorithm 4/6:
#
#   dA = Q ⊙ dQ - K ⊙ dK          # element-wise, O(L * d_k)
#   dG = revcumsum(dA)              # one reverse cumsum, O(L * d_k)
#
# Compare to Mao (2022) which requires:
#   for t in range(L):
#       dG[t] = (S[t-1] ⊙ dS[t]).sum()  # needs S[t] ∈ R^{d_k × d_v} stored!
#
# The closed form eliminates the O(L * d_k * d_v) state storage entirely.

# KEY INSIGHT: Why this works
# The gate α_t appears in the output o_t = q_t S_t in two ways:
#   1. Through Q: the cumulative gate b_t multiplies q_t (as Q̃ = Q ⊙ B)
#   2. Through K: the cumulative gate b_i divides k_i (as K̃ = K / B)
# So d(log b_t) receives contributions from both Q and K gradients,
# and these contributions are exactly q_t ⊙ dq_t and -k_t ⊙ dk_t.
# The suffix sum then "un-cumulates" to recover individual α_t gradients.
```

## References

- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635. (Section 4.3 "Memory-efficient $d\alpha_t$ computation" and Appendix p.20 derivation)
- Mao, H. H. (2022). Fine-Tuning Pre-Trained Transformers into Decaying Fast Weights. EMNLP 2022. (§3.1 — claimed gate gradient requires materializing all hidden states)
- Katsch, T. (2023). GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling. arXiv:2311.01927. (Also requires hidden state materialization for gate gradients)
- FLA Library: https://github.com/sustcsonglin/flash-linear-attention (Production implementation of GLA with this trick in the backward kernel)
