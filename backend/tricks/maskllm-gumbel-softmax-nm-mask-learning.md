# MaskLLM: Gumbel-Softmax N:M Mask Distribution Learning

**Category**: approximation
**Gain type**: efficiency
**Source**: Fang, Yin, Muralidharan, Heinrich, Pool, Kautz, Molchanov & Wang (NVIDIA & NUS, NeurIPS 2024)
**Paper**: [papers/maskllm-gumbel-softmax-nm-masks.pdf]
**Documented**: 2026-02-15

## Description

Channel permutation (e.g., PermLLM's blockwise Sinkhorn) improves N:M sparsity by *reordering* channels so that salient weights spread across pruning groups. MaskLLM takes an orthogonal approach: instead of permuting channels, it directly **learns which N:M mask pattern to apply** to each group of $M$ consecutive weights, treating mask selection as a differentiable categorical distribution optimized end-to-end on large-scale data.

For 2:4 sparsity, each group of 4 weights has $|S^{2:4}| = \binom{4}{2} = 6$ candidate binary masks. Traditional one-shot methods (Wanda, SparseGPT) select the mask based on handcrafted importance metrics evaluated on a tiny calibration set (128--256 samples). MaskLLM instead associates each group with a learnable categorical distribution $p = (p_1, \ldots, p_6)$ over the 6 candidate masks and optimizes these distributions by minimizing the actual language modeling loss on large-scale training data.

The key enabling trick is **Gumbel-Softmax reparameterization**: since sampling from a categorical distribution and applying `argmax` are non-differentiable, MaskLLM uses the Gumbel-Softmax relaxation to produce a differentiable "soft index" $\tilde{y}$ that approximates the one-hot sample. The soft mask is then computed as a weighted average of candidate masks $\tilde{M} = \tilde{y} \times S$, which is fully differentiable and allows gradient descent on the logits $\pi$.

A **scaling factor** $\kappa$ (linearly annealed from 1e2 to 5e2) controls the balance between exploration (Gumbel noise dominates) and exploitation (logits dominate), replacing the temperature annealing used in Sinkhorn-based approaches.

**Sparse Weight Regularization** prevents gradient vanishing through pruned pathways by encouraging large magnitudes in the retained weights: $-\lambda \sum_i \|W_i \odot \tilde{M}_i\|_2^2$.

**Mask Prior Transfer**: Pre-computed masks from one-shot methods (Magnitude, Wanda, SparseGPT) can initialize the Gumbel logits by biasing candidates similar to the prior mask, accelerating convergence from 9.12 PPL (no prior) to 6.77 PPL (Magnitude prior) on LLaMA-2 7B.

At inference, the mask with the highest logit per group is selected ($\text{argmax}$), yielding a standard N:M binary mask with zero runtime overhead.

## Mathematical Form

**Mask Selection as Categorical Sampling:**

For each parameter block $W_i \in \mathbb{R}^{1 \times M}$ (a group of $M$ consecutive weights), define the candidate mask set:

$$
S^{N:M} = \{M \in \mathbb{B}^{1 \times M} \mid \sum M = N\}
$$

For 2:4 sparsity, $|S^{2:4}| = \binom{4}{2} = 6$:

$$
S^{2:4} = \{[1,1,0,0], [1,0,1,0], [1,0,0,1], [0,1,0,1], [0,1,1,0], [0,0,1,1]\}
$$

**Optimization Objective:**

$$
\{p^*(\mathcal{M}_i)\} = \arg\min_{\{p(\mathcal{M}_i)\}} \mathbb{E}_{x \sim p(x), \mathcal{M}_i \sim p(\mathcal{M}_i)} \left[ \mathcal{L}_{LM}(x; \{W_i \odot \mathcal{M}_i\}) \right]
$$

where $p(\mathcal{M}_i)$ is the categorical distribution over candidate masks for the $i$-th group.

**Gumbel-Softmax Reparameterization:**

Draw Gumbel noise $g_i = -\log(-\log \epsilon_i)$, $\epsilon_i \sim U(0,1)$, and compute the soft index:

$$
\tilde{y}_i = \frac{\exp((\log(p_i) + g_i) / \tau)}{\sum_j \exp((\log(p_j) + g_j) / \tau)}
$$

where $\tau$ is the temperature. In practice, logits $\pi_i$ are learned with $p_i = \frac{\exp(\pi_i \cdot \kappa)}{\sum_j \exp(\pi_j \cdot \kappa)}$, where $\kappa$ is the scaling factor.

**Differentiable Mask Construction:**

$$
\tilde{\mathcal{M}} = \tilde{y} \times S = \sum_{i=1}^{|S|} \tilde{y}_i \cdot \hat{\mathcal{M}}_i
$$

This is a simple matrix multiplication between the soft index vector and the mask set matrix, producing a differentiable mask.

**Sparse Weight Regularization:**

$$
\min_{\{p_\pi(\mathcal{M}_i)\}} \mathbb{E}_{x, \tilde{\mathcal{M}}_i \sim p_\pi(\mathcal{M}_i)} \left[ \mathcal{L}_{LM}(x; \{W_i \odot \tilde{\mathcal{M}}_i\}) \right] - \lambda \sum_i \|W_i \odot \tilde{\mathcal{M}}_i\|_2^2
$$

The regularization term weighted by $\lambda$ encourages large magnitude in remaining weights, preventing gradient vanishing through pruned layers.

**Mask Prior Initialization:**

Given a prior mask $\mathcal{M}_0$ (e.g., from Wanda), compute similarity to each candidate:

$$
\text{sim}(\mathcal{M}_0, \hat{\mathcal{M}}_i) = \mathcal{M}_0 \hat{\mathcal{M}}_i^\top - \frac{1}{|S|} \sum_i (\mathcal{M}_i \hat{\mathcal{M}}^\top) = \mathcal{M}_i \hat{\mathcal{M}}^\top - (N/2)
$$

Initialize logits with bias toward the prior:

$$
\pi_i' = \pi_i + \sigma(\pi) \cdot \text{sim}(\mathcal{M}_0, \hat{\mathcal{M}}_i) \cdot \alpha
$$

where $\sigma(\pi)$ is the standard deviation of logits and $\alpha$ controls prior strength.

## Complexity

| Operation | One-shot (Wanda/SparseGPT) | MaskLLM |
|-----------|---------------------------|---------|
| Mask selection | $O(C_\text{in} \cdot C_\text{out} / M)$ top-$N$ per group | $O(T \cdot C_\text{in} \cdot C_\text{out} / M \cdot |S|)$ Gumbel-Softmax ($T$ steps) |
| Calibration data | 128--256 samples (fixed) | Scales to 512k+ samples (end-to-end) |
| Gradient through mask | None | Flows through Gumbel-Softmax to logits |
| Inference mask | Binary N:M (argmax of importance) | Binary N:M (argmax of logits) |
| Inference overhead | Zero | Zero |

**Learnable parameters:** $|S| = \binom{M}{N}$ logits per group of $M$ weights. For 2:4 sparsity: 6 logits per 4 weights = 1.5 parameters per weight. For LLaMA-2 7B: $\sim$10.5B logits total.

**Training cost:** 2,000 steps on large-scale data with frozen LLM weights. Comparable to a short fine-tuning run.

**Storage at inference:** Only 1 index per group ($\lceil\log_2 |S|\rceil$ bits). For 2:4: $\lceil\log_2 6\rceil / 4 \approx 0.65$ bits per parameter. Total for LLaMA-2 7B: $\sim$0.65 bits/param vs. 16 bits/param for fine-tuning.

**Inference speedup:** 1.4$\times$ wall-clock on A6000 GPU with 73% memory footprint (2:4 sparsity on Sparse Tensor Cores).

## Applicability

- **Post-training LLM pruning with frozen weights:** Primary application. On LLaMA-2 7B at 2:4 sparsity, achieves 6.72 Wikitext-2 PPL vs. 10.42 (SparseGPT), 11.29 (Wanda), closing 70% of the gap to the dense model (5.12 PPL). All model weights remain frozen; only mask logits are trained.
- **Domain-specific lossless compression:** By training task-specific masks on target domain data (CUDA, HTML, French, etc.), MaskLLM achieves lossless N:M sparsity on many downstream tasks (PPL equal to or better than dense on GPT-3 2B).
- **Transfer learning of sparsity patterns:** A general mask learned on broad data can be transferred as a prior to new tasks, reducing mask training to 2,000 steps. Transfer mask achieves 7.39 PPL vs. 7.51 (scratch) and 10.61 (general mask directly).
- **Complementary to channel permutation:** MaskLLM and PermLLM are orthogonal — permutation reorders channels while MaskLLM selects which pattern to apply per group. They could be composed: permute first, then learn masks on the permuted weight matrix.
- **Scales across model families:** Validated on LLaMA-2 (7B, 13B), Nemotron-4 (15B), GPT-3 (843M, 2B). Consistent improvements over one-shot baselines at all scales.
- **Sparse Tensor Core acceleration:** The learned mask is a standard 2:4 binary mask compatible with NVIDIA Sparse Tensor Core (A100/H100), enabling direct hardware acceleration with no custom kernel needed.

## Limitations

- Training requires $\sim$10.5B learnable logits for a 7B model (1.5$\times$ the model's parameter count), consuming significant GPU memory during training
- Requires large-scale training data for best results (512k+ samples); with only 128 samples, MaskLLM slightly underperforms SparseGPT
- Only demonstrated with frozen weights — not validated for sparse training from scratch where PA-DST or dynamic sparse training would apply
- The $\binom{M}{N}$ candidate set grows combinatorially: for 4:8 sparsity, $|S| = 70$ candidates per group, increasing memory and compute per step
- The Gumbel noise scaling factor $\kappa$ requires careful tuning — too small ($\kappa = 1$) causes slow convergence, too large ($\kappa = 10^5$) suppresses exploration
- Does not address the channel ordering problem — the mask quality is fundamentally limited by the channel order, which permutation methods like PermLLM address

## Implementation Notes

```python
import torch
import torch.nn.functional as F

# Enumerate all 2:4 candidate masks
def enumerate_nm_masks(N=2, M=4):
    """Generate all C(M,N) binary masks for N:M sparsity."""
    from itertools import combinations
    masks = []
    for positions in combinations(range(M), N):
        mask = [0] * M
        for p in positions:
            mask[p] = 1
        masks.append(mask)
    return torch.tensor(masks, dtype=torch.float32)  # (|S|, M)

S = enumerate_nm_masks(N=2, M=4)  # (6, 4)

def gumbel_softmax_mask(logits, S, kappa=100.0, tau=1.0):
    """
    Differentiable mask selection via Gumbel-Softmax.

    Args:
        logits: (num_groups, |S|) learnable logits per group
        S: (|S|, M) candidate mask set
        kappa: scaling factor for logits (annealed 1e2 -> 5e2)
        tau: Gumbel-Softmax temperature

    Returns:
        soft_mask: (num_groups, M) differentiable mask
    """
    # Scale logits
    scaled_logits = logits * kappa

    # Gumbel-Softmax sampling
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(scaled_logits) + 1e-20) + 1e-20)
    y_soft = F.softmax((scaled_logits + gumbel_noise) / tau, dim=-1)  # (num_groups, |S|)

    # Weighted average of candidate masks
    soft_mask = y_soft @ S  # (num_groups, M)
    return soft_mask

def maskllm_forward(W, logits, S, x, kappa=100.0):
    """
    MaskLLM forward pass for a single linear layer.

    Args:
        W: (d_out, d_in) frozen weight matrix
        logits: (d_out * d_in / M, |S|) learnable mask logits
        S: (|S|, M) candidate masks
        x: (batch, seq, d_in) input activation
        kappa: logit scaling factor

    Returns:
        y: (batch, seq, d_out) output
    """
    M = S.shape[1]
    num_groups = W.numel() // M

    # Get differentiable mask (training)
    soft_mask = gumbel_softmax_mask(logits, S, kappa)  # (num_groups, M)
    mask_2d = soft_mask.view(W.shape)  # (d_out, d_in)

    # Apply mask to weights
    W_sparse = W * mask_2d
    return F.linear(x, W_sparse)

def maskllm_inference(W, logits, S):
    """At inference: select hard mask via argmax."""
    M = S.shape[1]
    best_idx = logits.argmax(dim=-1)  # (num_groups,)
    hard_mask = S[best_idx]  # (num_groups, M)
    return (W * hard_mask.view(W.shape))

# Training loop (sketch):
# logits = nn.Parameter(torch.randn(num_groups, 6) * 0.01)
# kappa = 100.0  # linearly anneal to 500 over 2000 steps
# for step in range(2000):
#     kappa = 100 + step * 400 / 2000
#     soft_mask = gumbel_softmax_mask(logits, S, kappa)
#     W_masked = W * soft_mask.view(W.shape)
#     loss = LM_loss(W_masked, batch) - lambda * (W * soft_mask.view(W.shape)).norm()**2
#     loss.backward()  # gradients flow to logits only
#     optimizer.step()
```

## References

- Fang, G., Yin, H., Muralidharan, S., Heinrich, G., Pool, J., Kautz, J., Molchanov, P. & Wang, X. (2024). MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models. NeurIPS 2024. arXiv:2409.17481.
- Jang, E., Gu, S. & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. ICLR 2017.
- Gumbel, E.J. (1954). Statistical Theory of Extreme Values and Some Practical Applications.
- Frantar, E. & Alistarh, D. (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. ICML 2023.
- Sun, M. et al. (2023). Wanda: A Simple and Effective Pruning Approach for Large Language Models. arXiv:2306.11695.
- Pool, J. & Yu, C. (2021). Channel Permutations for N:M Sparsity. NeurIPS 2021.
- Zou, L. et al. (2025). PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models. NeurIPS 2025. arXiv:2510.10136.
