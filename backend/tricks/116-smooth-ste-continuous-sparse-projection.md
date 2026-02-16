# 116: Smooth STE (S-STE): Continuous Pruning for 2:4 Sparse Training

**Category**: stability
**Gain type**: efficiency
**Source**: Hu, Zhu & Chen "S-STE: Continuous Pruning Function for Efficient 2:4 Sparse Pre-training" (NeurIPS 2024)
**Paper**: [papers/s-ste-continuous-pruning-2-4.pdf]
**Documented**: 2025-06-15

## Description

S-STE (Smooth Straight-Through Estimator) solves a fundamental optimization problem in 2:4 sparse pre-training: the **discontinuity** of the hard-thresholding pruning function used by prior STE-based methods. When training with N:M sparsity from scratch, a dense weight is maintained and sparsified each iteration via a pruning function $S$. The standard approach uses hard-thresholding $S_h$, which is discontinuous — an arbitrarily small weight change can cause the sparse mask to flip, leading to three pathological phenomena:

1. **Incorrect descending direction**: gradient steps frequently *increase* the loss instead of decreasing it
2. **Inability to predict amount of descent**: the actual loss reduction deviates wildly from the predicted (Taylor-approximation) reduction
3. **Mask oscillation ("flip rate explosion")**: the sparsity mask oscillates between patterns indefinitely, never converging

S-STE replaces hard-thresholding with a **continuous soft-thresholding** projection that eliminates all three problems. The key insight is that at the boundary where two elements have equal magnitude and a mask flip should occur, the continuous function smoothly transitions through a state where *three* elements are zeroed out (rather than abruptly swapping which two are kept). Combined with a fixed MSE-minimizing scaling factor, this yields a fully continuous loss landscape for sparse training that matches dense training's optimization behavior.

## Mathematical Form

**Hard-Thresholding (Prior Work — Discontinuous):**

For a block $\mathbf{a} = [a_1, a_2, a_3, a_4]^\top \in \mathbb{R}^4$, let $t$ be the $N$-th largest magnitude:

$$
(S_h(\mathbf{a}))_i = \begin{cases} a_i & \text{if } |a_i| \geq t \\ 0 & \text{if } |a_i| < t \end{cases}
$$

This is **discontinuous** when two elements have equal magnitude.

**Soft-Thresholding (S-STE — Continuous):**

Let $[t_1, t_2, t_3, t_4]$ be a rearrangement of $\mathbf{a}$ such that $|t_1| \leq |t_2| \leq |t_3| \leq |t_4|$. Set the threshold $t = |t_2|$ (the second-smallest magnitude). Then:

$$
(S_{soft}(\mathbf{a}))_i = \begin{cases} a_i - t & \text{if } a_i \in [t, +\infty) \\ 0 & \text{if } a_i \in (-t, t) \\ a_i + t & \text{if } a_i \in (-\infty, -t] \end{cases}
$$

where $t = |t_2|$.

**Key Property:** $S_{soft}(\mathbf{a})$ is a **continuous projection** for all $\mathbf{a} \in \mathbb{R}^d$.

**Fixed Weight Rescaling:**

Because soft-thresholding shrinks magnitudes, a per-tensor scaling factor $\beta$ is applied:

$$
S(\mathbf{w}) = \beta \cdot S_{soft}(\mathbf{w})
$$

The optimal $\beta$ minimizes the MSE between dense and sparse weights:

$$
\text{MSE} = \|\mathbf{w} - \beta S_{soft}(\mathbf{w})\|^2
$$

Taking the derivative and setting to zero:

$$
\beta = \frac{\mathbf{w}^\top S_{soft}(\mathbf{w})}{\|S_{soft}(\mathbf{w})\|^2}
$$

**Critical design choice:** $\beta$ is computed **once** at initialization and **frozen** throughout training. Dynamically updating $\beta$ causes flip rate explosion in later layers.

**STE Training Loop:**

At each iteration $k$:
1. Sparsify: $\tilde{\mathbf{w}}_k = \beta S_{soft}(\mathbf{w}_k)$
2. Forward pass with sparse $\tilde{\mathbf{w}}_k$ (accelerated by Sparse Tensor Cores)
3. Compute gradient: $\nabla_{\tilde{\mathbf{w}}_k} f$
4. Update **dense** weights: $\mathbf{w}_{k+1} = \mathbf{w}_k - \alpha_k \nabla_{\tilde{\mathbf{w}}_k} f$ (STE approximation: $\partial S / \partial \mathbf{w} \approx \mathbf{I}$)

**Key Definitions:**

- $\mathbf{w} \in \mathbb{R}^d$ — dense weight vector
- $\tilde{\mathbf{w}} \in \mathcal{W} \subset \mathbb{R}^d$ — sparse weight in the N:M constrained space
- $S_{soft}$ — continuous 2:4 soft-thresholding projection
- $\beta \in \mathbb{R}$ — per-tensor fixed scaling factor (frozen after first iteration)
- $m_h(\mathbf{w})$ — the 0/1 hard mask vector
- Flip rate: $r_k = \|m_h(\mathbf{w}_k) \oplus m_h(\mathbf{w}_{k-1})\|_1 / d$ (fraction of mask bits that change)

## Complexity

| Operation | Hard-Thresholding (STE/SR-STE) | S-STE |
|-----------|-------------------------------|-------|
| Pruning function | $O(d)$ per-block top-k | $O(d)$ per-block sort + subtract |
| Gradient computation | Same (STE identity) | Same (STE identity) |
| Scaling factor | Dynamic per-iteration (or none) | **One-time** computation, then frozen |
| Hyperparameters | $\lambda_W$ decay strength (SR-STE) | **None** (hyper-parameter free) |

**Memory:** Same as other STE methods — maintains a dense copy of weights ($2\times$ the sparse model size during training).

**Training Speedup:**
- FFN layer: 1.31–1.54× speedup (pre-training: 1.31×, inference: 1.54×)
- End-to-end transformer block: 1.15–1.23× speedup
- With FP8 quantization: theoretically up to 3× faster forward + backward (2× from sparsity, additional from FP8)

## Applicability

- **Transformer FFN layers**: Primary target — the two linear projections in feed-forward networks. S-STE pre-trains GPT-2 (124M–774M), Transformer-base, and DeiT-small with only 1–2% accuracy gap vs. dense
- **Language models**: GPT-2 pre-training on OpenWebText with S-STE matches or exceeds SR-STE+dense-finetuning on GLUE and SQuAD benchmarks
- **Vision transformers**: DeiT-tiny/small on ImageNet-1K — S-STE achieves 68.5%/78.5% top-1 accuracy vs. 72.2%/79.9% dense (best among all 2:4 methods)
- **Machine translation**: Transformer-base on WMT14 En-De — 26.11 Test BLEU (dense baseline: 26.42)
- **Composes with**: MVUE (minimum-variance unbiased estimation) for backward pass acceleration, FP8 quantization, and transposable masks

## Limitations

- Only tested on FFN layers — attention QKV projections remain dense (the paper notes this needs further exploration)
- On H100 GPUs, cuSPARSELt library performance is suboptimal — actual 2:4-spMM achieves only 1900 TFLOPS vs. 3200 TFLOPS theoretical (Table 11)
- Still requires maintaining a dense copy of weights during training (no memory savings during training, only inference)
- The 1–2% accuracy gap vs. dense models may be significant for some applications
- Scaling factor $\beta$ is frozen after first iteration — if weight distribution shifts dramatically, the frozen $\beta$ may become suboptimal

## Implementation Notes

```python
import torch

def soft_threshold_2_4(w: torch.Tensor) -> torch.Tensor:
    """Continuous 2:4 soft-thresholding projection.

    For each group of 4 elements, subtracts the 2nd-smallest
    magnitude from the kept elements, zeros the rest.
    """
    # Reshape to groups of 4
    original_shape = w.shape
    w_flat = w.view(-1, 4)

    # Sort by magnitude within each group
    magnitudes = w_flat.abs()
    sorted_mag, _ = magnitudes.sort(dim=1)  # ascending

    # Threshold = 2nd smallest magnitude (index 1)
    t = sorted_mag[:, 1:2]  # shape: (num_groups, 1)

    # Apply soft-thresholding: shrink toward zero by t
    signs = w_flat.sign()
    shrunk = magnitudes - t
    result = signs * torch.clamp(shrunk, min=0.0)

    return result.view(original_shape)

def compute_frozen_beta(w: torch.Tensor) -> float:
    """Compute per-tensor MSE-minimizing scale factor (frozen)."""
    s = soft_threshold_2_4(w)
    beta = (w * s).sum() / (s * s).sum()
    return beta.item()

# Training loop (sketch):
# beta = compute_frozen_beta(model.ffn.weight)  # compute once
# for step in range(num_steps):
#     w_sparse = beta * soft_threshold_2_4(model.ffn.dense_weight)
#     loss = forward(w_sparse, x)
#     loss.backward()  # gradients flow through STE
#     model.ffn.dense_weight -= lr * model.ffn.dense_weight.grad
```

## References

- Hu, Y., Zhu, J. & Chen, J. "S-STE: Continuous Pruning Function for Efficient 2:4 Sparse Pre-training" (NeurIPS 2024). arXiv:2409.09099
- Zhou, A., et al. "Learning N:M Fine-grained Structured Sparse Neural Networks from Scratch" (ICLR 2021). SR-STE baseline.
- Vanderschueren, A. & De Vleeschouwer, C. "Are Straight-Through Gradients and Soft-Thresholding All You Need for Sparse Training?" (2022). Original soft-thresholding for pruning.
- Chmiel, B., et al. "Minimum Variance Unbiased N:M Sparsity for the Neural Gradients" (ICLR 2023). MVUE technique composed with S-STE.
- Hu, Y., et al. "Accelerating Transformer Pre-training with 2:4 Sparsity" (ICML 2024). Transposable SR-STE + dense fine-tuning workflow.
