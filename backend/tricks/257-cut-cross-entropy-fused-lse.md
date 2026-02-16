# 257: Cut Cross-Entropy (CCE)

**Category**: kernel
**Gain type**: efficiency
**Source**: Wijmans et al., "Cut Your Losses in Large-Vocabulary Language Models" (Apple, ICLR 2025)
**Paper**: [papers/cut-cross-entropy.pdf]
**Documented**: 2026-02-16

## Description

Cut Cross-Entropy (CCE) computes the cross-entropy loss for large-vocabulary language models **without materializing the full logit matrix** into global GPU memory. In standard LLM training, the cross-entropy loss layer has become the dominant memory bottleneck — for Gemma 2 (2B) with vocabulary size 256K, the logit matrix alone consumes 24 GB of an 80 GB H100. CCE eliminates this by decomposing the loss into (1) an **indexed matrix multiplication** for the ground-truth logit and (2) a **fused linear-log-sum-exp** reduction over all vocabulary entries, both computed in SRAM via custom tiled kernels.

The key insight is that the cross-entropy loss and its gradient depend only on a single log-probability per token (the ground truth) plus the log-sum-exp normalizer — there is no need to store all $N \times |V|$ logits simultaneously. CCE further exploits the **inherent sparsity of softmax** for the backward pass: empirically, fewer than 0.02% of softmax entries are non-negligible in bf16, enabling gradient filtering that skips entire blocks of the gradient computation.

CCE reduces the memory footprint of the loss computation from **24 GB to 1 MB** (24,000× reduction) for Gemma 2 (2B), enabling 1.5×–10× larger batch sizes with no sacrifice in speed or convergence.

## Mathematical Form

**Standard Cross-Entropy (baseline):**

For an autoregressive LLM with embeddings $\mathbf{E} \in \mathbb{R}^{D \times N}$, classifier $\mathbf{C} \in \mathbb{R}^{D \times |V|}$, and ground-truth token indices $\mathbf{x} = (x_1, \ldots, x_N)$:

$$
\ell(\tilde{\mathbf{x}}) = \sum_{i=1}^{N} \log \text{softmax}_{x_i}\left(\mathbf{C}^\top E_i\right)
$$

This requires materializing $\mathbf{C}^\top \mathbf{E} \in \mathbb{R}^{|V| \times N}$ — an $O(N|V|)$ memory cost.

**Cut Cross-Entropy Decomposition:**

CCE reformulates the per-token loss as:

$$
\ell_i(\mathbf{x}) = C_{x_i}^\top E_i - \log \sum_j \exp\left(C_j^\top E_i\right)
$$

The full training loss decomposes into two terms:

$$
\boldsymbol{\ell} = \left(\mathbf{C}^\top \mathbf{E}\right)_\mathbf{x} - \log \sum_j \exp\left(C_j^\top \mathbf{E}\right)
$$

**Term 1 — Indexed Matrix Multiplication:** $\left(\mathbf{C}^\top \mathbf{E}\right)_\mathbf{x} = [C_{x_1}^\top E_1, \ldots, C_{x_N}^\top E_N]$

This fuses classifier indexing with a dot product: load $C_{x_i}$ and $E_i$ into SRAM, compute the dot product, write the scalar result. Memory cost: $O(ND)$ instead of $O(N|V|)$.

**Term 2 — Fused Linear-Log-Sum-Exp:**

Computed via a joint tiled matmul + reduction kernel. The output $\mathbf{O} = \mathbf{C}^\top \mathbf{E}$ is divided into blocks of size $V_B \times N_B$. Each block computes:

$$
\mathbf{A}_{nv} = \sum_d \mathbf{C}_{v,d}^\top \cdot \mathbf{E}_{n,d} \quad \text{(blockwise matmul in SRAM)}
$$

$$
\text{LSE}_{nv} = \log \sum \exp(\mathbf{A}_{nv}) \quad \text{(numerically stable with max subtraction)}
$$

$$
\text{LSE}_n = \log(\exp(\text{LSE}_n) + \exp(\text{LSE}_{nv})) \quad \text{(thread-safe log-add-exp)}
$$

Multiple CUDA blocks writing to the same LSE location are synchronized via atomic spin-locks.

**Backward Pass — Gradient with Sparsity Filtering:**

The gradients are:

$$
\nabla \mathbf{E}^\top = (\mathbf{S} \cdot \nabla\text{LSE})\, \mathbf{C} \quad \text{and} \quad \nabla \mathbf{C}^\top = (\mathbf{S} \cdot \nabla\text{LSE})^\top\, \mathbf{E}
$$

where $\mathbf{S} = \text{softmax}(\mathbf{C}^\top \mathbf{E}) = \exp(\mathbf{C}^\top \mathbf{E} - \text{LSE})$.

**Gradient filtering:** For any block where all entries $S_{nm} < \varepsilon = 2^{-12}$ (the bf16 truncation threshold), the entire gradient computation for that block is **skipped**. This yields a 3.5× backward pass speedup since the softmax matrix is $> 99.98\%$ sparse for frontier models.

## Complexity

| Operation | Standard | With CCE |
|-----------|----------|----------|
| Forward memory | $O(N \cdot |V|)$ | $O(N + |V|)$ |
| Loss computation memory | 24 GB (Gemma 2B) | 1 MB |
| Classifier head memory | 28 GB | 1 GB |
| Forward time | 49 ms (torch.compile) | 46 ms |
| Loss+Gradient time | 143 ms (torch.compile) | 145 ms |

**Memory:** $O(N + |V|)$ vs $O(N \times |V|)$ — vocabulary-size and sequence-length independent

**With gradient filtering:** Backward 3.5× faster than without filtering, no precision loss in bf16

## Applicability

- **Large-vocabulary LLMs**: Maximum benefit when $|V|$ is large (128K–256K tokens). The loss layer accounts for 40–89% of total training memory for modern LLMs
- **Pipeline parallelism**: CCE eliminates the classifier head as a memory/compute outlier stage, improving pipeline balance
- **Any model with large output space**: Image classification with many classes, contrastive learning
- **Pretraining at scale**: CCE-Kahan-FullC variant uses Kahan summation for numerical stability during pretraining, matching torch.compile convergence exactly
- **Memory-constrained training**: Enables 1.5×–10× larger batch sizes, which can reduce total training time (16% reduction observed with Mistral NeMo)

## Limitations

- **Requires custom CUDA/Triton kernels**: Not a drop-in replacement — needs specialized tiled implementation
- **Kahan summation for pretraining**: The basic CCE variant shows slight precision loss in bf16 pretraining (not fine-tuning). The CCE-Kahan-FullC variant fixes this at the cost of ~2× more memory for gradient buffers (still far less than baseline)
- **Gradient filtering inapplicable to pretraining**: Gradient filtering on $\nabla C$ skips tokens with low support, which hurts pretraining on diverse data. Only applicable to fine-tuning
- **Triton control flow constraints**: Block-level control flow in Triton means gradient filtering operates at tile granularity, not element-level. A CUDA implementation could be more fine-grained
- **Diminishing returns for small vocabularies**: When $|V|/D$ ratio is small, the memory savings and speedup decrease proportionally

## Implementation Notes

```python
# Pseudocode for the CCE forward pass (Algorithm 2 from paper)
# Key: Never materialize the full |V| x N logit matrix in HBM

def cce_forward(E, C, x, V_B, N_B, D_B):
    """
    E: embeddings [D, N], C: classifier [D, |V|], x: labels [N]
    V_B, N_B, D_B: block sizes for tiling
    """
    N = E.shape[1]
    LSE = torch.full((N,), -float('inf'))  # Log-sum-exp accumulators

    # Term 1: Indexed matmul (fused load + dot product in SRAM)
    o = torch.zeros(N)
    for n_block in range(0, N, N_B):
        for d_block in range(0, D, D_B):
            c = C[d_block:d_block+D_B, x[n_block:n_block+N_B]]  # Indexed load
            o[n_block:n_block+N_B] += (c * E[d_block:d_block+D_B, n_block:n_block+N_B]).sum(0)

    # Term 2: Fused linear-log-sum-exp (tiled matmul + online LSE)
    for n_block in range(0, N, N_B):
        for v_block in range(0, V, V_B):
            A = torch.zeros(V_B, N_B)  # In SRAM
            for d_block in range(0, D, D_B):
                A += C[d_block:d_block+D_B, v_block:v_block+V_B].T @ E[d_block:d_block+D_B, n_block:n_block+N_B]
            lse_block = torch.logsumexp(A, dim=0)  # Stable with max subtraction
            LSE[n_block:n_block+N_B] = logaddexp(LSE[n_block:n_block+N_B], lse_block)  # Atomic

    return o - LSE  # Per-token log-probabilities

# Backward pass skips blocks where softmax < 2^{-12} (gradient filtering)
# This exploits the extreme sparsity of softmax over large vocabularies
```

## GPU Efficiency Analysis

**Memory Access Pattern:**
- Forward: Streaming matmul tiles through SRAM — no HBM allocation for logits
- Backward: Recomputes $\mathbf{C}^\top \mathbf{E}$ tiles in SRAM rather than loading from HBM
- Access pattern identical to standard blocked GEMM — fully coalesced

**Parallelism:**
- Blocks over $(n, v)$ pairs are fully independent in forward pass
- Atomic spin-lock for LSE accumulation — low contention in practice
- Backward: block-level sparsity check enables early exit (entire CUDA blocks skip)
- Maps naturally to tensor cores for the matmul component

**Arithmetic Intensity:**
- Same FLOPs as standard cross-entropy (both compute $\mathbf{C}^\top \mathbf{E}$)
- Higher effective arithmetic intensity due to SRAM reuse (no HBM writes for logits)
- Gradient filtering reduces actual FLOPs by up to 3.5× in backward

**Hardware Utilization:**
- Tensor core utilization for the matmul tiles ($D_B \times N_B$ and $D_B \times V_B$ blocks)
- SRAM-only intermediate storage — fits in shared memory of modern GPUs
- Vocabulary sorting heuristic groups non-zero gradient blocks for better occupancy

## References

- Wijmans, E., Huval, B., Hertzberg, A., Koltun, V., Krähenbühl, P. "Cut Your Losses in Large-Vocabulary Language Models." ICLR 2025. arXiv:2411.09009
- Milakov, M. and Gimelshein, N. "Online normalizer calculation for softmax." arXiv:1805.02867 (2018) — online softmax used by FlashAttention and CCE
- Hsu, P.-L. et al. "Liger-Kernel: Efficient Triton kernels for LLM training." (2024) — chunked cross-entropy alternative
- Kahan, W. "Pracniques: further remarks on reducing truncation errors." Communications of the ACM (1965) — Kahan summation for numerical stability
- Apple open-source implementation: https://github.com/apple/ml-cross-entropy
