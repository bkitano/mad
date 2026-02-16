# 017: Block-wise Sinkhorn Channel Permutation (PermLLM)

**Category**: approximation
**Gain type**: efficiency
**Source**: Zou, Yin, Pei, Ho, Farnia & Yu (CUHK, NeurIPS 2025)
**Paper**: [papers/permllm-channel-permutation-nm-sparse.pdf]
**Documented**: 2026-02-15

## Description

Full-matrix channel permutation for N:M sparsity learns a $C_\text{in} \times C_\text{in}$ soft permutation matrix, requiring $O(C_\text{in}^2)$ parameters and $O(C_\text{in}^3)$ Hungarian hardening cost per layer. For large language models with $C_\text{in} \geq 4096$, this is prohibitively expensive. PermLLM introduces **block-wise learnable channel permutation (LCP)** that partitions the $C_\text{in}$ input channels into $N_B$ non-overlapping blocks of size $B$ and learns a separate small permutation within each block. This reduces the parameter count by a factor of $C_\text{in} / B$ and the hardening cost from $O(C_\text{in}^3)$ to $O(C_\text{in} \cdot B^2)$.

The key insight is that within-block permutation captures most of the benefit of full permutation for N:M sparsity. Since N:M pruning operates on small groups of $M$ consecutive elements, the salient-weight clustering problem is predominantly local: redistributing salient weights within a block of size $B$ (where $B = 64$ by default) is sufficient to improve N:M mask quality significantly. Global cross-block reordering adds diminishing returns at much higher cost.

PermLLM uses **Sinkhorn normalization** to relax each block's discrete permutation to a doubly stochastic matrix, combined with the **straight-through estimator (STE)** to harden via the Hungarian algorithm in the forward pass while propagating gradients through the soft matrix in the backward pass. The optimization objective directly minimizes the **output discrepancy** between the dense and sparse models using a cosine similarity loss, rather than maximizing a handcrafted saliency metric.

A custom **CUDA kernel** for the channel permutation operation achieves an 84× speedup over PyTorch, making the permutation overhead negligible at inference (0.039ms vs. 3.288ms for the full sparse GEMM pipeline in LLaMA-2 7B).

## Mathematical Form

**Channel Permutation Problem:**

Given a weight matrix $\mathbf{W} \in \mathbb{R}^{C_\text{out} \times C_\text{in}}$, find a permutation matrix $\mathbf{P} \in \{0,1\}^{C_\text{in} \times C_\text{in}}$ such that the reordered weight $\widehat{\mathbf{W}} = \mathbf{W}\mathbf{P}$ achieves improved accuracy after N:M pruning.

**Sinkhorn Relaxation:**

Each block's learnable parameter matrix $\mathbf{W}_P^i \in \mathbb{R}^{B \times B}$ is converted to a soft permutation via:

$$
S^0(\mathbf{X}) = \exp(\mathbf{X})
$$

$$
S^{l+1}(\mathbf{X}) = \mathcal{T}_c(\mathcal{T}_r(S^l(\mathbf{X})))
$$

$$
\widehat{\mathbf{P}}_i = S^L(\mathbf{W}_P^i / \tau)
$$

where $\mathcal{T}_r$ is row normalization, $\mathcal{T}_c$ is column normalization, and $\tau$ is a temperature parameter linearly decayed from 1 to 0.1 during training.

**STE Hardening:**

In the forward pass, the Hungarian algorithm extracts the closest hard permutation:

$$
\mathbf{P}_i = \arg\max_{\mathbf{P} \in \mathcal{P}} \text{Tr}(\mathbf{P}^\top \widehat{\mathbf{P}}_i)
$$

In the backward pass, the STE approximation $\partial \mathbf{P}_i / \partial \widehat{\mathbf{P}}_i \approx \mathbf{I}$ passes gradients through:

$$
\mathbf{P}_i \approx \mathbf{P}_i^{\text{hard}} + (\widehat{\mathbf{P}}_i - \widehat{\mathbf{P}}_i^{\text{detach}})
$$

**Block-wise Composition:**

The full permutation matrix is block-diagonal:

$$
\mathbf{P}_B = \text{diag}(\mathbf{P}_1, \mathbf{P}_2, \ldots, \mathbf{P}_{N_B})
$$

where $N_B = C_\text{in} / B$ is the number of blocks. The reordered weight is:

$$
\widehat{\mathbf{W}}_B = \mathbf{W} \mathbf{P}_B
$$

**Pruning with Permutation:**

After permutation, apply any one-shot pruning method (e.g., Wanda). Let $\mathbf{S}$ denote the saliency matrix and $\mathbf{M}$ the N:M mask. The permuted saliency is:

$$
\widehat{\mathbf{S}} = \mathbf{S} \mathbf{P}_B
$$

The mask $\mathbf{M}$ is computed from $\widehat{\mathbf{S}}$:

$$
\arg\max_{\mathbf{M}} \sum_{i=0}^{C_\text{out}} \sum_{k=0}^{C_\text{in}/M} \sum (\mathbf{M} \odot \widehat{\mathbf{S}})_{i, kM:(k+1)M}, \quad \text{s.t.} \; \|\mathbf{M}_{i, kM:(k+1)M}\|_0 = M - N
$$

**Soft Mask for Gradient Flow:**

To enable backpropagation through the non-differentiable $\arg\max$ mask selection, a soft mask is used in the backward pass:

$$
\widetilde{\mathbf{M}}_{i, kM:(k+1)M} = \text{Softmax}(\widehat{\mathbf{S}}_{i, kM:(k+1)M})
$$

**Activation Permutation Propagation:**

The preceding layer's pruned output must be reordered to match. Let $\mathbf{P}_{l,B}^*$ be the learned permutation for layer $l$:

$$
\widehat{\mathbf{W}}_{l-1}'' = \mathbf{P}_{l,B}^* \widehat{\mathbf{W}}_{l-1}'
$$

This is a row-wise operation that preserves the N:M sparsity of $\widehat{\mathbf{W}}_{l-1}'$.

**Optimization Objective:**

PermLLM minimizes the cosine similarity loss between dense and sparse model outputs:

$$
\mathcal{L}_\text{cosine}(\mathbf{y}, \widetilde{\mathbf{y}}) = 1 - \frac{\mathbf{y} \cdot \widetilde{\mathbf{y}}}{\|\mathbf{y}\| \cdot \|\widetilde{\mathbf{y}}\|}
$$

where $\mathbf{y}$ is the dense model output and $\widetilde{\mathbf{y}}$ is the permuted-and-pruned model output. Only the block-wise cost matrices $\{\mathbf{W}_P^i\}$ are trainable; all weights remain frozen.

## Complexity

| Operation | Full-Matrix LCP | Block-wise LCP (PermLLM) |
|-----------|----------------|--------------------------|
| Learnable parameters per layer | $C_\text{in}^2$ | $N_B \times B^2 = C_\text{in} \times B$ |
| Hungarian hardening | $O(C_\text{in}^3)$ | $O(N_B \cdot B^3) = O(C_\text{in} \cdot B^2)$ |
| Sinkhorn per iteration | $O(C_\text{in}^2)$ | $O(N_B \cdot B^2) = O(C_\text{in} \cdot B)$ |
| Inference permutation | $O(C_\text{in})$ index remap | $O(C_\text{in})$ index remap |

**Concrete numbers (LLaMA-2 7B, $C_\text{in} = 4096$, $B = 64$):**
- Parameters per layer: $4096 \times 64 = 262{,}144$ (vs. $4096^2 = 16{,}777{,}216$ for full)
- Hungarian cost: $O(4096 \times 64^2) = O(16.8\text{M})$ vs. $O(4096^3) = O(68.7\text{B})$ — a **4096×** reduction
- Training time: ~2.5 hours (7B model, 4 GPUs) or ~5.5 hours (13B model, 8 GPUs)

**Inference overhead:**
- Custom CUDA kernel: 0.039ms for channel permutation (84× faster than PyTorch)
- Overall model speedup with 2:4 sparsity + CP: 1.67× on LLaMA-2 7B

**Memory:** Only $\mathbf{W}_P^i$ matrices are stored during training (frozen weights are not duplicated). At inference, only an integer index array per block is needed.

## Applicability

- **Post-training LLM pruning:** Primary application. Plugs into Wanda, RIA, SparseGPT as a drop-in permutation optimizer. On LLaMA-2 7B (2:4), reduces Wikitext-2 perplexity from 12.16 (Wanda) to 9.39 (PermLLM+Wanda) and from 11.30 (RIA) to 9.60 (PermLLM+RIA).
- **N:M sparse inference acceleration:** The block-wise permutation preserves N:M structure and composes with Sparse Tensor Core acceleration. The custom CUDA kernel makes the permutation overhead negligible.
- **Multiple LLM families:** Validated on LLaMA (7B–13B), LLaMA-2 (7B–13B), LLaMA-3.1 (8B), Qwen-2.5 (7B), and OPT (6.7B), showing consistent improvements across architectures.
- **Composable with weight update methods:** When combined with AdmmPruner (a weight-update method), PermLLM achieves the best overall performance: 47.40 average zero-shot accuracy on LLaMA-2 7B.
- **Hierarchical sparsity (V:N:M):** The block-wise approach is naturally compatible with V:N:M hierarchical sparsity, where column pruning + 2:4 operates at a block level.
- **SSM state permutation:** The block-wise approach could be applied to permute state channels in diagonal SSMs (like S4D, Mamba) where the state dimension is moderate ($N = 16$–$256$).

## Limitations

- Block-wise permutation cannot move channels across block boundaries — inter-block reordering is precluded, limiting global optimization
- Default block size $B = 64$ is a heuristic trade-off; $B = 128$ gives marginal accuracy gains but doubles training time
- Requires calibration data (128 samples of 1024 tokens) — sensitivity to calibration distribution not extensively studied
- STE gradient approximation ($\partial \mathbf{P} / \partial \widehat{\mathbf{P}} \approx \mathbf{I}$) introduces gradient bias; the soft Sinkhorn matrix and hard Hungarian output may diverge significantly at high temperatures
- Only 5 Sinkhorn iterations used by default — may not fully converge to a doubly stochastic matrix, though experiments show this is sufficient
- Cosine similarity loss optimizes for output alignment but does not directly target downstream task metrics; the cross-entropy + distillation loss in the bipartite matching approach (arXiv 2601.22980) may be more principled
- Custom CUDA kernel is architecture-specific (developed for NVIDIA A100); portability to other accelerators not guaranteed

## Implementation Notes

```python
import torch
from scipy.optimize import linear_sum_assignment

def sinkhorn_block(W_P, tau=1.0, n_iters=5):
    """
    Sinkhorn normalization for a single block's learnable matrix.

    Args:
        W_P: (B, B) learnable parameter matrix
        tau: temperature (annealed 1.0 -> 0.1)
        n_iters: Sinkhorn iterations (default 5)

    Returns:
        P_soft: (B, B) doubly stochastic matrix
    """
    log_alpha = W_P / tau
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)

def hungarian_ste(P_soft):
    """
    Harden soft permutation via Hungarian + STE.

    Forward: hard permutation from Hungarian.
    Backward: gradient flows through P_soft.
    """
    cost = -P_soft.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    P_hard = torch.zeros_like(P_soft)
    P_hard[row_ind, col_ind] = 1.0

    # STE: hard in forward, soft gradient in backward
    return P_hard + (P_soft - P_soft.detach())

def blockwise_permute(W, W_P_list, tau=1.0, n_iters=5):
    """
    Block-wise learnable channel permutation.

    Args:
        W: (C_out, C_in) weight matrix (frozen)
        W_P_list: list of N_B learnable (B, B) matrices
        tau: Sinkhorn temperature
        n_iters: Sinkhorn iterations

    Returns:
        W_permuted: (C_out, C_in) reordered weight
    """
    C_out, C_in = W.shape
    B = W_P_list[0].shape[0]
    N_B = C_in // B

    P_blocks = []
    for i in range(N_B):
        P_soft = sinkhorn_block(W_P_list[i], tau, n_iters)
        P_hard_ste = hungarian_ste(P_soft)
        P_blocks.append(P_hard_ste)

    P_B = torch.block_diag(*P_blocks)  # (C_in, C_in)
    W_permuted = W @ P_B
    return W_permuted

# PermLLM training loop (sketch):
# Only W_P_list parameters are trainable; all model weights frozen
# for epoch in range(num_epochs):
#     tau = 1.0 - epoch * 0.9 / num_epochs  # decay 1.0 -> 0.1
#     for x_cal in calibration_loader:
#         W_perm = blockwise_permute(W, W_P_list, tau)
#         mask = wanda_mask(W_perm, x_cal)  # N:M mask
#         y_dense = dense_model(x_cal)
#         y_sparse = sparse_forward(W_perm * mask, x_cal)
#         loss = 1 - cosine_similarity(y_dense, y_sparse)
#         loss.backward()
#         optimizer.step()  # update only W_P_list
```

## References

- Zou, L., Yin, S., Pei, Z., Ho, T.-Y., Farnia, F. & Yu, B. (2025). PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models. NeurIPS 2025. arXiv:2510.10136.
- Pool, J. & Yu, C. (2021). Channel Permutations for N:M Sparsity. NeurIPS 2021.
- Sun, M. et al. (2023). Wanda: A Simple and Effective Pruning Approach for Large Language Models. arXiv:2306.11695.
- Zhang, Y. et al. (2024). Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models. ICLR 2024.
- Mena, G. et al. (2018). Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018.
- Knight, P. (2008). The Sinkhorn-Knopp Algorithm: Convergence and Applications. SIMAX.
- Kuhn, H. (1955). The Hungarian Method for the Assignment Problem.
