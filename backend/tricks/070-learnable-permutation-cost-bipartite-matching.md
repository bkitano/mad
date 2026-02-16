# 070: Learnable Permutation Cost with Differentiable Bipartite Matching

**Category**: decomposition
**Gain type**: expressivity
**Source**: Li, Liu, Li, Xu, Liu, Yin, Li & Barsoum (AMD, AAAI 2026)
**Paper**: [papers/learnable-permutation-bipartite-matching.pdf]
**Documented**: 2026-02-15

## Description

Standard N:M structured pruning preserves the top-$N$ magnitude weights per group of $M$, but when channels are arbitrarily ordered, important weights often cluster in the same groups — forcing the pruner to discard salient weights. Channel permutation reorders the input channels so that salient and non-salient weights are distributed more uniformly across pruning groups, increasing the total retained saliency. Prior methods use heuristic quality metrics (e.g., sum of kept magnitudes) to guide this reordering, but such handcrafted proxies can diverge from actual model performance: the permutation that maximizes the importance score may not minimize the pruning-induced loss.

This trick replaces heuristic permutation search with an **end-to-end learnable permutation framework** consisting of three components:

1. **Permutation cost predictor**: A lightweight learnable parameter matrix $\mathbf{C} \in \mathbb{R}^{d_\text{in} \times d_\text{in}}$ that encodes the cost of assigning input channel $i$ to position $j$, incorporating both sparsity alignment and semantic structure.

2. **Differentiable bipartite matching solver**: The discrete assignment problem $\min_{\mathbf{P} \in \mathcal{P}} \langle \mathbf{C}, \mathbf{P} \rangle$ is relaxed to entropy-regularized optimal transport over the Birkhoff polytope, solved by Sinkhorn iterations. Temperature annealing drives the soft solution toward a hard permutation during training.

3. **Sparsity optimization loss**: A composite of task-level cross-entropy and layer-wise distillation that directly optimizes the permutation with respect to pruned model performance, not a handcrafted saliency proxy.

At inference, the learned soft matrix is hardened to a true permutation via the Hungarian algorithm, and the permutation is absorbed by reordering the preceding layer's output channels — no additional runtime cost.

The key distinction from Sinkhorn permutation relaxation and OT4P is that the learnable object here is not the permutation itself but a **cost matrix** that defines the assignment problem. The Sinkhorn solver then extracts the optimal permutation from this cost. This decoupling means the cost predictor captures rich channel-interaction information that a direct permutation parameterization cannot.

The key distinction from PA-DST is the application context: PA-DST learns permutations for structured sparse *training* from scratch (with a Lipschitz penalty), while this method learns permutations for *post-training* one-shot pruning of pretrained Transformers. The cost predictor + bipartite matching formulation is designed for the post-training regime where only the permutation parameters are optimized with frozen weights.

## Mathematical Form

**Channel Permutation for Linear Layers:**

Given the $i$-th linear layer with weight $\mathbf{W}_i^\top \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$ and permutation matrix $\mathbf{P} \in \{0,1\}^{d_\text{in} \times d_\text{in}}$, the permuted weight is:

$$
\widehat{\mathbf{W}}_i^\top = \mathbf{P} \mathbf{W}_i^\top
$$

To maintain output consistency, the input activation must be transformed as $\widehat{\mathbf{A}}_i = \mathbf{A}_i \mathbf{P}^\top$, which is realized by propagating the permutation backward to the preceding layer:

$$
\widehat{\mathbf{W}}_{i-1} = \mathbf{W}_{i-1} \mathbf{P}^\top
$$

so that $\widehat{\mathbf{A}}_i = \mathbf{A}_{i-1} \widehat{\mathbf{W}}_{i-1} = \mathbf{A}_{i-1} \mathbf{W}_{i-1} \mathbf{P}^\top = \mathbf{A}_i \mathbf{P}^\top$.

**Transformer binding constraints:**

In Transformer attention blocks, the input channel permutation of $\mathbf{W}_o$ becomes the binding permutation for $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$. In the FFN, the input channel permutation of $\mathbf{W}_\text{down}$ binds $\mathbf{W}_\text{up}$ and $\mathbf{W}_\text{gate}$.

**Permutation Cost Predictor:**

For each weight matrix $\mathbf{W} \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, introduce a learnable cost matrix $\mathbf{C} \in \mathbb{R}^{d_\text{in} \times d_\text{in}}$ where $C_{i,j}$ quantifies the cost of assigning input channel $i$ to position $j$. The cost is parameterized as a normalized learnable matrix.

**Differentiable Bipartite Matching:**

The discrete assignment problem:

$$
\min_{\mathbf{P} \in \mathcal{P}} \langle \mathbf{C}, \mathbf{P} \rangle
$$

is relaxed to the Birkhoff polytope $\mathcal{B}_N$:

$$
\mathcal{B}_N = \left\{ \mathbf{P} \in \mathbb{R}^{N \times N}_{\geq 0} \mid \mathbf{P}\mathbf{1} = \mathbf{1}, \; \mathbf{P}^\top \mathbf{1} = \mathbf{1} \right\}
$$

with entropy regularization:

$$
\min_{\mathbf{P} \in \mathcal{B}_N} \langle \mathbf{C}, \mathbf{P} \rangle + \varepsilon \sum_{i,j} P_{ij}(\log P_{ij} - 1)
$$

The solution has closed form $\mathbf{P} = \text{Diag}(u) \mathbf{K} \text{Diag}(v)$ where $\mathbf{K} = \exp(-\mathbf{C}/\varepsilon)$, and the scaling vectors $u, v$ are computed by Sinkhorn iterations (alternating row/column normalization in log-domain).

**Temperature Annealing:**

The soft permutation is:

$$
\widehat{\mathbf{P}} = S^L(\mathbf{W}_P / \tau)
$$

where $\tau$ is linearly decayed during training from 1.0 to a small value, driving $\widehat{\mathbf{P}}$ toward a hard permutation. At inference, the hard permutation is recovered by the Hungarian algorithm:

$$
\mathbf{P}^* = \arg\max_{\mathbf{P} \in \mathcal{P}} \text{Tr}(\mathbf{P}^\top \widehat{\mathbf{P}})
$$

**Group-wise Permutation (scalability):**

For large-dimensional weight matrices, the input channels are partitioned into $G = d_\text{in} / g$ non-overlapping groups of size $g$, and a separate permutation is learned within each group. This reduces:
- Parameters: from $d_\text{in}^2$ to $G \times g^2 = d_\text{in} \times g$
- Hungarian cost: from $O(d_\text{in}^3)$ to $O(G \cdot g^3) = O(d_\text{in} \cdot g^2)$

Default group number is $G = 4$.

**End-to-End Optimization Loss:**

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{task} + \alpha \, \mathcal{L}_\text{distill}
$$

where:

$$
\mathcal{L}_\text{task} = \text{CE}(f_{\text{perm+prune}}(x), \, y)
$$

$$
\mathcal{L}_\text{distill} = \sum_{\ell=1}^{L} \| h_\ell^{\text{orig}} - h_\ell^{\text{perm+prune}} \|_2^2
$$

The task loss ensures the permuted-and-pruned model matches ground truth; the distillation loss aligns intermediate features with the dense teacher, preserving semantic structure through the permutation.

## Complexity

| Operation | Heuristic CP (RIA/Plug-and-Play) | Learnable Permutation |
|-----------|----------------------------------|-----------------------|
| Permutation search | $O(d_\text{in}^3)$ Hungarian per layer (one-shot) | $O(E \cdot L_S \cdot d_\text{in}^2)$ Sinkhorn ($E$ epochs, $L_S$ iters) |
| Gradient through permutation | None (non-differentiable) | Flows through Sinkhorn iterations |
| Optimizes for | Handcrafted saliency metric | Actual task + distillation loss |
| Group-wise variant | $O(d_\text{in} \cdot g^2)$ | $O(E \cdot L_S \cdot d_\text{in} \cdot g)$ |
| Inference overhead | Zero (absorbed) | Zero (absorbed) |

**Memory:** $G \times g^2$ learnable parameters per weight tensor (with $G=4$: $0.25 \times d_\text{in}$ parameters per layer).

**Training time:** ~4 hours for ViT-Base/16 (20 epochs), ~10 hours for LLaMA-3.2-1B, ~40 hours for LLaMA-2-7B (20 epochs on 2 AMD MI250 GPUs).

## Applicability

- **Post-training N:M pruning of LLMs:** The primary application. Plugs into existing one-shot pruners (Wanda, RIA, SparseGPT) and improves their accuracy without weight updates. On LLaMA-3.2-1B at 2:4 sparsity, improves average zero-shot accuracy from 33.23% (Wanda) to 35.90%.
- **Vision Transformer pruning:** On ViT-Base/16 at 2:4 sparsity, achieves 67.9% top-1 (vs. 66.6% RIA, 65.8% Wanda). At 4:8 sparsity: 71.8% (vs. 71.4% RIA).
- **Vision-language models:** On Qwen2.5-VL-3B at 2:4 sparsity, achieves 38.1 MMMU score (vs. 37.3 RIA, 37.2 Wanda).
- **Coupled permutations in Transformers:** The framework handles the structural coupling between attention projections ($\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$ share input permutation from $\mathbf{W}_o$) and FFN layers ($\mathbf{W}_\text{up}, \mathbf{W}_\text{gate}$ share input permutation from $\mathbf{W}_\text{down}$).
- **Composable with PA-DST and V:N:M:** The cost predictor approach can be composed with any structured sparsity pattern, not just N:M.

## Limitations

- Training overhead is non-trivial: 40 hours for a 7B model on 2 GPUs, compared to zero additional cost for heuristic methods like Wanda
- Group-wise permutation ($G=4$) sacrifices some global reordering flexibility for scalability; accuracy only drops ~0.3 points but inter-group channel moves are precluded
- Requires a calibration dataset (128 samples of 1024 tokens from C4) — the permutation quality depends on calibration data representativeness
- The distillation loss requires forward passes through both the dense and sparse models simultaneously, doubling activation memory during training
- Only demonstrated for post-training pruning; not validated for sparse training from scratch (where PA-DST or STEAM would be more appropriate)
- Sinkhorn iterations (default $L_S = 5$) add overhead per training step; convergence to doubly stochastic may require more iterations for large blocks

## Implementation Notes

```python
import torch
from scipy.optimize import linear_sum_assignment

def sinkhorn_matching(cost_matrix, tau=1.0, n_iters=5):
    """
    Differentiable bipartite matching via entropy-regularized OT.

    Args:
        cost_matrix: (N, N) learnable cost matrix C
        tau: temperature (annealed from 1.0 down during training)
        n_iters: number of Sinkhorn iterations

    Returns:
        soft_perm: (N, N) doubly stochastic approximation
    """
    # Compute kernel K = exp(-C / tau)
    log_alpha = -cost_matrix / tau
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)

def harden_permutation(soft_perm):
    """Hungarian algorithm to extract hard permutation at inference."""
    cost = -soft_perm.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    P = torch.zeros_like(soft_perm)
    P[row_ind, col_ind] = 1.0
    return P

def groupwise_permutation(W, cost_predictors, G, tau=1.0):
    """
    Block-wise learnable permutation for a weight matrix.

    Args:
        W: (d_out, d_in) weight matrix
        cost_predictors: list of G learnable (g, g) cost matrices
        G: number of groups
        tau: Sinkhorn temperature

    Returns:
        W_permuted: (d_out, d_in) reordered weight
        P_block: (d_in, d_in) block-diagonal permutation
    """
    d_in = W.shape[1]
    g = d_in // G  # group size

    P_blocks = []
    for i in range(G):
        C_i = cost_predictors[i]  # (g, g) learnable cost
        P_soft = sinkhorn_matching(C_i, tau=tau)
        P_hard = harden_permutation(P_soft)
        # STE: use hard in forward, soft gradient in backward
        P_i = P_hard + (P_soft - P_soft.detach())
        P_blocks.append(P_i)

    P_block = torch.block_diag(*P_blocks)  # (d_in, d_in)
    W_permuted = W @ P_block.T
    return W_permuted, P_block

# Training loop (sketch):
# for epoch in range(20):
#     tau = 1.0 - epoch * (1.0 - tau_min) / num_epochs  # anneal
#     for x, y in dataloader:
#         W_perm, P = groupwise_permutation(W, costs, G, tau)
#         mask = wanda_mask(W_perm, x)  # N:M mask from Wanda
#         W_sparse = W_perm * mask
#         # Inverse permute for loss computation
#         y_hat = forward_with_sparse(W_sparse, P_inv @ x)
#         loss = CE(y_hat, y) + alpha * distill_loss(h_orig, h_sparse)
#         loss.backward()  # gradients flow through Sinkhorn to cost_predictors
```

## References

- Li, Z., Liu, J., Li, G., Xu, Y., Liu, Z., Yin, X., Li, D. & Barsoum, E. (2026). Learnable Permutation for Structured Sparsity on Transformer Models. AAAI 2026. arXiv:2601.22980.
- Pool, J. & Yu, C. (2021). Channel Permutations for N:M Sparsity. NeurIPS 2021.
- Zhang, Y., Bai, H., Lin, H., Zhao, J., Hou, L. & Cannistraci, C.V. (2024). Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models. ICLR 2024.
- Mena, G. et al. (2018). Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018.
- Kuhn, H. (1955). The Hungarian Method for the Assignment Problem.
- Sun, M. et al. (2023). Wanda: A Simple and Effective Pruning Approach for Large Language Models. arXiv:2306.11695.
