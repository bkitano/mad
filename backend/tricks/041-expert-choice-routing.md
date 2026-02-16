# 041: Expert Choice Routing

**Category**: parallelization
**Gain type**: efficiency
**Source**: Zhou et al. (2022), "Mixture-of-Experts with Expert Choice Routing"
**Paper**: [papers/expert-choice-routing.pdf]
**Documented**: 2026-02-15

## Description

Invert the routing direction in Mixture-of-Experts (MoE) layers: instead of each **token** selecting its top-$k$ experts (*token choice*), let each **expert** select its top-$k$ tokens (*expert choice*). This simple transposition of the routing matrix guarantees perfect load balancing by construction (each expert processes exactly $k$ tokens), eliminates the need for auxiliary balancing losses, allows tokens to be processed by a variable number of experts (0 to all), and enables uniform compute kernels across experts. The result is >2x faster training convergence than GShard top-2 gating with the same computational budget, plus a ~20% step-time reduction due to eliminated load imbalance.

## Mathematical Form

**Token-to-Expert Affinity Scores:**

Given input token representations $X \in \mathbb{R}^{n \times d}$ and expert embeddings $W_g \in \mathbb{R}^{d \times e}$:

$$
S = \text{Softmax}(X \cdot W_g) \in \mathbb{R}^{n \times e}
$$

where $n$ is the total number of tokens in the batch, $d$ is the model dimension, and $e$ is the number of experts.

**Expert Choice — The Key Transposition:**

Instead of applying top-$k$ along the expert dimension (token choice), apply it along the **token dimension** (expert choice):

$$
G, I = \text{TopK}(S^\top, k) \qquad \text{where } S^\top \in \mathbb{R}^{e \times n}
$$

- $I \in \mathbb{R}^{e \times k}$ — index matrix where $I[i, j]$ is the $j$-th selected token for expert $i$
- $G \in \mathbb{R}^{e \times k}$ — gating weights for selected tokens per expert

**Expert Capacity (bucket size):**

$$
k = \frac{n \times c}{e}
$$

where $c$ is the *capacity factor* controlling average expert allocation per token. Setting $c = 2$ matches the computation of top-2 token-choice routing.

**Permutation and Expert Computation:**

The permutation matrix $P = \text{OneHot}(I) \in \mathbb{R}^{e \times k \times n}$ gathers tokens to experts:

$$
X_{\text{in}} = P \cdot X \in \mathbb{R}^{e \times k \times d}
$$

Each expert $i$ computes its FFN independently:

$$
X_e[i] = \text{GeLU}(X_{\text{in}}[i] \cdot W_1[i]) \cdot W_2[i]^\top
$$

where $W_1[i] \in \mathbb{R}^{d \times d'}$ and $W_2[i] \in \mathbb{R}^{d \times d'}$ are expert-specific parameters.

**Unshuffle (scatter back):**

The final output $X_{\text{out}} \in \mathbb{R}^{n \times d}$ is assembled using the permutation and gating matrices:

$$
X_{\text{out}}[l, d] = \sum_{i, j} P[i, j, l] \; G[i, j] \; X_e[i, j, d]
$$

Both $X_{\text{in}}$ and $X_{\text{out}}$ can be efficiently computed using **einsum** operations.

**Key Definitions:**

- $n$ — total tokens in batch ($= \text{batch\_size} \times \text{seq\_len}$)
- $e$ — number of experts
- $k$ — expert capacity (tokens per expert)
- $c$ — capacity factor ($c = 1$: each token sees 1 expert on average; $c = 2$: two experts on average)
- $S \in \mathbb{R}^{n \times e}$ — token-to-expert affinity (softmax-normalized)
- $P \in \mathbb{R}^{e \times k \times n}$ — one-hot permutation matrix (the "shuffle" operation)
- $G \in \mathbb{R}^{e \times k}$ — gating weights for weighted output combination

**Optional: Capped Expert Choice with Entropy Regularization:**

To limit the maximum number of experts per token, solve a constrained linear program:

$$
\max_A \; \langle S^\top, A \rangle + \lambda H(A)
$$

$$
\text{s.t.} \quad \sum_{j'} A[i, j'] = k; \quad \sum_{i'} A[i', j] \leq b; \quad 0 \leq A[i,j] \leq 1
$$

where $H(A) = -\sum_{i,j} A[i,j] \log A[i,j]$ is elementwise entropy and $b$ upper-bounds tokens per expert. Solved via Dykstra's alternating projections (100 iterations, $\lambda = 0.001$).

## Complexity

| Operation | Token-Choice Top-2 | Expert Choice (c=2) |
|-----------|--------------------|---------------------|
| Routing | $O(n \cdot e)$ | $O(n \cdot e)$ (same affinity computation) |
| Expert compute | $O(n \cdot 2 \cdot d \cdot d')$ | $O(n \cdot c \cdot d \cdot d')$ (same total) |
| Load balance | Auxiliary loss (imperfect) | **Perfect by construction** |
| Step time | Bottlenecked by most-loaded expert | **Uniform across experts** |
| Convergence | Baseline | **>2x faster** (same compute budget) |

**Memory:** Same parameter count as conventional MoE. The permutation matrix $P$ is sparse (one-hot) and stored as indices, so memory overhead is $O(e \cdot k)$ integers.

**Step-time savings:** ~20% faster per step than GShard top-2 due to elimination of load imbalance (no expert receives more tokens than its fixed capacity $k$).

## Applicability

- **MoE Transformer FFN layers:** Direct replacement for token-choice routing (Switch Transformer, GShard). Applied to every-other-layer MoE in standard Transformer architectures
- **Large-scale pretraining:** Validated at 8B/64E scale (143B total params, 9.8B activated) on 1.6T tokens; outperforms dense 8B baseline on 7/11 GLUE/SuperGLUE tasks
- **Scaling experts:** Consistent improvements when scaling from 16 to 128 experts with fixed expert size
- **Any sparse conditional computation:** The expert-choice principle applies wherever computation is routed to specialized sub-networks

## Limitations

- **Autoregressive inference:** At inference time with small batch sizes, the top-$k$ token selection per expert may not work well since future tokens are unavailable. Workaround: use large batches or group tokens from the same sequence
- **Variable token coverage:** Some tokens may receive zero experts (unprocessed) or many experts. In practice, ~74% of tokens receive 1-2 experts, ~23% receive 3-4, and ~3% receive >4 (with $c=2$, 64 experts)
- **Sequence-level causality:** The routing decision sees all tokens in the batch simultaneously, which conflicts with strict causal ordering in autoregressive training (though this is typically handled by the attention mask, not the FFN routing)
- **Memory footprint:** Total parameters still scale linearly with number of experts (e.g., 143B total for 8B activated), requiring distributed training infrastructure

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertChoiceMoE(nn.Module):
    """Expert Choice routing for MoE layers.

    Each expert selects its top-k tokens instead of each token selecting experts.
    Guarantees perfect load balance with no auxiliary loss.
    """
    def __init__(self, d_model: int, d_ff: int, n_experts: int, capacity_factor: float = 2.0):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

        # Expert embeddings for routing
        self.gate = nn.Linear(d_model, n_experts, bias=False)

        # Expert FFN parameters (batched)
        self.W1 = nn.Parameter(torch.randn(n_experts, d_model, d_ff))
        self.W2 = nn.Parameter(torch.randn(n_experts, d_ff, d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, T, D = x.shape
        n = B * T
        x_flat = x.view(n, D)  # flatten batch and sequence

        # Compute token-to-expert affinity
        S = F.softmax(self.gate(x_flat), dim=-1)  # (n, e)

        # === THE TRICK: Expert selects top-k tokens (transpose!) ===
        k = int(n * self.capacity_factor / self.n_experts)
        # TopK along token dimension for each expert
        G, I = torch.topk(S.T, k=k, dim=-1)  # G: (e, k), I: (e, k)

        # Gather tokens to experts using index matrix
        # X_in[i] = x_flat[I[i]] for each expert i
        X_in = x_flat[I]  # (e, k, D)

        # Expert computation (batched across experts)
        hidden = F.gelu(torch.bmm(X_in, self.W1))  # (e, k, d_ff)
        X_e = torch.bmm(hidden, self.W2)             # (e, k, D)

        # Scatter back with gating weights
        X_e = X_e * G.unsqueeze(-1)  # weight by gating scores
        X_out = torch.zeros_like(x_flat)
        # Accumulate (tokens may receive output from multiple experts)
        X_out.scatter_add_(0, I.reshape(-1, 1).expand(-1, D), X_e.reshape(-1, D))

        return X_out.view(B, T, D)
```

**Key implementation details:**
- The `S.T` transposition is the entire trick — top-$k$ along tokens instead of experts
- `scatter_add_` accumulates outputs for tokens routed to multiple experts (variable expert count)
- For TPU/GPU efficiency, the shuffle/unshuffle operations are implemented as einsum with the one-hot permutation matrix $P$
- No auxiliary load-balancing loss is needed — balance is guaranteed by design
- Setting $c = 2$ makes computation directly comparable to GShard top-2

## References

- Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A., Chen, Z., Le, Q., & Laudon, J. (2022). Mixture-of-Experts with Expert Choice Routing. NeurIPS 2022. arXiv:2202.09368.
- Fedus, W., Zoph, B., & Shazeer, N. (2021). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. arXiv:2101.03961.
- Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., & Chen, Z. (2020). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. arXiv:2006.16668.
- Du, N. et al. (2021). GLaM: Efficient Scaling of Language Models with Mixture-of-Experts. arXiv:2112.06905.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.
