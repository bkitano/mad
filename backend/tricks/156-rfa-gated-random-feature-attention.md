# 156: RFA-Gate (Gated Random Feature Attention)

**Category**: approximation
**Gain type**: efficiency
**Source**: Peng, Pappas, Yogatama, Schwartz, Smith & Kong (2021), "Random Feature Attention" (ICLR 2021)
**Paper**: [papers/random-feature-attention.pdf]
**Documented**: 2026-02-15

## Description

Random Feature Attention (RFA) is a linear-time attention mechanism that approximates softmax attention using random feature maps, with a key innovation: an **optional learned gating mechanism** (RFA-Gate) that introduces exponential recency bias into the causal linear attention recurrence. The gating converts the vanilla linear attention's flat-weighted history into an exponentially decaying memory, analogous to the forget gate in LSTMs.

The core observation is that causal linear attention can be written as a recurrence: the hidden state $(\mathbf{S}_t, \mathbf{z}_t)$ accumulates the outer products $\phi(\mathbf{k}_i) \otimes \mathbf{v}_i$ and normalizing sums $\phi(\mathbf{k}_i)$. Without gating, all past tokens contribute equally to the hidden state, which is problematic for language modeling where recent context is more relevant. RFA-Gate adds a learned scalar gate $g_t = \sigma(\mathbf{w}_g \cdot \mathbf{x}_t + b_g)$ that exponentially discounts older contributions:

$$
\mathbf{S}_t = g_t \mathbf{S}_{t-1} + (1 - g_t)\phi(\mathbf{k}_t) \otimes \mathbf{v}_t
$$

This is a strictly more expressive mechanism than vanilla linear attention — it can learn when to retain and when to forget history, while maintaining $O(Dd)$ constant memory and $O(1)$ per-step computation (no growing KV-cache). Unlike FAVOR+ which focuses on variance reduction in the feature map, RFA-Gate focuses on the **recurrence dynamics** and shows that gating is more important than the specific feature map choice for language modeling quality.

## Mathematical Form

**Core Operation — Random Feature Attention:**

Starting from the softmax attention kernel decomposition:

$$
\exp(\mathbf{x} \cdot \mathbf{y} / \sigma^2) = \exp\left(\frac{\|\mathbf{x}\|^2}{2\sigma^2} + \frac{\|\mathbf{y}\|^2}{2\sigma^2}\right)\exp\left(-\frac{\|\mathbf{x} - \mathbf{y}\|^2}{2\sigma^2}\right)
$$

$$
\approx \exp\left(\frac{\|\mathbf{x}\|^2}{2\sigma^2} + \frac{\|\mathbf{y}\|^2}{2\sigma^2}\right) \phi(\mathbf{x}) \cdot \phi(\mathbf{y})
$$

where $\phi : \mathbb{R}^d \to \mathbb{R}^{2D}$ is the random Fourier feature map (Theorem 1):

$$
\phi(\mathbf{x}) = \frac{1}{\sqrt{D}}\left[\sin(\mathbf{w}_1 \cdot \mathbf{x}), \ldots, \sin(\mathbf{w}_D \cdot \mathbf{x}), \cos(\mathbf{w}_1 \cdot \mathbf{x}), \ldots, \cos(\mathbf{w}_D \cdot \mathbf{x})\right]^\top
$$

with $\mathbf{w}_i \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I}_d)$.

**RFA Bidirectional Attention:**

$$
\text{RFA}(\mathbf{q}_t, \{\mathbf{k}_i\}, \{\mathbf{v}_i\}) = \frac{\phi(\mathbf{q}_t)^\top \sum_i \phi(\mathbf{k}_i) \otimes \mathbf{v}_i}{\phi(\mathbf{q}_t) \cdot \sum_j \phi(\mathbf{k}_j)}
$$

**RFA Causal Attention (recurrence form):**

The causal variant computes attention using running sums (prefix-sum/recurrence):

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t) \otimes \mathbf{v}_t, \quad \mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t)
$$

$$
\text{RFA}(\mathbf{q}_t, \{\mathbf{k}_i\}_{i \leq t}, \{\mathbf{v}_i\}_{i \leq t}) = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}{\phi(\mathbf{q}_t) \cdot \mathbf{z}_t}
$$

where $\mathbf{S}_t \in \mathbb{R}^{2D \times d}$ and $\mathbf{z}_t \in \mathbb{R}^{2D}$.

**Key Definitions:**

- $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t \in \mathbb{R}^d$ — query, key, value at timestep $t$
- $D$ — number of random features (typically $D = d$ or $D = 2d$)
- $\sigma^2$ — temperature / kernel bandwidth
- $\mathbf{w}_i \in \mathbb{R}^d$ — random projection vectors
- $\otimes$ — outer product

**RFA-Gate: Gated Recurrence (Eq. 7):**

$$
g_t = \sigma(\mathbf{w}_g \cdot \mathbf{x}_t + b_g)
$$

$$
\mathbf{S}_t = g_t\,\mathbf{S}_{t-1} + (1 - g_t)\,\phi(\mathbf{k}_t) \otimes \mathbf{v}_t
$$

$$
\mathbf{z}_t = g_t\,\mathbf{z}_{t-1} + (1 - g_t)\,\phi(\mathbf{k}_t)
$$

where $\mathbf{w}_g \in \mathbb{R}^d$ and $b_g \in \mathbb{R}$ are learned parameters, $\sigma(\cdot)$ is the sigmoid function, and $\mathbf{x}_t$ is the input representation at timestep $t$.

When $g_t \to 1$, the state retains history and ignores the current token. When $g_t \to 0$, the state resets to the current token. The effective memory span is $\sim 1/(1-g_t)$ tokens.

**Learned Feature Map Bandwidth (Eq. 8):**

Instead of fixing $\sigma$, RFA learns it per-dimension:

$$
\tilde{\mathbf{w}}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d), \quad \mathbf{w}_i = \boldsymbol{\sigma} \circ \tilde{\mathbf{w}}_i
$$

where $\boldsymbol{\sigma} \in \mathbb{R}^d$ is a learned vector and $\circ$ is elementwise product. This lifts the isotropic constraint on the Gaussian distribution, allowing the kernel to adapt per dimension.

## Complexity

| Operation | Softmax Attention | RFA / RFA-Gate |
|-----------|------------------|----------------|
| Training (parallel) | $O(L^2 d)$ | $O(LDd)$ |
| Decoding (per step) | $O(Ld)$ (KV cache) | $O(Dd)$ (constant) |
| Memory (decoding) | $O(Ld)$ (KV cache grows) | $O(Dd)$ (constant) |
| Memory (training) | $O(L^2)$ (attention matrix) | $O(LD + Dd)$ |

**Memory:** RFA stores $\mathbf{S}_t \in \mathbb{R}^{2D \times d}$ and $\mathbf{z}_t \in \mathbb{R}^{2D}$, totaling $O(Dd)$ — constant in sequence length. With $D = d = 64$, this is $2 \times 128 \times 64 + 128 = 16{,}512$ floats per head, vs $O(Ld)$ for a KV cache that grows linearly.

**Decoding speedup:** At inference, each decoding step is $O(Dd)$ instead of $O(Ld)$. For $L = 2048$ and $D = d = 64$, RFA decodes $\sim 12\times$ faster with $< 10\%$ of the memory.

**Gate overhead:** The gate $g_t$ requires one dot product $\mathbf{w}_g \cdot \mathbf{x}_t$ and a sigmoid — $O(d)$ per step, negligible relative to the $O(Dd)$ state update.

## Applicability

- **Autoregressive language models**: RFA-Gate-Gaussian outperforms the softmax Transformer baseline on WikiText-103 (29.4 vs 33.0 dev perplexity for small models; 30.5 vs 34.5 for large), making it one of the first linear attention methods to beat standard attention at language modeling
- **Machine translation**: RFA achieves comparable BLEU to softmax Transformers on WMT14 (EN-DE: 28.0 vs 28.1, EN-FR: 39.2 vs 39.0) while decoding $1.8$–$1.9\times$ faster
- **Long text classification**: Competitive on Long Range Arena (ListOps, IMDb, AAN), with $1.1$–$5.3\times$ speedup and much lower memory
- **Drop-in replacement**: Works in encoder self-attention (bidirectional RFA), decoder causal attention (gated recurrence), and cross-attention
- **Constant-memory infinite context**: In decoder mode, the recurrent state is fixed-size regardless of sequence length, enabling generation with unbounded context
- **GPU-friendly**: The gated recurrence is sequential (like all RNN-style attention), but the per-step computation is a small outer product update + matmul, which is well-suited to GPU. For training, the bidirectional RFA variant can be parallelized

## Limitations

- **Sequential training bottleneck**: The gated causal variant requires sequential processing during training (like an RNN), preventing full parallelization over the time dimension. The paper notes RFA trains ~15% slower than baseline due to feature map overhead
- **Gating is sequential**: Cannot be parallelized via prefix-sum/scan because the gate $g_t$ is input-dependent (unlike a fixed decay rate). This limits training throughput on GPUs compared to methods like Mamba that use input-dependent gates but with hardware-aware scan implementations
- **Not exactly softmax**: RFA learns a different attention pattern from softmax. The paper shows that training with one attention type and evaluating with the other gives poor results — it's not a faithful approximation of softmax but rather a different mechanism that works well in practice
- **Trigonometric features**: The original RFA uses sin/cos (trigonometric) features rather than the positive features of FAVOR+. These can produce negative kernel estimates. However, the Gaussian kernel variant (which normalizes Q, K to unit norm) empirically stabilizes training
- **Sensitivity to $D$**: Performance saturates at $D = d$ to $2d$; using $D > 2d$ gives marginal improvement. For very small $D$ (e.g., $D = 16$), quality degrades noticeably
- **Stateful training complication**: The "stateful" variant (carrying hidden state across mini-batches) gives the best perplexity but requires careful implementation for distributed training

## Implementation Notes

```python
import torch
import torch.nn as nn
import math

class RFAGate(nn.Module):
    """Random Feature Attention with learned gating.

    Key innovation: exponential recency bias via learned scalar gate,
    enabling constant-memory causal attention with locality bias.
    """
    def __init__(self, d_model, n_features=None, n_heads=1):
        super().__init__()
        self.d = d_model // n_heads
        self.D = n_features or self.d  # random feature dimension
        self.n_heads = n_heads

        # Learned bandwidth (Eq. 8): sigma per dimension
        self.sigma = nn.Parameter(torch.ones(self.d))

        # Gate parameters (Eq. 7)
        self.w_gate = nn.Linear(d_model, n_heads, bias=True)

        # Random projection vectors (fixed after init, or redrawn)
        # w_i ~ N(0, I_d), then scaled by learned sigma
        self.register_buffer(
            'w_proj',
            torch.randn(self.D, self.d)  # (D, d)
        )

    def feature_map(self, x):
        """Gaussian kernel random feature map.

        phi(x) = 1/sqrt(D) [sin(w1.x), ..., sin(wD.x),
                             cos(w1.x), ..., cos(wD.x)]

        Args:
            x: (..., d) input
        Returns:
            (..., 2D) random features
        """
        # Scale projection by learned sigma
        W = self.w_proj * self.sigma.unsqueeze(0)  # (D, d)
        proj = x @ W.T  # (..., D)
        return torch.cat([torch.sin(proj), torch.cos(proj)],
                        dim=-1) / math.sqrt(self.D)

    def forward_bidirectional(self, Q, K, V):
        """Bidirectional RFA (for encoder self-attention / cross-attention).

        O(LDd) time, O(LD + Dd) memory.
        """
        phi_q = self.feature_map(Q)  # (L, 2D)
        phi_k = self.feature_map(K)  # (L, 2D)

        # Norm factor for unbiased estimation
        q_norm = torch.exp(
            (Q ** 2).sum(-1, keepdim=True) / (2 * self.sigma.pow(2).sum())
        )
        k_norm = torch.exp(
            (K ** 2).sum(-1, keepdim=True) / (2 * self.sigma.pow(2).sum())
        )

        # Linear attention: phi(Q) @ (phi(K)^T @ V) / phi(Q) @ phi(K)^T @ 1
        S = phi_k.T @ V            # (2D, d)
        z = phi_k.sum(dim=0)       # (2D,)

        num = phi_q @ S            # (L, d)
        den = phi_q @ z            # (L,)

        return num / den.unsqueeze(-1)

    def forward_causal_gated(self, Q, K, V, x_input):
        """Gated causal RFA (for decoder causal attention).

        Sequential recurrence with learned gate for recency bias.
        O(LDd) time, O(Dd) memory per step (constant in L).

        Args:
            Q, K, V: (L, d) query, key, value
            x_input: (L, d_model) input for computing gate
        Returns:
            (L, d) attention output
        """
        L, d = Q.shape
        D2 = 2 * self.D

        phi_q = self.feature_map(Q)  # (L, 2D)
        phi_k = self.feature_map(K)  # (L, 2D)

        # Compute all gates at once (can be parallelized)
        gates = torch.sigmoid(self.w_gate(x_input))  # (L, n_heads)
        g = gates[:, 0]  # (L,) for single head

        # Sequential recurrence with gating
        S = torch.zeros(D2, d, device=Q.device)  # hidden state
        z = torch.zeros(D2, device=Q.device)      # normalizer state
        outputs = []

        for t in range(L):
            gt = g[t]
            # Gated update (Eq. 7)
            S = gt * S + (1 - gt) * torch.outer(phi_k[t], V[t])
            z = gt * z + (1 - gt) * phi_k[t]

            # Output
            num = phi_q[t] @ S     # (d,)
            den = phi_q[t] @ z     # scalar
            outputs.append(num / (den + 1e-6))

        return torch.stack(outputs, dim=0)  # (L, d)
```

## References

- Peng, H., Pappas, N., Yogatama, D., Schwartz, R., Smith, N. A., & Kong, L. (2021). Random Feature Attention. ICLR 2021. arXiv:2103.02143.
- Rahimi, A. & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. NeurIPS 2007.
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.
- Choromanski, K., Likhosherstov, V., et al. (2021). Rethinking Attention with Performers. ICLR 2021.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
