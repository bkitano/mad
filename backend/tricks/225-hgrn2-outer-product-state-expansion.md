# 225: HGRN2 Outer-Product State Expansion

**Category**: parallelization
**Gain type**: expressivity
**Source**: Qin, Yang, Sun, Shen, Li, Sun & Zhong, "HGRN2: Gated Linear RNNs with State Expansion" (COLM 2024)
**Paper**: papers/hgrn2-state-expansion.pdf
**Documented**: 2026-02-15

## Description

HGRN2 introduces a simple yet powerful trick for gated linear RNNs: replace the elementwise (Hadamard) product in the input gate with an **outer product**, expanding the hidden state from a vector $h_t \in \mathbb{R}^d$ to a matrix $H_t \in \mathbb{R}^{d \times d}$ without adding any new parameters. This dramatically increases the recurrent state's memory capacity — from $O(d)$ to $O(d^2)$ effective state size — while maintaining the linear recurrence structure needed for parallel training.

The critical insight is that this outer-product expansion makes the recurrence form **identical to Gated Linear Attention (GLA)**, enabling HGRN2 to directly leverage GLA's highly optimized chunkwise training kernels (which use tensor cores via matrix multiplications). This means HGRN2 inherits hardware-efficient training for free, unlike parametric state expansion methods (low-rank, Kronecker product, etc.) that require elementwise operations in expanded dimensions and cannot use tensor cores.

The correspondence between HGRN2 and GLA is: output gate $\leftrightarrow$ query, input gate $(1-f)$ $\leftrightarrow$ key, input vector $\leftrightarrow$ value, forget gate $\leftrightarrow$ decay factor. HGRN2 saves parameters over GLA by tying the input gate to the forget gate (as $1-f_t$) instead of learning an independent key projection.

## Mathematical Form

**HGRN1 (vector state, Eq. 1-2 from paper):**

$$
g_t = \sigma(U x_t + b_u), \quad i_t = \tau(V x_t + b_v), \quad o_t = \sigma(W x_t + b_w)
$$

$$
f_t^i = \beta^i + (1 - \beta^i) \odot g_t^i
$$

$$
h_t^i = f_t^i \odot h_{t-1}^i + (1 - f_t^i) \odot i_t^i \in \mathbb{R}^d
$$

$$
y_t = h_t \odot o_t
$$

**HGRN2 (matrix state via outer product, Eq. 3 from paper):**

$$
H_t = H_{t-1} \cdot \mathrm{Diag}\{f_t\} + i_t \otimes (1 - f_t) \in \mathbb{R}^{d \times d}
$$

$$
y_t = H_t \cdot o_t \in \mathbb{R}^d
$$

where:
- $\mathrm{Diag}\{f_t\}$ creates a diagonal matrix from the forget gate vector $f_t$
- $\otimes$ denotes the outer product: $i_t \otimes (1 - f_t) \in \mathbb{R}^{d \times d}$
- $\cdot$ denotes matrix-vector or matrix-matrix multiplication
- The matrix multiply $H_{t-1} \cdot \mathrm{Diag}\{f_t\}$ scales each column of $H_{t-1}$ by the corresponding element of $f_t$

**Multi-head variant (practical implementation):**

With $H$ heads, per-head dimension $d_h = d/H$, and expansion ratio $n = d_h$:

$$
H_t^{(h)} = H_{t-1}^{(h)} \cdot \mathrm{Diag}\{f_t^{(h)}\} + i_t^{(h)} \otimes (1 - f_t^{(h)}) \in \mathbb{R}^{d_h \times d_h}
$$

$$
y_t^{(h)} = H_t^{(h)} \cdot o_t^{(h)} \in \mathbb{R}^{d_h}
$$

**Correspondence to GLA (Table 4 from paper):**

| HGRN2 | GLA |
|-------|-----|
| $o_t$ (output gate) | $q_t$ (query vector) |
| $1 - f_t$ (input gate) | $k_t$ (key vector) |
| $i_t$ (input vector) | $v_t$ (value vector) |
| $f_t$ (forget gate) | $\alpha_t$ (decay factor) |
| — | $o_t$ (output gate, omitted in HGRN2) |

**Key Definitions:**

- $H_t \in \mathbb{R}^{d \times d}$ — matrix-valued hidden state (expanded from vector)
- $f_t \in \mathbb{R}^d$ — forget gate with learned lower bound $\beta^i$
- $i_t \in \mathbb{R}^d$ — input vector (SiLU activation)
- $o_t \in \mathbb{R}^d$ — output gate (sigmoid)
- $\beta \in \mathbb{R}^{L \times d}$ — learnable per-layer forget gate lower bound (monotonically increasing across layers)
- $d_h = d/H$ — per-head dimension (practical expansion ratio)

## Complexity

| Operation | HGRN1 (vector state) | HGRN2 (matrix state) |
|-----------|---------------------|---------------------|
| State size per step | $O(d)$ | $O(d^2/H)$ per head |
| Recurrence (naive sequential) | $O(BNd)$ | $O(BNd^2)$ |
| Recurrence (multi-head) | $O(BNd)$ | $O(BNd^2/H)$ |
| Training (chunkwise parallel) | Not tensor-core friendly | $O(BNd^2/H)$ with tensor cores |
| Parameters | $O(d^2)$ | $O(d^2)$ (same — no new params!) |

**Memory:** Per-head state is $d_h \times d_h$ matrix. With $d_h = 128$ and $H = d/128$ heads, total state size is $H \cdot d_h^2 = d \cdot d_h = 128d$ per layer, compared to $d$ for HGRN1.

**GPU considerations:**
- The outer product + matrix multiply formulation maps directly to GEMMs, enabling **tensor core acceleration**
- HGRN2 leverages GLA's chunkwise parallel algorithm: intra-chunk computation is quadratic attention (matmul), inter-chunk is linear state propagation
- Unlike Mamba's selective scan (which cannot be expressed as matmul), HGRN2's recurrence is a linear attention variant that benefits from tensor cores
- The FlashLinearAttention kernels from the `fla` library provide optimized implementations
- Training throughput comparable to Transformers and better than Mamba at large expansion ratios

## Applicability

- **Language modeling**: HGRN2 matches or outperforms Mamba, GLA, RetNet, and RWKV-4 at 1.3B and 2.7B parameter scales on SlimPajama (100B tokens). At 7B scale (300B tokens on The Pile), competitive with LLaMA.
- **In-context recall (MQAR)**: State expansion significantly improves associative recall capability — HGRN2 with $d_h = 128$ outperforms HGRN1 across all model dimensions and sequence lengths.
- **Long-context retrieval**: HGRN2 outperforms Mamba on Needle-in-a-Haystack tests due to larger state size, though still behind LLaMA-level Transformers.
- **Drop-in for GLA**: Since the recurrence is identical to GLA, HGRN2 can use GLA's entire software stack (kernels, parallelism strategies, sequence parallelism).
- **Efficient inference**: Like all linear RNNs, maintains constant $O(d_h^2)$ memory per step in recurrent mode — no KV cache growth.

## Limitations

- **Quadratic state growth**: The $d_h \times d_h$ per-head state matrix means state size grows quadratically with head dimension. Diminishing returns observed beyond $d_h = 128$ (Figure 2), so practical implementations cap the expansion.
- **No explicit training throughput numbers**: The paper defers to GLA's training kernel efficiency but does not report tokens/sec or wall-clock comparisons against Mamba or Transformers.
- **Input gate tied to forget gate**: HGRN2 constrains $\text{input gate} = 1 - f_t$, which saves parameters but reduces flexibility compared to GLA's independently parameterized key vector. This may limit performance in settings where independent gating is important.
- **Forget gate lower bound adds complexity**: The monotonically increasing $\beta$ across layers (enforced via cumulative softmax) adds a per-layer hyperparameter and initialization sensitivity.
- **Still behind Transformers on long-context**: Despite the larger state, HGRN2 at 7B does not match LLaMA on SCROLLs long-context benchmarks, suggesting the $O(d_h^2)$ state is still insufficient for some retrieval-heavy tasks.

## Implementation Notes

```python
import torch
import torch.nn.functional as F

class HGRN2Layer(torch.nn.Module):
    """
    HGRN2: Gated Linear RNN with outer-product state expansion.
    Equivalent to GLA with tied input/forget gates.
    """
    def __init__(self, d_model, d_head=128, num_heads=None):
        super().__init__()
        self.d_head = d_head
        self.num_heads = num_heads or d_model // d_head
        d = self.num_heads * d_head

        # Three projections: forget gate, input, output gate
        self.W_g = torch.nn.Linear(d_model, d)       # forget gate
        self.W_i = torch.nn.Linear(d_model, d)       # input vector
        self.W_o = torch.nn.Linear(d_model, d)       # output gate

        # Learnable lower bound for forget gate (per-layer)
        self.beta = torch.nn.Parameter(torch.zeros(d))

    def forward(self, x):
        B, T, D = x.shape

        # Compute gates (all independent of h_{t-1}!)
        g = torch.sigmoid(self.W_g(x))             # [B, T, d]
        f = self.beta.sigmoid() + (1 - self.beta.sigmoid()) * g  # forget with lower bound
        i = F.silu(self.W_i(x))                    # [B, T, d] input vector
        o = torch.sigmoid(self.W_o(x))             # [B, T, d] output gate

        # Reshape to multi-head: [B, T, H, d_h]
        f = f.view(B, T, self.num_heads, self.d_head)
        k = 1 - f                                   # input gate = 1 - forget gate (= GLA key)
        v = i.view(B, T, self.num_heads, self.d_head)  # = GLA value
        q = o.view(B, T, self.num_heads, self.d_head)  # = GLA query

        # The recurrence is now identical to GLA:
        # H_t = H_{t-1} * Diag{f_t} + v_t (outer) k_t
        # y_t = H_t @ q_t
        #
        # Use GLA's chunkwise parallel algorithm for training:
        # - Intra-chunk: quadratic attention via matmul (tensor core friendly)
        # - Inter-chunk: linear state propagation
        #
        # In practice, call: from fla.ops.gla import chunk_gla
        # y = chunk_gla(q, k, v, log_decay=torch.log(f))

        return y  # [B, T, D]
```

## References

- Qin, Z., Yang, S., Sun, W., Shen, X., Li, D., Sun, W. & Zhong, Y. "HGRN2: Gated Linear RNNs with State Expansion." COLM 2024. arXiv:2404.07904.
- GitHub: https://github.com/OpenNLPLab/HGRN2
- Yang, S. et al. "Gated Linear Attention Transformers with Hardware-Efficient Training." ICML 2024. arXiv:2312.06635.
- Qin, Z. et al. "Hierarchically Gated Recurrent Neural Network for Sequence Modeling." NeurIPS 2023. (HGRN1)
- Dao, T. & Gu, A. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024. arXiv:2405.21060.
- Yang, S. & Zhang, Y. "FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism." https://github.com/sustcsonglin/flash-linear-attention
