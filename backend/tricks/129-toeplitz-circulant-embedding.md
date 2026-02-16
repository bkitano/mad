# 129: Toeplitz-to-Circulant Embedding

**Category**: decomposition
**Gain type**: efficiency
**Source**: Classical numerical linear algebra; applied to neural networks by Qin et al. "Toeplitz Neural Network for Sequence Modeling" (ICLR 2023)
**Paper**: [papers/toeplitz-circulant-embedding.pdf]
**Documented**: 2026-02-15

## Description

A Toeplitz matrix $T \in \mathbb{R}^{n \times n}$ (constant along diagonals, defined by $2n-1$ parameters) can be embedded into a $2n \times 2n$ circulant matrix $C$, enabling $O(n \log n)$ matrix-vector multiplication via FFT. The key insight is that the product $Tx$ equals the first $n$ entries of $Cx_1$ where $x_1 = [x; 0_n]$ is a zero-padded version of $x$. Since circulant matrices are diagonalized by the DFT, the product $Cx_1$ reduces to FFT, element-wise multiply, IFFT.

This trick is the foundational bridge between **Toeplitz structure** (which arises naturally in sequence models via relative position encodings, convolutions, and state-space models) and **FFT-based computation** (which gives $O(n \log n)$ complexity). The Toeplitz Neural Network (TNN) uses this trick as its core computational primitive for token mixing, achieving state-of-the-art results on the Long-Range Arena benchmark while maintaining $O(nd \log n)$ time and $O(nd)$ space complexity.

Critically, the TNN paper shows that Toeplitz token mixing is a **unifying framework**: Transformers with relative position embedding, 1D CNNs, and state-space models (S4, DSS) are all special cases of Toeplitz matrix-vector products with different coefficient patterns.

## Mathematical Form

**Toeplitz Matrix:**

A Toeplitz matrix $T \in \mathbb{R}^{n \times n}$ has constant values along each diagonal:

$$
T_{ij} = t_{i-j}, \quad T = \begin{bmatrix} t_0 & t_{-1} & t_{-2} & \cdots & t_{-n+1} \\ t_1 & t_0 & t_{-1} & \ddots & \vdots \\ t_2 & t_1 & \ddots & \ddots & t_{-2} \\ \vdots & \ddots & \ddots & t_0 & t_{-1} \\ t_{n-1} & \cdots & t_2 & t_1 & t_0 \end{bmatrix} \in \mathbb{R}^{n \times n}
$$

requiring only $2n - 1$ parameters: $t_{-(n-1)}, \ldots, t_0, \ldots, t_{n-1}$.

**Circulant Embedding:**

Embed $T$ into a circulant matrix $C \in \mathbb{R}^{2n \times 2n}$ by defining the first column of $C$ as:

$$
c_k = \begin{cases} t_k, & 0 \leq k \leq n-1 \\ t_0, & k = n \\ t_{k-2n}, & n+1 \leq k \leq 2n-1 \end{cases}
$$

This yields a $2n \times 2n$ circulant matrix:

$$
C = \begin{bmatrix} C_1 & C_2 \\ C_3 & C_4 \end{bmatrix} \in \mathbb{R}^{2n \times 2n}, \quad C_1 = T
$$

**Projection to recover Toeplitz product:**

For vector $x \in \mathbb{R}^n$, define the zero-padded vector:

$$
x_1 = \begin{bmatrix} x \\ 0_n \end{bmatrix} \in \mathbb{R}^{2n}
$$

Then:

$$
Tx = [I_n \quad 0_{n \times n}] \cdot C x_1
$$

i.e., the Toeplitz product equals the first $n$ entries of the circulant product.

**FFT-based computation:**

Since every circulant matrix is diagonalized by the DFT matrix $F$:

$$
C = F^\top \Lambda F, \quad \Lambda = \text{diag}(\text{FFT}(c))
$$

The product becomes:

$$
Tx = [I_n \quad 0_{n \times n}] \cdot \text{IFFT}(\text{FFT}(c) \circ \text{FFT}(x_1))
$$

**Application to sequence modeling (TNN):**

The token mixing operation is:

$$
y = Tx \in \mathbb{R}^n
$$

where $T_{ij} = t_{i-j}$ encodes relative positional interactions. For $d$-dimensional sequences, apply independently per dimension:

$$
Y = TX, \quad X \in \mathbb{R}^{n \times d}
$$

**Relative Position Encoder (RPE):**

Rather than storing $2n-1$ learnable parameters (which depends on sequence length), use a small MLP:

$$
[t_{-(n-1)}, \ldots, t_{n-1}] = \text{RPE}(-(n-1), \ldots, (n-1))
$$

where $\text{RPE}: \mathbb{Z} \to \mathbb{R}^d$ is a lightweight fully-connected network with $K$ layers. This makes parameters independent of sequence length $n$.

**Exponential decay bias:**

To enable length extrapolation, apply decay:

$$
\tilde{t}_{i-j} = \lambda^{|i-j|} t_{i-j}, \quad \lambda \in [0, 1]
$$

This generalizes ALiBi (which applies $\exp(m|i-j|)$ to attention scores).

**Key Definitions:**

- $T \in \mathbb{R}^{n \times n}$ — Toeplitz matrix, defined by $2n-1$ parameters
- $C \in \mathbb{R}^{2n \times 2n}$ — circulant embedding of $T$
- $F$ — DFT matrix, $F_{st} = \exp(2\pi s t i / n)$
- $\text{RPE}$ — Relative Position Encoder, a small MLP mapping integer offsets to coefficients
- $\lambda$ — exponential decay rate for length extrapolation

## Complexity

| Operation | Dense Attention | Toeplitz (via Circulant Embedding) |
|-----------|----------------|-----------------------------------|
| Token mixing (per dim) | $O(n^2)$ | $O(n \log n)$ |
| Token mixing (all dims) | $O(n^2 d)$ | $O(nd \log n)$ |
| Parameters (per layer) | $O(n^2)$ or $O(d^2)$ | $O(nd)$ (with RPE: $O(d)$, independent of $n$) |

**Memory:** $O(nd)$ vs $O(n^2)$ for attention matrices.

**Speed (steps/sec on A6000, seq len 4K):** TNN achieves 9.90 vs Transformer 3.05, a ~3.2x speedup.

The 2x overhead of the circulant embedding (operating on $2n$ instead of $n$) is a constant factor absorbed into the $O(n \log n)$.

## Applicability

- **Sequence token mixing**: Direct replacement for attention in 1D sequence models — the core application of TNN. Achieves 74.97 average on LRA benchmark (vs 70.87 for S4, 57.37 for Transformer)
- **Language modeling**: Autoregressive and bidirectional (use lower-triangular Toeplitz for causal masking)
- **State-space models**: SSMs produce causal Toeplitz matrices (lower-triangular with $T_{i-j} = CA^{i-j-1}B$); this embedding is how S4/DSS efficiently compute their convolution kernels
- **1D convolutions**: A 1D CNN is a special case of Toeplitz multiplication with a banded (sparse) Toeplitz matrix
- **Image modeling**: Applied to DeiT architecture with comparable results to DeiT-Small
- **Any relative-position-based interaction**: The Toeplitz structure naturally encodes $f(i-j)$ relationships

## Limitations

- Toeplitz structure assumes **translation invariance** — interactions depend only on relative position $i-j$, not on content. This sacrifices the content-dependent attention of Transformers
- The $2\times$ size blowup from embedding ($n \to 2n$) wastes memory and computation on padding coefficients (mitigated by the Split FFT algorithm of Siron & Molesky 2024)
- For **causal (autoregressive)** models, need lower-triangular Toeplitz (zero out future positions), which adds complexity
- RPE network adds overhead: 6 layers of MLP with 64 hidden dims in the full model
- Exponential decay rate $\lambda$ is a sensitive hyperparameter — $\lambda = 0.99$ works well but $\lambda = 1$ (no decay) causes extrapolation failure (PPL explodes from 24 to 672)
- FFT does not benefit from Tensor Cores on modern GPUs (unlike GEMM-based attention)

## Implementation Notes

```python
import torch
import torch.fft as fft

class ToeplitzTokenMixer(torch.nn.Module):
    """Token mixing via Toeplitz matrix with circulant embedding."""

    def __init__(self, max_seq_len, d_model, decay=0.99):
        super().__init__()
        self.d_model = d_model
        self.decay = decay
        # RPE: small MLP mapping integer offsets -> d_model coefficients
        # Input: scalar position offset, Output: d_model values
        self.rpe = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, d_model)
        )

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        n = x.shape[1]

        # Generate Toeplitz coefficients via RPE
        # Positions: -(n-1), ..., 0, ..., (n-1)
        positions = torch.arange(-(n-1), n, device=x.device).float().unsqueeze(-1)
        t = self.rpe(positions)  # (2n-1, d_model)

        # Apply exponential decay
        abs_pos = positions.abs().squeeze(-1)
        decay_mask = self.decay ** abs_pos  # (2n-1,)
        t = t * decay_mask.unsqueeze(-1)  # (2n-1, d_model)

        # Build circulant embedding: first column of C
        # c = [t_0, t_1, ..., t_{n-1}, t_0, t_{-n+1}, ..., t_{-1}]
        c = torch.zeros(2 * n, self.d_model, device=x.device)
        c[:n] = t[n-1:]       # t_0 through t_{n-1}
        c[n] = t[n-1]         # t_0 again (wrap)
        c[n+1:] = t[:n-1]     # t_{-n+1} through t_{-1}

        # FFT-based multiplication
        # Zero-pad input: (batch, 2n, d_model)
        x_padded = torch.nn.functional.pad(x, (0, 0, 0, n))

        # FFT along sequence dimension
        x_fft = fft.fft(x_padded, dim=1)  # (batch, 2n, d_model)
        c_fft = fft.fft(c, dim=0)          # (2n, d_model)

        # Element-wise multiply and IFFT
        out = fft.ifft(x_fft * c_fft.unsqueeze(0), dim=1).real

        # Take first n entries (projection from circulant to Toeplitz)
        return out[:, :n, :]
```

## References

- Qin, Z., Han, X., Sun, W., He, B., Li, D., Li, D., Dai, Y., Kong, L., Zhong, Y. "Toeplitz Neural Network for Sequence Modeling" (ICLR 2023). arXiv:2305.04749
- Gray, R.M. "Toeplitz and Circulant Matrices: A Review" Foundations and Trends in Communications and Information Theory, 2006
- Siron, A. & Molesky, S. "A Split Fast Fourier Transform Algorithm for Block Toeplitz Matrix-Vector Multiplication" arXiv:2406.17981, 2024
- Ferreira, P.J.S.G. & Dominguez, M.E. "Trading-off matrix size and matrix structure: Handling Toeplitz equations by embedding on a larger circulant set" Digital Signal Processing, 2010
- Gu, A., Goel, K., Re, C. "Efficiently Modeling Long Sequences with Structured State Spaces" (ICLR 2022)
