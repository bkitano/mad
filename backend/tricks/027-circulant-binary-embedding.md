# 027: Circulant Binary Embedding

**Category**: approximation
**Gain type**: efficiency
**Source**: Yu et al. "Circulant Binary Embedding" (ICML 2014)
**Paper**: [papers/circulant-binary-embedding.pdf]
**Documented**: 2026-02-15

## Description

Circulant Binary Embedding (CBE) replaces the dense random projection matrix in binary hashing with a **circulant matrix**, reducing the time complexity of generating $d$-bit binary codes from $d$-dimensional data from $O(d^2)$ to $O(d \log d)$ and the space complexity from $O(d^2)$ to $O(d)$. This enables efficient dimensionality reduction and hashing for ultra-high-dimensional data (up to $d \sim 100$M) where dense projection matrices would require terabytes of storage.

The core idea: given input $\mathbf{x} \in \mathbb{R}^d$, the binary code is $h(\mathbf{x}) = \text{sign}(\mathbf{R}\mathbf{D}\mathbf{x})$ where $\mathbf{R} = \text{circ}(\mathbf{r})$ is a circulant matrix defined by a single vector $\mathbf{r}$, and $\mathbf{D}$ is a diagonal sign-flipping matrix. The circulant matrix-vector product $\mathbf{R}\mathbf{x}$ is equivalent to **circular convolution** $\mathbf{r} \circledast \mathbf{x}$, computable via FFT as $\mathcal{F}^{-1}(\mathcal{F}(\mathbf{r}) \circ \mathcal{F}(\mathbf{x}))$.

A key theoretical result is that despite the rows of the circulant matrix being dependent (each row is a cyclic shift of $\mathbf{r}$), the normalized Hamming distance of CBE has **nearly identical variance** to independent random projections. This means CBE preserves angular distances as well as full random projection (LSH), but at a fraction of the cost.

The paper also introduces **CBE-opt**, a data-dependent variant that learns the circulant projection via a novel **time-frequency alternating optimization**: the binary code matrix $\mathbf{B}$ is optimized in the time (spatial) domain, while the circulant vector $\mathbf{r}$ is optimized in the frequency domain via DFT, exploiting the fact that $\mathbf{R} = (1/d)\mathbf{F}_d^H \text{diag}(\mathcal{F}(\mathbf{r})) \mathbf{F}_d$.

## Mathematical Form

**Standard binary embedding:**

$$
h(\mathbf{x}) = \text{sign}(\mathbf{R}\mathbf{x}), \quad \mathbf{R} \in \mathbb{R}^{k \times d}
$$

where $\text{sign}(\cdot)$ returns element-wise $\pm 1$.

**Circulant Binary Embedding (CBE):**

$$
h(\mathbf{x}) = \text{sign}(\mathbf{R}\mathbf{D}\mathbf{x})
$$

where $\mathbf{R} = \text{circ}(\mathbf{r})$ is a $d \times d$ circulant matrix and $\mathbf{D} = \text{diag}(\pm 1)$ is a random diagonal sign-flip matrix (each entry $\pm 1$ with probability $1/2$).

**Circulant matrix:**

$$
\mathbf{R} = \text{circ}(\mathbf{r}) = \begin{bmatrix} r_0 & r_{d-1} & \cdots & r_1 \\ r_1 & r_0 & \cdots & r_2 \\ \vdots & & \ddots & \vdots \\ r_{d-1} & r_{d-2} & \cdots & r_0 \end{bmatrix}
$$

**FFT-based computation:**

Since circulant multiplication equals circular convolution:

$$
\mathbf{R}\mathbf{x} = \mathbf{r} \circledast \mathbf{x}
$$

By the convolution theorem:

$$
\mathcal{F}(\mathbf{R}\mathbf{x}) = \mathcal{F}(\mathbf{r}) \circ \mathcal{F}(\mathbf{x})
$$

Therefore:

$$
h(\mathbf{x}) = \text{sign}\left(\mathcal{F}^{-1}(\mathcal{F}(\mathbf{r}) \circ \mathcal{F}(\mathbf{x}))\right)
$$

For $k < d$ bits, take the first $k$ elements of the result.

**Angle preservation (randomized CBE):**

When $\mathbf{r} \sim \mathcal{N}(0, \mathbf{I})$, the expected normalized Hamming distance between codes of $\mathbf{x}_1, \mathbf{x}_2$:

$$
\mathbb{E}[\mathcal{H}_k(\mathbf{x}_1, \mathbf{x}_2)] = \frac{\theta}{\pi}
$$

where $\theta$ is the angle between $\mathbf{x}_1$ and $\mathbf{x}_2$, and:

$$
\mathcal{H}_k(\mathbf{x}_1, \mathbf{x}_2) = \frac{1}{k} \sum_{i=0}^{k-1} |\text{sign}(\mathbf{R}_{i \cdot} \mathbf{x}_1) - \text{sign}(\mathbf{R}_{i \cdot} \mathbf{x}_2)| / 2
$$

For independent projections, the variance is:

$$
\text{Var}(\mathcal{H}_k) = \frac{\theta(\pi - \theta)}{k\pi^2}
$$

Empirically, the circulant variance is **indistinguishable** from this independent-projection variance.

**DFT diagonalization of $\mathbf{R}$:**

$$
\mathbf{R} = \frac{1}{d} \mathbf{F}_d^H \text{diag}(\mathcal{F}(\mathbf{r})) \mathbf{F}_d
$$

where $\mathbf{F}_d$ is the $d$-dimensional DFT matrix and $\mathbf{F}_d^H = d \cdot \mathbf{F}_d^{-1}$.

**Learning objective (CBE-opt):**

$$
\min_{\mathbf{B}, \mathbf{r}} \|\mathbf{B} - \mathbf{X}\mathbf{R}^\top\|_F^2 + \lambda \|\mathbf{R}\mathbf{R}^\top - \mathbf{I}\|_F^2, \quad \text{s.t.} \quad \mathbf{R} = \text{circ}(\mathbf{r})
$$

where $\mathbf{X} \in \mathbb{R}^{n \times d}$ is the data matrix and $\mathbf{B} \in \{-1, +1\}^{n \times d}$ is the target binary code matrix.

- First term: minimize distortion from binarization
- Second term: encourage orthogonality to decorrelate bits

**Time-frequency alternating optimization:**

*For fixed $\mathbf{r}$* (time domain): Update $\mathbf{B}$ element-wise:

$$
B_{ij} = \begin{cases} +1 & \text{if } \mathbf{R}_{j \cdot} \mathbf{x}_i \geq 0 \\ -1 & \text{if } \mathbf{R}_{j \cdot} \mathbf{x}_i < 0 \end{cases}
$$

*For fixed $\mathbf{B}$* (frequency domain): Optimize $\tilde{\mathbf{r}} = \mathcal{F}(\mathbf{r})$ by decomposing into $\lfloor d/2 \rfloor$ independent 4th-order polynomial problems in the real and imaginary parts of $\tilde{\mathbf{r}}$, solvable via gradient descent on 2-variable polynomials with constant-time convergence.

The orthogonality penalty in frequency domain becomes:

$$
\|\mathbf{R}\mathbf{R}^\top - \mathbf{I}\|_F^2 = \|\tilde{\mathbf{r}}^H \circ \tilde{\mathbf{r}} - 1\|_2^2 = \||\Re(\tilde{\mathbf{r}})|^2 + |\Im(\tilde{\mathbf{r}})|^2 - 1\|_2^2
$$

**Key Definitions:**

- $\mathbf{r} \in \mathbb{R}^d$ — defining vector of the circulant projection matrix
- $\tilde{\mathbf{r}} = \mathcal{F}(\mathbf{r}) \in \mathbb{C}^d$ — Fourier transform of $\mathbf{r}$
- $\mathbf{D}$ — diagonal $\pm 1$ sign-flip matrix (preprocessing step)
- $k$ — number of output bits ($k \leq d$; for $k < d$, take first $k$ elements)
- $\theta$ — angle between input vectors (the quantity preserved by embedding)
- $\lambda$ — regularization weight for orthogonality penalty

## Complexity

| Operation | Full Projection | Bilinear Projection | Circulant (CBE) |
|-----------|----------------|--------------------|-----------------|
| Encode time | $O(d^2)$ | $O(d^{1.5})$ | $O(d \log d)$ |
| Storage | $O(d^2)$ | $O(d)$ | $O(d)$ |
| Learning time | $O(nd^2)$ | $O(nd^{1.5})$ | $O(nd \log d)$ |

**Concrete timings** (single 2.9 GHz CPU core):

| $d$ | Full projection | Bilinear | CBE |
|-----|----------------|----------|-----|
| $2^{17}$ | 544 sec | 2.85 sec | 1.11 sec |
| $2^{19}$ (1M) | — (OOM) | 37.7 sec | 37.7 sec |
| $2^{24}$ | — | $1.22 \times 10^4$ sec | $8.10 \times 10^2$ sec |
| $2^{27}$ (100M) | — | $2.68 \times 10^5$ sec | $8.15 \times 10^3$ sec |

**Memory:** $O(d)$ for the circulant vector $\mathbf{r}$ vs $O(d^2)$ for a full projection matrix. At $d = 10^6$, this is $O(10^6)$ vs $O(10^{12})$ — the difference between megabytes and terabytes.

**GPU:** Preliminary tests show up to 20x speedup over CPU due to highly optimized cuFFT.

## Applicability

- **Efficient random feature layers**: Replace dense random feature projection $\mathbf{R}\mathbf{x}$ in kernel approximation methods (Random Fourier Features, FAVOR+) with circulant projection for $O(d \log d)$ cost — directly applicable to linear attention
- **Embedding layers for retrieval**: Fast approximate nearest neighbor search in high-dimensional feature spaces (e.g., CLIP embeddings, LLM hidden states) via Hamming distance on binary codes
- **Structured random projections in transformers**: The circulant projection can replace the random feature matrix in FAVOR+ and other random-feature-based attention approximations, maintaining the JL property with $O(d \log d)$ projection cost
- **Hashing-based attention**: Binary codes from CBE enable ultra-fast attention via Hamming-distance-based retrieval of relevant keys, applicable to long-sequence models
- **Dimensionality reduction preprocessing**: Fast $O(d \log d)$ preprocessing for reducing high-dimensional inputs before feeding into neural network layers
- **Privacy-preserving embeddings**: Binary codes provide a form of information compression that can serve as a privacy mechanism

## Limitations

- **Sign-flip preprocessing required**: The diagonal $\mathbf{D}$ matrix (random sign flips) is essential — without it, a circulant projection of an all-ones vector would be constant, destroying information. This adds a $O(d)$ preprocessing step
- **Binarization introduces distortion**: The $\text{sign}(\cdot)$ operation is inherently lossy; the circulant structure can only minimize, not eliminate, the distortion. For very precise distance preservation, more bits are needed
- **Row dependence**: Unlike independent projections, all rows of $\text{circ}(\mathbf{r})$ are deterministic given $\mathbf{r}$. While variance analysis shows this is empirically negligible, theoretical guarantees are slightly weaker (Johnson-Lindenstrauss holds with worse constants)
- **Square matrix constraint**: The circulant matrix is $d \times d$, so for $k < d$ output bits, the first $k$ entries are taken (equivalent to frequency cutoff). For $k \ll d$, most FFT computation is wasted
- **Non-convex learning**: The CBE-opt objective is non-convex; the alternating optimization finds local minima. In practice, 5-10 iterations suffice for good solutions
- **Not directly applicable to non-metric tasks**: CBE preserves angular/cosine similarity; for tasks requiring other distance metrics, the embedding may be suboptimal

## Implementation Notes

```python
import torch
import torch.fft as fft

class CirculantBinaryEmbedding(torch.nn.Module):
    """Circulant Binary Embedding for fast dimensionality reduction.

    Generates k-bit binary codes from d-dimensional input in O(d log d).
    Can be used as a fast random projection layer or learned embedding.
    """

    def __init__(self, d, k=None, learnable=False):
        super().__init__()
        self.d = d
        self.k = k or d  # default: d-bit codes

        # Circulant defining vector
        if learnable:
            self.r = torch.nn.Parameter(torch.randn(d))
        else:
            self.register_buffer('r', torch.randn(d))

        # Random sign-flip diagonal (fixed preprocessing)
        self.register_buffer('signs', (torch.randint(0, 2, (d,)) * 2 - 1).float())

    def forward(self, x):
        """
        x: (batch, d) input vectors
        Returns: (batch, k) binary codes in {-1, +1}
        """
        # Step 1: Random sign flip (D @ x)
        x_flipped = x * self.signs  # (batch, d)

        # Step 2: Circulant multiplication via FFT
        # R @ x = IFFT(FFT(r) * FFT(x))
        r_fft = fft.fft(self.r)                    # (d,)
        x_fft = fft.fft(x_flipped, dim=-1)          # (batch, d)
        proj = fft.ifft(r_fft.unsqueeze(0) * x_fft, dim=-1).real  # (batch, d)

        # Step 3: Take first k components and binarize
        proj_k = proj[:, :self.k]

        if self.training:
            # Soft sign for gradient flow during learning
            return torch.tanh(proj_k * 10)
        else:
            return torch.sign(proj_k)

    def encode_batch(self, X):
        """Efficient batch encoding for retrieval."""
        with torch.no_grad():
            codes = self.forward(X)
            # Pack into uint8 for compact storage
            binary = (codes > 0).byte()
            return binary


def hamming_distance(codes1, codes2):
    """Fast Hamming distance between binary code matrices.

    codes1: (n1, k) binary tensor
    codes2: (n2, k) binary tensor
    Returns: (n1, n2) pairwise Hamming distances
    """
    # XOR and popcount via integer operations
    return (codes1.unsqueeze(1) != codes2.unsqueeze(0)).float().sum(dim=-1)


# Example: CBE as fast random feature layer for linear attention
class CirculantRandomFeatureAttention(torch.nn.Module):
    """Use CBE as structured random projection for FAVOR+-style attention."""

    def __init__(self, dim, num_features):
        super().__init__()
        self.dim = dim
        self.m = num_features
        # Circulant projection replaces dense random features
        self.r = torch.nn.Buffer(torch.randn(dim))
        self.signs = torch.nn.Buffer(
            (torch.randint(0, 2, (dim,)) * 2 - 1).float()
        )

    def random_feature_map(self, x):
        """Project via circulant matrix: O(d log d) instead of O(d*m)."""
        x_flipped = x * self.signs
        r_fft = fft.fft(self.r)
        x_fft = fft.fft(x_flipped, dim=-1)
        proj = fft.ifft(r_fft * x_fft, dim=-1).real[:, :self.m]
        # Apply softmax kernel approximation
        return torch.exp(proj - proj.max(dim=-1, keepdim=True).values)
```

## References

- Yu, F.X., Kumar, S., Gong, Y., Chang, S.-F. "Circulant Binary Embedding" (ICML 2014). arXiv:1405.3162
- Yu, F.X., Bhaskara, A., Kumar, S., Gong, Y., Chang, S.-F. "On Binary Embedding using Circulant Matrices" (JMLR 2017). arXiv:1511.06480
- Charikar, M. "Similarity Estimation Techniques from Rounding Algorithms" (STOC 2002)
- Hinrichs, A. & Vybiral, J. "Johnson-Lindenstrauss lemma for circulant matrices" (Random Structures & Algorithms, 2011)
- Choromanska, A., et al. "Binary Embeddings with Structured Hashed Projections" (ICML 2016)
- Katharopoulos, A., et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (ICML 2020)
