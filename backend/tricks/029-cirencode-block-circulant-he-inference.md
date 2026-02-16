# 029: CirEncode: Block Circulant Nested Encoding for HE Inference

**Category**: kernel
**Gain type**: efficiency
**Source**: Xu et al. "PrivCirNet: Efficient Private Inference via Block Circulant Transformation" (NeurIPS 2024)
**Paper**: [papers/privcirnet-block-circulant-he.pdf]
**Documented**: 2026-02-15

## Description

CirEncode is a novel encoding algorithm that makes homomorphic encryption (HE)-based neural network inference dramatically faster by exploiting block circulant weight structure. The key observation is that **circulant matrix-vector multiplication is equivalent to 1D polynomial convolution**, which is the native operation of lattice-based HE schemes (BFV). By transforming DNN weight matrices into block circulant form and co-designing the HE encoding, CirEncode converts expensive general matrix multiplications (GEMMs) into HE-friendly convolutions, reducing both HE multiplications (HE-Pmult) and the extremely costly HE rotations (HE-Rot).

Standard HE-based inference encodes tensors as polynomials and evaluates linear layers via polynomial arithmetic. For general GEMMs, this requires many HE rotations (each ~10x slower than HE-Pmult) to accumulate partial products. CirEncode's insight is two-fold:

1. **Within a circulant block**: A $b \times b$ circulant weight matrix times a $b \times d_1$ input can be encoded as a single polynomial multiplication $\hat{w} \times \hat{x}$ via coefficient encoding, requiring only 1 HE-Pmult and 0 HE-Rot per block.

2. **Across circulant blocks**: Multiple circulant blocks are packed in parallel using SIMD diagonal encoding with DFT preprocessing, enabling simultaneous evaluation via a single SIMD polynomial multiplication.

The framework also includes **loss-aware initialization** (using diagonal Fisher information to determine layer sensitivity) and **latency-aware block size assignment** (ILP-based search for per-layer block sizes under a latency budget).

## Mathematical Form

**Block Circulant GEMM:**

Consider $Y = WX$ where $W \in \mathbb{Z}^{d_3 \times d_2}$ is block circulant with block size $b$, and $X \in \mathbb{Z}^{d_2 \times d_1}$.

$$
W = \begin{bmatrix} W_{00} & W_{01} & \cdots \\ W_{10} & W_{11} & \cdots \\ \vdots & & \ddots \end{bmatrix}, \quad W_{ij} \in \mathbb{Z}^{b \times b} \text{ circulant}
$$

**CirEncode within a circulant block (Theorem 1):**

Given circulant $W \in \mathbb{Z}^{b \times b}$ and input $X \in \mathbb{Z}^{b \times d_1}$ where $bd_1 \leq n$ (polynomial degree), define encoding functions:

$$
\pi_W(W): \quad \hat{w}[id_1] = W[i, 0], \quad \forall i \in [b], j \in [d_1]
$$

$$
\pi_X(X): \quad \hat{x}[id_1 + j] = X[i, j], \quad \forall i \in [b], j \in [d_1]
$$

Then the GEMM is evaluated by a single polynomial multiplication in $\mathbb{A}_n = \mathbb{Z}[X]/(X^n - 1)$:

$$
\hat{y} = \hat{w} \times \hat{x}, \quad Y[i, j] = \hat{y}[id_1 + j]
$$

This requires **1 HE-Pmult and 0 HE-Rot** per circulant block.

**CirEncode across blocks (Theorem 2):**

Given $M$ circulant weight matrices $W_0, \ldots, W_{M-1}$ and inputs $X_0, \ldots, X_{M-1}$, all $M$ products can be computed simultaneously via SIMD encoding:

$$
\langle \text{DFT}(\hat{y}_0) | \cdots | \text{DFT}(\hat{y}_{M-1}) \rangle_{\text{Coeff}} = \langle \text{DFT}(\hat{w}_0) | \cdots | \text{DFT}(\hat{w}_{M-1}) \rangle_{\text{SIMD}} \times \langle \text{DFT}(\hat{x}_0) | \cdots | \text{DFT}(\hat{x}_{M-1}) \rangle_{\text{SIMD}}
$$

where $|$ denotes concatenation of polynomial coefficients.

**Connection between coefficient and SIMD encoding (Lemma 1):**

$$
\langle \text{DFT}(w) \rangle_{\text{SIMD}} \times \langle \text{DFT}(x) \rangle_{\text{SIMD}} = \text{DFT}(\langle w \rangle_{\text{Coeff}} \times \langle x \rangle_{\text{Coeff}})
$$

This bridges the two HE encoding methods: polynomial multiplication in coefficient domain equals element-wise multiplication in SIMD (frequency) domain.

**Loss-aware sensitivity for block size assignment:**

For layer $i$ with weight perturbation $\Delta W_i = W_i' - W_i$ (from circulant transformation):

$$
\Omega_i = \Delta W_i^\top H \Delta W_i \approx \Delta W_i^\top \text{diag}\left(\left(\frac{\partial \mathcal{L}(\mathcal{D})}{\partial W_i}\right)^2\right) \Delta W_i
$$

where $H$ is approximated by the diagonal Fisher information matrix. The loss-aware initialization solves:

$$
W_i' = \mathbb{E}\left[W_i \odot \left(\frac{\partial \mathcal{L}(\mathcal{D})}{\partial W_i}\right)^2\right]_{\text{diag}}
$$

**Latency-aware block size assignment (ILP):**

$$
\min_{\{b_i\}_{i=1}^{L}} \sum_{i=1}^{L} \Omega_i^{b_i}, \quad \text{subject to} \quad \sum_{i=1}^{L} \text{LAT}_i^{b_i} \leq \text{Latency Limit}
$$

where $b_i \in \{2^0, 2^1, \ldots, 2^{k-1}\}$ and $\text{LAT}_i^{b_i}$ is pre-computed per layer.

**Key Definitions:**

- $n$ — polynomial degree in BFV HE scheme (e.g., $n = 8192$)
- $b$ — block size of circulant transformation (power of 2)
- $\mathbb{A}_n = \mathbb{Z}[X]/(X^n - 1)$ — ring of integer polynomials mod $X^n - 1$
- $(d_1, d_2, d_3)$ — input, hidden, output dimensions of a GEMM
- HE-Pmult — ciphertext-plaintext multiplication
- HE-Rot — ciphertext rotation (order of magnitude slower than HE-Pmult)
- SIMD encoding — single instruction multiple data encoding for HE
- Coefficient encoding — polynomial coefficient-based encoding for HE

## Complexity

| Operation | Bolt+BSGS (SOTA) | CirEncode (b=8) |
|-----------|-------------------|-----------------|
| HE-Pmult (GEMM) | $O(d_1 d_2 d_3 / n)$ | $O(d_1 d_2 d_3 / (nb))$ |
| HE-Rot (GEMM) | $O(\sqrt{d_3 d_2 d_3} / n)$ | $O(\sqrt{b} d_2 d_3 / n)$ |
| HE-Pmult (Conv) | $O(HWC K / n)$ | $O(HWC K / (nb))$ |
| HE-Rot (Conv) | $O(\sqrt{HWCK R^2} / n)$ | $O(\sqrt{HWCR^2 / (nb)})$ |

**Reduction factors (CirEncode vs Bolt+BSGS):**
- HE-Pmult: reduced by factor of $b$
- HE-Rot: reduced by factor of $\sqrt{b}$

**End-to-end latency (Tiny ImageNet):**

| Model | Bolt Latency | PrivCirNet Latency | Speedup |
|-------|-------------|-------------------|---------|
| MobileNetV2 | ~45s | ~24s | 1.9x |
| ResNet-18 | ~100s | ~20s | 5.0x |
| ViT | ~170s | ~130s | 1.3x |

**Memory:** Block circulant weights require $O(d_2 d_3 / b)$ parameters vs $O(d_2 d_3)$ for dense.

## Applicability

- **Private inference (HE/MPC)**: Primary application — enables encrypted neural network inference with dramatically reduced latency by making linear layers circulant-compatible with HE encoding
- **ConvNets (MobileNetV2, ResNet-18)**: Achieves 1.7-5.0x latency reduction with iso-accuracy over SOTA HE frameworks
- **Vision Transformers (ViT)**: 1.3x latency reduction; linear layers in attention and FFN are natural targets for block circulant transformation
- **Federated learning**: Block circulant weights reduce both computation and communication overhead in privacy-preserving distributed settings
- **Edge deployment**: The block circulant structure simultaneously reduces model size ($b$x compression) and computation cost ($O(n \log b)$ vs $O(n^2)$), complementing the HE efficiency gains
- **Any DNN with large linear layers**: The framework generalizes to any architecture where GEMMs dominate latency

## Limitations

- **HE-specific encoding**: CirEncode is designed for BFV lattice-based HE; the polynomial ring structure $\mathbb{A}_n$ and the coefficient/SIMD duality are specific to this cryptographic setting
- **Accuracy degradation at high compression**: Block size $b=16$ can cause significant accuracy loss (e.g., 4-5% on Tiny ImageNet), requiring careful per-layer block size tuning
- **Block size must be power of 2**: Constrained to $b \in \{1, 2, 4, 8, 16, \ldots\}$ by the NTT structure underlying BFV
- **Nonlinear layers not addressed**: PrivCirNet only optimizes linear layers; nonlinear activations (ReLU, GELU) still require costly MPC communication rounds
- **ConvBN fusion requires special handling**: Naive batch normalization fusion destroys circulant structure; a custom circulant-aware fusion is needed (averaging BN parameters within circulant groups)
- **Search cost**: The ILP-based block size assignment requires pre-computing sensitivity $\Omega_i^{b_i}$ for all layers and block sizes, though this is a one-time cost

## Implementation Notes

```python
import torch
import torch.fft as fft

class CirEncodeGEMM:
    """Simulates CirEncode for block circulant GEMM in plaintext.

    In actual HE deployment, the polynomial multiplication
    w_hat * x_hat is performed over the encrypted ring A_n.
    This plaintext version demonstrates the encoding structure.
    """

    @staticmethod
    def encode_weight(W_circ_row, d1, n):
        """Encode circulant weight vector into polynomial.

        W_circ_row: (b,) defining vector of b x b circulant block
        d1: input batch dimension
        n: polynomial degree
        """
        b = W_circ_row.shape[0]
        w_hat = torch.zeros(n)
        for i in range(b):
            w_hat[i * d1] = W_circ_row[i]
        return w_hat

    @staticmethod
    def encode_input(X_block, n):
        """Encode b x d1 input block into polynomial.

        X_block: (b, d1) input matrix for one circulant block
        """
        b, d1 = X_block.shape
        x_hat = torch.zeros(n)
        for i in range(b):
            for j in range(d1):
                x_hat[i * d1 + j] = X_block[i, j]
        return x_hat

    @staticmethod
    def poly_mult_mod(a, b, n):
        """Polynomial multiplication mod x^n - 1 (circular convolution)."""
        # This is the operation that maps to a single HE-Pmult
        result = torch.zeros(n)
        for i in range(n):
            for j in range(n):
                result[(i + j) % n] += a[i] * b[j]
        return result

    @staticmethod
    def decode_output(y_hat, b, d1):
        """Decode polynomial back to b x d1 output matrix."""
        Y = torch.zeros(b, d1)
        for i in range(b):
            for j in range(d1):
                Y[i, j] = y_hat[i * d1 + j]
        return Y


# For practical block-circulant layers (plaintext fast path):
class BlockCirculantLinearWithSensitivity(torch.nn.Module):
    """Block circulant linear layer with loss-aware initialization."""

    def __init__(self, in_features, out_features, block_size,
                 pretrained_weight=None, grad_sq=None):
        super().__init__()
        self.b = block_size
        self.q = in_features // block_size
        self.p = out_features // block_size

        if pretrained_weight is not None and grad_sq is not None:
            # Loss-aware initialization: W' = E[W * (dL/dW)^2]_diag
            # Extract circulant vectors weighted by gradient sensitivity
            w_init = self._loss_aware_init(pretrained_weight, grad_sq)
        else:
            w_init = torch.randn(self.p, self.q, self.b) / (in_features ** 0.5)

        self.w = torch.nn.Parameter(w_init)

    def _loss_aware_init(self, W, grad_sq):
        """Initialize circulant vectors using diagonal Fisher weighting."""
        b = self.b
        w_vecs = torch.zeros(self.p, self.q, b)
        for i in range(self.p):
            for j in range(self.q):
                block = W[i*b:(i+1)*b, j*b:(j+1)*b]
                g_block = grad_sq[i*b:(i+1)*b, j*b:(j+1)*b]
                # Weight each diagonal by gradient magnitude
                weighted = block * g_block
                for k in range(b):
                    diag_vals = torch.diagonal(weighted, offset=-k)
                    if k > 0:
                        diag_vals2 = torch.diagonal(weighted, offset=b-k)
                        w_vecs[i, j, k] = torch.cat([diag_vals, diag_vals2]).mean()
                    else:
                        w_vecs[i, j, k] = diag_vals.mean()
        return w_vecs

    def forward(self, x):
        batch = x.shape[0]
        x_blocks = x.view(batch, self.q, self.b)
        x_fft = fft.fft(x_blocks, dim=-1)
        w_fft = fft.fft(self.w, dim=-1)
        out_fft = (x_fft.unsqueeze(1) * w_fft.unsqueeze(0)).sum(dim=2)
        out = fft.ifft(out_fft, dim=-1).real
        return out.reshape(batch, -1)
```

## References

- Xu, T., Wu, L., Wang, R., Li, M. "PrivCirNet: Efficient Private Inference via Block Circulant Transformation" (NeurIPS 2024). arXiv:2405.14569
- Pang, Q., et al. "Bolt: Privacy-preserving, accurate and efficient inference for transformers" (IEEE S&P 2024)
- Ding, C., et al. "CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices" (MICRO 2017)
- Huang, Z., et al. "Cheetah: Lean and fast secure two-party deep neural network inference" (USENIX Security 2022)
- Ju, J.H., et al. "Neujeans: Private neural network inference with joint optimization of convolution and bootstrapping" (ACM CCS 2024)
- Lou, Q., Lu, W., Hong, C., Jiang, L. "Falcon: Fast spectral inference on encrypted data" (NeurIPS 2020)
