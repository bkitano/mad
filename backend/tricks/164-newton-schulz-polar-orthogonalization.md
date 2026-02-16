# 164: Newton-Schulz Polar Orthogonalization

**Category**: approximation
**Gain type**: efficiency
**Source**: Bernstein & Newhouse (2024) — "Old Optimizer, New Norm"; Jordan et al. (2024) — Muon optimizer; Kim & Oh (2026) — ICLR 2026 convergence analysis
**Paper**: [papers/newton-schulz-polar-muon.pdf], [papers/muon-newton-schulz-convergence.pdf]
**Documented**: 2026-02-15

## Description

The Newton-Schulz iteration provides an SVD-free, inversion-free method to compute the **polar factor** of a matrix — i.e., given $G = U\Sigma V^\top$, it approximates $UV^\top$ (the closest semi-orthogonal matrix in Frobenius norm) using only matrix multiplications. This is the core computational primitive of the **Muon optimizer**, which orthogonalizes the momentum matrix before applying gradient updates. The key insight from Bernstein & Newhouse (2024) is that Shampoo without accumulation is equivalent to **steepest descent under the spectral norm**, and the update direction is the polar factor $UV^\top$ of the gradient. Newton-Schulz replaces the expensive SVD ($O(d^3)$ in fp64) with ~5 iterations of a degree-$\kappa$ polynomial, each requiring only 2-3 matrix multiplications that run entirely on tensor cores in **bfloat16**.

The Newton-Schulz polynomial $p_\kappa(\lambda)$ is the Taylor truncation of $\lambda^{-1/2}$ at $\lambda = 1$. When applied recursively as $X \leftarrow p_\kappa(XX^\top) X$, each step drives the singular values of $X$ toward 1 (orthogonality), with the orthogonality residual $\delta$ decaying as $\delta \mapsto \delta^{\kappa+1}$ per step. For the quintic polynomial ($\kappa = 2$), this means $\delta \mapsto \delta^3$ per step — so 5 steps achieve $\delta_0^{3^5} = \delta_0^{243}$ convergence, which is practically exact for any $\delta_0 < 1$. Kim & Oh (ICLR 2026) prove that Muon with Newton-Schulz converges to a stationary point at the same rate as the idealized SVD-polar variant, up to a multiplicative factor $\chi_q$ that converges to 1 doubly exponentially in the number of Newton-Schulz steps $q$.

## Mathematical Form

**Core Operation: Newton-Schulz Polar Approximation**

Given a matrix $G \in \mathbb{R}^{m \times n}$ (with $m \geq n$), compute $\text{Polar}(G) = UV^\top$ where $G = U\Sigma V^\top$:

$$
X_0 = \frac{G}{\|G\|_F}, \qquad X_{j+1} = p_\kappa(X_j X_j^\top) \, X_j, \quad j = 1, \ldots, q
$$

After $q$ steps, $X_q \approx UV^\top$.

**Newton-Schulz Polynomial (Definition 2, Kim & Oh 2026):**

$$
p_\kappa(\lambda) = \sum_{s=0}^{\kappa} c_s (1 - \lambda)^s, \qquad c_s = \frac{(2s)!}{4^s (s!)^2} > 0
$$

This is the degree-$\kappa$ Taylor truncation of $\lambda^{-1/2}$ at $\lambda = 1$.

**Explicit forms:**

- $\kappa = 1$ (cubic): $p_1(\lambda) = \frac{1}{2}(3 - \lambda)$, giving $X \leftarrow \frac{1}{2}X(3I - X^\top X)$
- $\kappa = 2$ (quintic): $p_2(\lambda) = \frac{1}{8}(15 - 10\lambda + 3\lambda^2)$, giving $X \leftarrow \frac{1}{8}X(15I - 10X^\top X + 3(X^\top X)^2)$

**Muon's Practical Quintic (Jordan et al. 2024):**

In practice, Muon uses optimized coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$:

$$
X \leftarrow a X + b (X X^\top) X + c (X X^\top)^2 X
$$

This is algebraically equivalent to a degree-2 Newton-Schulz polynomial but with coefficients chosen to maximize the slope at $\sigma = 0$ (inflating small singular values as fast as possible), rather than minimizing worst-case error.

**Each iteration requires exactly 2 symmetric products + 1 rectangular product:**

$$
A = X X^\top, \quad B = bA + cA^2, \quad X \leftarrow aX + BX
$$

**Muon Algorithm (Algorithm 1, Kim & Oh 2026):**

1. Compute mini-batch gradient $G_t$
2. Update momentum: $M_t \leftarrow \beta M_{t-1} + G_t$
3. Pre-scale: $X_{t,0} \leftarrow M_t / \max(1, \|M_t\|_F)$
4. For $j = 1, \ldots, q$: $X_{t,j} \leftarrow p_\kappa(X_{t,j-1} X_{t,j-1}^\top) X_{t,j-1}$
5. Update weights: $W_t \leftarrow W_{t-1} - \eta X_{t,q}$

**Orthogonality Residual Decay (Theorem 2, Kim & Oh 2026):**

Define the orthogonality residual $\delta_{t,j} := \|I - X_{t,j} X_{t,j}^\top\|_{\text{op}}$. Then:

$$
\delta_{t,j+1} \leq \delta_{t,j}^{\kappa+1}, \qquad \delta_{t,q} \leq \delta_{t,0}^{(\kappa+1)^q}
$$

The polar approximation error $\varepsilon_q := \|X_{t,q} - \text{Polar}(M_t)\|_{\text{op}}$ satisfies:

$$
\varepsilon_q \leq 1 - \sqrt{1 - \delta_0^{(\kappa+1)^q}} \leq \delta_0^{(\kappa+1)^q}
$$

The multiplicative overhead factor $\chi_q$ in the convergence rate satisfies:

$$
\chi_q = \frac{1}{1 - \varepsilon_q} \leq \frac{1}{\sqrt{1 - \delta_0^{(\kappa+1)^q}}} \to 1 \text{ doubly exponentially}
$$

**Key Definitions:**

- $G \in \mathbb{R}^{m \times n}$ — gradient or momentum matrix
- $\text{Polar}(G) = UV^\top$ — polar factor (nearest semi-orthogonal matrix)
- $p_\kappa$ — Newton-Schulz polynomial of degree $\kappa$
- $q$ — number of Newton-Schulz iteration steps
- $\delta_{t,j}$ — orthogonality residual after $j$ steps at iteration $t$
- $\varepsilon_q$ — polar approximation error (gap from exact SVD-polar)
- $\chi_q$ — multiplicative overhead factor in convergence rate

## Complexity

| Operation | SVD-based | Newton-Schulz ($q$ steps, degree $\kappa$) |
|-----------|-----------|-------------------------------------------|
| Compute polar factor | $O(mn^2)$ in fp64 | $(2\kappa + 1) \cdot q$ matmuls in bf16 |
| Per-step (Muon quintic) | — | 5 matmuls × $q$ steps |
| Total matmuls ($\kappa=2, q=5$) | — | 25 matmuls |
| Precision required | fp32/fp64 | **bfloat16** ✓ |
| Tensor core compatible | No (SVD is iterative+branching) | **Yes** (pure GEMM) ✓ |

**Convergence comparison (Table 1, Kim & Oh 2026):**

| Method | Convergence Rate |
|--------|-----------------|
| SGD with momentum | $O\!\left(\sqrt{\frac{rLD}{T}} + \left(\frac{r^2\sigma^2 LD}{BT}\right)^{1/4}\right)$ |
| Muon with SVD | $O\!\left(\sqrt{\frac{LD}{T}} + \frac{\sigma r}{\sqrt{BT}} + \left(\frac{r\sigma^2 LD}{BT}\right)^{1/4}\right)$ |
| Muon with Newton-Schulz | $\chi_q \cdot O\!\left(\sqrt{\frac{LD}{T}} + \frac{\sigma r}{\sqrt{BT}} + \left(\frac{r\sigma^2 LD}{BT}\right)^{1/4}\right)$ |

where $r = \min(m,n)$ (rank), $L$ = Lipschitz constant, $D$ = initial suboptimality, $\sigma^2$ = variance, $B$ = batch size. Muon removes the $\sqrt{r}$ factor from the deterministic term vs SGD.

**Memory:** $O(mn)$ for $X$ and $A$ — same as storing the momentum matrix.

## Applicability

- **Muon optimizer for pretraining:** Replaces AdamW for matrix-shaped parameters (linear layers, attention projections). Achieves ~35% faster convergence on NanoGPT speedruns vs AdamW. Now in PyTorch core as `torch.optim.Muon` (v2.9+). Works at LLM scale.

- **Spectral descent / Shampoo alternative:** Bernstein & Newhouse show that Shampoo without accumulation computes $UV^\top$ (polar factor of the gradient), which is steepest descent under the spectral norm. Newton-Schulz provides a cheaper path to the same update direction without computing $L_t^{-1/4}$ and $R_t^{-1/4}$.

- **Orthogonal weight reparameterization:** Any architecture using orthogonal constraints on weight matrices (orthogonal RNNs, orthogonal convolutions) can use Newton-Schulz to project weights onto the orthogonal manifold efficiently.

- **Gradient preconditioning:** The SOAP optimizer variant (SOAP-Muon) replaces eigendecomposition-based whitening with Newton-Schulz orthogonalization, gaining tensor-core compatibility.

## Limitations

- **Only applies to matrix-shaped parameters:** Muon is designed for 2D weight matrices. Embedding layers, biases, and 1D parameters still use AdamW. The Bernstein & Newhouse framework assigns different norms to different tensor roles.

- **Spectral norm scaling required:** The momentum must be scaled to have $\|X_0\|_{\text{op}} \leq 1$ before applying Newton-Schulz. The Frobenius norm is used as a cheap upper bound ($\|X\|_{\text{op}} \leq \|X\|_F$), but this can slow convergence when the matrix is far from rank-1.

- **Cost per optimizer step:** 5 iterations × 5 matmuls = 25 matmuls of shape $(m, n) \times (n, m)$ per step, which for large layers (e.g., $m = n = 4096$) adds non-trivial FLOPs. However, this is amortized across the training step and runs in bf16.

- **Coefficient sensitivity:** The Muon quintic coefficients $(3.4445, -4.7750, 2.0315)$ are empirically tuned, not theoretically optimal. The CANS paper (arXiv:2506.10935) shows Chebyshev-optimal polynomials can improve on these.

- **Not a second-order method:** Despite superficial similarity to Shampoo, Muon does not estimate curvature. It is a first-order method (steepest descent under spectral norm) that benefits from the geometry of matrix-valued parameters.

## Implementation Notes

```python
import torch

def newton_schulz_polar(G, q=5, coeffs=(3.4445, -4.7750, 2.0315)):
    """
    Approximate the polar factor of G using Newton-Schulz iteration.

    G: (m, n) matrix, m >= n
    q: number of Newton-Schulz steps
    coeffs: (a, b, c) polynomial coefficients

    Returns: X ≈ Polar(G) = U @ V^T  (m, n) semi-orthogonal
    """
    a, b, c = coeffs

    # Pre-scale to unit Frobenius norm (ensures ||X||_op <= 1)
    X = G / max(G.norm(), 1e-7)  # (m, n)

    for _ in range(q):
        # All operations are tensor-core-friendly matmuls
        A = X @ X.T              # (m, m) symmetric GEMM
        B = b * A + c * (A @ A)  # (m, m) GEMM + scaling
        X = a * X + B @ X        # (m, n) GEMM + scaling

    return X


def muon_step(W, G, M, eta, beta=0.95, q=5):
    """
    One step of the Muon optimizer.

    W: (m, n) weight matrix
    G: (m, n) gradient
    M: (m, n) momentum buffer
    eta: learning rate
    beta: momentum coefficient

    Returns: updated W, M
    """
    # Update momentum
    M = beta * M + G

    # Orthogonalize momentum via Newton-Schulz
    O = newton_schulz_polar(M, q=q)

    # Weight update with spectral-norm-aware scaling
    W = W - eta * (m / n) ** 0.5 * O  # sqrt(fan_out/fan_in) scaling

    return W, M


# Comparison: SVD-based polar (expensive, fp64 required)
def svd_polar(G):
    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
    return U @ Vt  # exact polar factor

# Newton-Schulz: 25 matmuls in bf16
# SVD: O(mn^2) in fp64, not tensor-core friendly
```

**GPU efficiency analysis:**

- **All GEMM, all the time:** Every operation in the Newton-Schulz iteration is a matrix multiplication or elementwise scaling — both map perfectly to tensor cores (WGMMA on H100, MMA on A100).
- **bfloat16 throughout:** Unlike SVD which requires fp32/fp64 for numerical stability, Newton-Schulz is stable in bfloat16 because the polynomial has only positive coefficients and the iteration is self-correcting.
- **Arithmetic intensity:** For $m = n = d$, each iteration costs $O(d^3)$ FLOPs with $O(d^2)$ memory, giving arithmetic intensity $O(d)$ — well above the compute-bound threshold on modern GPUs.
- **No kernel launch overhead:** The entire inner loop can be fused into a single persistent kernel or implemented as a sequence of cuBLAS GEMM calls with minimal launch overhead.
- **In PyTorch core:** `torch.optim.Muon` (v2.9+) provides an optimized implementation.

## References

- Bernstein, J. & Newhouse, L. (2024). Old Optimizer, New Norm: An Anthology. OPT2024 Workshop. arXiv:2409.20325.
- Jordan, K. et al. (2024). Muon: An optimizer for hidden layers in neural networks. GitHub: KellerJordan/Muon.
- Kim, G. Y. & Oh, M.-H. (2026). Convergence of Muon with Newton-Schulz. ICLR 2026. arXiv:2601.19156.
- Gupta, V., Koren, T. & Singer, Y. (2018). Shampoo: Preconditioned Stochastic Tensor Optimization. ICML 2018.
- Vyas, N. et al. (2024). SOAP: Improving and Stabilizing Shampoo using Adam. arXiv:2409.11321.
