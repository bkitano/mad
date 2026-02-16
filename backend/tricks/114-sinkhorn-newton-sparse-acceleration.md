# 114: Sinkhorn-Newton-Sparse (SNS) Acceleration

**Category**: approximation
**Gain type**: efficiency
**Source**: Tang, Shavlovsky, Rahmanian, Tardini, Thekumparampil, Xiao & Ying (Stanford & Amazon, ICLR 2024)
**Paper**: [papers/sinkhorn-newton-sparse.pdf]
**Documented**: 2026-02-15

## Description

The standard Sinkhorn algorithm for entropy-regularized optimal transport converges at a linear rate that degrades as the regularization parameter $\varepsilon \to 0$. In practice, this means hundreds to thousands of iterations may be required for high-precision solutions — a significant bottleneck when Sinkhorn iterations are used inside training loops (e.g., blockwise Sinkhorn channel permutation, differentiable permutation learning, OT-based losses).

Sinkhorn-Newton-Sparse (SNS) is a two-stage algorithm that **replaces the tail end of Sinkhorn iterations with Newton steps**, achieving empirically super-exponential convergence while maintaining $O(n^2)$ per-iteration cost — the same as standard Sinkhorn.

The key insight is a **variational viewpoint**: the Sinkhorn algorithm maximizes a concave Lyapunov potential $f(x, y)$ via alternating coordinate ascent. Newton's method on this potential converges super-exponentially but requires solving a $2n \times 2n$ linear system at $O(n^3)$ per step. SNS observes that after a warm-up phase of $N_1$ Sinkhorn iterations, the **Hessian matrix $\nabla^2 f$ is approximately sparse** — most off-diagonal entries decay exponentially. A simple thresholding sparsification reduces the Hessian to $O(n)$ nonzeros per row (target sparsity $\lambda = O(1/n)$), enabling conjugate gradient solves at $O(\lambda n^3) = O(n^2)$ per Newton step.

The algorithm converges **orders of magnitude faster** than Sinkhorn in total iteration count (e.g., 29 iterations vs. 56,199 on a random $500 \times 500$ assignment problem with $\eta = 1200$), while the wall-clock time improvement is similarly dramatic (0.34s vs. 233s).

This directly accelerates the Sinkhorn normalization step in blockwise channel permutation: instead of running 5-20 Sinkhorn iterations per block to produce a doubly stochastic matrix, a few Sinkhorn steps followed by 1-2 Newton steps can achieve much higher precision with fewer total iterations.

## Mathematical Form

**Entropic OT Problem:**

$$
\min_{P: P\mathbf{1}=r, P^\top\mathbf{1}=c} C \cdot P + \frac{1}{\eta} H(P)
$$

where $C \in \mathbb{R}^{n \times n}$ is the cost matrix, $r, c \in \mathbb{R}^n$ are marginals, $\eta > 0$ is the entropy regularization parameter, and $H(P) := \sum_{ij} p_{ij} \log p_{ij}$.

**Lyapunov Potential (Dual):**

$$
f(x, y) := -\frac{1}{\eta} \sum_{ij} \exp\left(\eta(-c_{ij} + x_i + y_j) - 1\right) + \sum_i r_i x_i + \sum_j c_j y_j
$$

The Sinkhorn algorithm is equivalent to alternating coordinate ascent on $f$:

$$
x \leftarrow \arg\max_x f(x, y), \quad y \leftarrow \arg\max_y f(x, y)
$$

**First and Second Derivatives:**

$$
\nabla_x f(x, y) = r - P\mathbf{1}, \quad \nabla_y f(x, y) = c - P^\top\mathbf{1}
$$

$$
\nabla^2 f(x, y) = \eta \begin{bmatrix} \text{diag}(P\mathbf{1}) & P \\ P^\top & \text{diag}(P^\top\mathbf{1}) \end{bmatrix}
$$

where $P = \exp\left(\eta(-C + x\mathbf{1}^\top + \mathbf{1}y^\top) - 1\right)$.

**Approximate Sparsity of the Hessian (Theorem 1):**

After $t$ Sinkhorn steps with $\eta > \frac{1 + 2\log n}{\Delta}$ (where $\Delta$ is the vertex optimality gap), the transport matrix $P_{t,\eta}$ is $(\lambda^*, \epsilon_{t,\eta})$-sparse with:

$$
\epsilon_{t,\eta} := 6n^2 \exp(-\eta\Delta) + \frac{\sqrt{q}}{\sqrt{t}}
$$

where $\lambda^* = \tau(\mathcal{F})$ is the sparsity of the optimal face, and for uniform marginals $\lambda^* = 1/n$.

**SNS Algorithm (two stages):**

*Stage 1 — Sinkhorn warm-up ($N_1$ iterations):*
$$
P \leftarrow \exp\left(\eta(-C + x\mathbf{1}^\top + \mathbf{1}y^\top) - 1\right)
$$
$$
x \leftarrow x + (\log(r) - \log(P\mathbf{1})) / \eta
$$
$$
y \leftarrow y + (\log(c) - \log(P^\top\mathbf{1})) / \eta
$$

*Stage 2 — Sparse Newton ($N_2$ iterations):*

1. Sparsify: $M \leftarrow \text{Sparsify}(\nabla^2 f(z), \rho)$ — threshold entries below $\rho$
2. Solve: $\Delta z \leftarrow \text{ConjugateGradient}(M, -\nabla f(z))$ — sparse CG solve
3. Line search: $\alpha \leftarrow \text{LineSearch}(f, z, \Delta z)$
4. Update: $z \leftarrow z + \alpha \Delta z$

**Sparsification Rule:**

Set target sparsity $\lambda = 2/n$. Pick $\rho$ as the $\lceil \lambda n^2 \rceil$-largest entry of $\nabla^2 f(z)$, ensuring $\tau(M) \leq \lambda$.

## Complexity

| Operation | Sinkhorn | Newton (dense) | SNS |
|-----------|----------|---------------|-----|
| Per iteration | $O(n^2)$ | $O(n^3)$ | $O(n^2)$ |
| Convergence rate | Linear: $(1 - e^{-24\|C\|\eta})^t$ | Super-exponential | Super-exponential (empirical) |
| Iterations to machine precision | $O(\text{poly}(\varepsilon^{-1}))$ | Few ($< 10$) | $N_1 + N_2$ (typically $20 + 10$) |

**Concrete benchmarks ($n = 500$, $\eta = 1200$, machine precision):**

| Case | Method | Time (s) | Iterations |
|------|--------|----------|------------|
| Random LAP | SNS | **0.34** | **29** (20 Sinkhorn + 9 Newton) |
| Random LAP | Sinkhorn | 233.36 | 56,199 |
| MNIST $L_2$ | SNS | **2.33** | **53** (20 Sinkhorn + 33 Newton) |
| MNIST $L_2$ | Sinkhorn | 18.84 | 2,041 |
| MNIST $L_1$ | SNS | **12.22** | **777** (700 Sinkhorn + 77 Newton) |
| MNIST $L_1$ | Sinkhorn | 45.75 | 5,748 |

**Memory:** $O(n^2)$ for the transport matrix $P$ and the sparsified Hessian $M$. The sparse Hessian stores $O(\lambda n^2) = O(n)$ nonzeros per row when $\lambda = O(1/n)$.

## Applicability

- **Blockwise Sinkhorn channel permutation:** Each block's $B \times B$ Sinkhorn normalization (converting learnable $\mathbf{W}_P^i$ to a doubly stochastic matrix) can use SNS to achieve higher-precision doubly stochastic matrices in fewer iterations. With block size $B = 64$, the Hessian is $128 \times 128$ — small enough that even a few CG iterations on the sparse Hessian are cheap. This improves the quality of the soft permutation before Hungarian hardening.
- **Sinkhorn permutation relaxation:** Any Gumbel-Sinkhorn or temperature-annealed Sinkhorn loop benefits from SNS, especially at low temperatures ($\tau \to 0$, equivalently high $\eta$) where standard Sinkhorn convergence degrades severely.
- **Optimal transport losses in training:** Wasserstein distance, Sinkhorn divergence, and OTDD dataset comparison losses computed during training benefit from SNS's faster convergence, reducing the per-batch OT computation cost.
- **FlashSinkhorn composition:** SNS and FlashSinkhorn are complementary — FlashSinkhorn makes each Sinkhorn iteration IO-efficient, while SNS reduces the total number of iterations needed. The Sinkhorn warm-up stage of SNS can use FlashSinkhorn's tiled kernels, and the Newton stage requires only sparse matrix operations.
- **High-precision permutation learning:** When the Sinkhorn output must closely approximate a true permutation (low temperature), standard Sinkhorn requires many iterations. SNS achieves machine-precision doubly stochastic matrices in an order of magnitude fewer iterations.

## Limitations

- The Hessian sparsity guarantee requires sufficient Sinkhorn warm-up ($N_1 = O(1/\epsilon^2)$ steps to be within $\epsilon$ of convergence) — the Newton stage cannot start from a cold initialization
- The theoretical super-exponential convergence is empirically observed but not proven — the formal guarantee is for approximate sparsity, not convergence rate
- For very small problems ($n < 50$), the overhead of Hessian assembly, sparsification, and CG solve may exceed the cost of simply running more Sinkhorn iterations
- The CG solver in the Newton stage is sequential (not trivially parallelizable like Sinkhorn row/column scaling), which may limit GPU utilization for the Newton phase
- Sparsity threshold $\rho$ is determined by a global ranking of Hessian entries — this requires an $O(n^2)$ selection step per Newton iteration
- Currently validated on CPU (Apple M1 Pro); GPU-optimized sparse Newton implementations are an open engineering challenge
- Non-unique OT solutions (e.g., $L_1$ cost) require larger $N_1$ warm-up and higher target sparsity $\lambda = 15/n$

## Implementation Notes

```python
import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg as sparse_cg

def sinkhorn_newton_sparse(C, r, c, eta, N1=20, N2=10, lam=None):
    """
    Sinkhorn-Newton-Sparse (SNS) algorithm for entropic OT.

    Args:
        C: (n, n) cost matrix
        r: (n,) source marginal
        c: (n,) target marginal
        eta: entropy regularization parameter (larger = less regularization)
        N1: number of Sinkhorn warm-up iterations
        N2: number of Newton iterations
        lam: target sparsity (default 2/n)

    Returns:
        P: (n, n) optimal transport plan
    """
    n = C.shape[0]
    if lam is None:
        lam = 2.0 / n

    x = torch.zeros(n)
    y = torch.zeros(n)

    # Stage 1: Sinkhorn warm-up
    for _ in range(N1):
        P = torch.exp(eta * (-C + x.unsqueeze(1) + y.unsqueeze(0)) - 1)
        x = x + (torch.log(r) - torch.log(P.sum(dim=1))) / eta
        P = torch.exp(eta * (-C + x.unsqueeze(1) + y.unsqueeze(0)) - 1)
        y = y + (torch.log(c) - torch.log(P.sum(dim=0))) / eta

    # Stage 2: Sparse Newton iterations
    z = torch.cat([x, y])

    for _ in range(N2):
        P = torch.exp(eta * (-C + z[:n].unsqueeze(1) + z[n:].unsqueeze(0)) - 1)

        # Gradient
        grad_x = r - P.sum(dim=1)
        grad_y = c - P.sum(dim=0)
        grad = -torch.cat([grad_x, grad_y])

        # Hessian (2n x 2n block matrix)
        # H = eta * [[diag(P@1), P], [P^T, diag(P^T@1)]]
        H = torch.zeros(2*n, 2*n)
        H[:n, :n] = eta * torch.diag(P.sum(dim=1))
        H[:n, n:] = eta * P
        H[n:, :n] = eta * P.T
        H[n:, n:] = eta * torch.diag(P.sum(dim=0))

        # Sparsify: keep only top ceil(lam * (2n)^2) entries
        k = int(np.ceil(lam * (2*n)**2))
        threshold = torch.topk(H.abs().flatten(), k).values[-1]
        H_sparse = H.clone()
        H_sparse[H.abs() < threshold] = 0.0

        # Solve sparse system via CG
        H_sp = csr_matrix(H_sparse.numpy())
        dz, _ = sparse_cg(H_sp, grad.numpy(), maxiter=50)
        dz = torch.tensor(dz, dtype=torch.float64)

        # Line search (simple backtracking)
        alpha = 1.0
        f_curr = lyapunov(z, C, r, c, eta)
        for _ in range(20):
            z_new = z + alpha * dz
            if lyapunov(z_new, C, r, c, eta) > f_curr:
                break
            alpha *= 0.5
        z = z + alpha * dz

    P = torch.exp(eta * (-C + z[:n].unsqueeze(1) + z[n:].unsqueeze(0)) - 1)
    return P


def lyapunov(z, C, r, c, eta):
    """Lyapunov potential f(x, y) for the entropic OT dual."""
    n = len(r)
    x, y = z[:n], z[n:]
    P = torch.exp(eta * (-C + x.unsqueeze(1) + y.unsqueeze(0)) - 1)
    return -(1.0/eta) * P.sum() + (r * x).sum() + (c * y).sum()


# For blockwise Sinkhorn channel permutation:
# Instead of running 5-20 Sinkhorn iterations per block,
# use SNS with N1=3 Sinkhorn + N2=2 Newton steps for
# higher-precision doubly stochastic matrices at similar cost.
#
# For B=64 blocks:
#   Hessian is 128x128, sparse with ~2/64 = 3% nonzeros
#   CG solve: ~50 iterations of sparse mat-vec at O(128 * 0.03 * 128^2) = O(31K)
#   vs. 20 Sinkhorn iterations at O(20 * 64^2) = O(82K)
```

## References

- Tang, X., Shavlovsky, M., Rahmanian, H., Tardini, E., Thekumparampil, K. K., Xiao, T. & Ying, L. (2024). Accelerating Sinkhorn Algorithm with Sparse Newton Iterations. ICLR 2024. arXiv:2401.12253.
- Brauer, C., Clason, C., Lorenz, D. & Wirth, B. (2017). A Sinkhorn-Newton Method for Entropic Optimal Transport. arXiv:1710.06635.
- Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. NeurIPS 2013.
- Altschuler, J., Niles-Weed, J. & Rigollet, P. (2017). Near-Linear Time Approximation Algorithms for Optimal Transport via Sinkhorn Iteration. NeurIPS 2017.
- Carlier, G. (2022). On the Linear Convergence of the Multimarginal Sinkhorn Algorithm. SIAM J. Optim.
- Knight, P. A. (2008). The Sinkhorn-Knopp Algorithm: Convergence and Applications. SIAM J. Matrix Anal. Appl.
