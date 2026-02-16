# 085: Orthogonal Group Permutation Relaxation (OT4P)

**Category**: approximation
**Gain type**: flexibility
**Source**: Guo et al. (NeurIPS 2024)
**Paper**: [papers/ot4p-orthogonal-permutation-relaxation.pdf]
**Documented**: 2026-02-15

## Description

OT4P (Orthogonal Group-based Transformation for Permutation Relaxation) provides a temperature-controlled differentiable mapping from unconstrained vectors in $\mathbb{R}^{n(n-1)/2}$ to a manifold $\mathcal{M}_P$ that tightly wraps around the permutation matrices inside the special orthogonal group $\text{SO}(n)$. As the temperature $\tau \to 0$, points on $\mathcal{M}_P$ converge to the nearest permutation matrix. This enables gradient-based optimization over latent permutations without the local-minima issues of Birkhoff polytope (Sinkhorn) relaxations.

The key insight is that permutation matrices live inside the orthogonal group $O(n)$, and the orthogonal group offers a **lower-dimensional** and **geometrically richer** relaxation than the Birkhoff polytope: $n(n-1)/2$ parameters vs $(n-1)^2$, with the additional benefit of preserving inner products. OT4P operates in two steps: (I) map a free vector to $\text{SO}(n)$ via the Lie exponential, (II) move the resulting orthogonal matrix along a geodesic toward the closest permutation matrix, controlled by temperature.

This trick is directly relevant to column-sparse transition matrices (PD-SSM), where the permutation component $P$ must be learned differentiably. OT4P provides an alternative to Gumbel-Softmax and Sinkhorn for learning the column one-hot selection matrix, with better optimization landscape properties.

## Mathematical Form

**Step I — Map to $\text{SO}(n)$:**

Given an unconstrained vector $a \in \mathbb{R}^{n(n-1)/2}$, fill it into a strictly upper-triangular matrix $A$, then map to $\text{SO}(n)$ via:

$$
\phi: \mathbb{R}^{n(n-1)/2} \to \text{SO}(n), \quad A \mapsto \text{expm}(A - A^\top)
$$

where $A - A^\top$ is skew-symmetric (lies in the Lie algebra $\mathfrak{so}(n)$), and $\text{expm}(\cdot)$ is the matrix exponential.

**Step II — Move toward nearest permutation:**

Find the closest permutation matrix via the linear assignment problem:

$$
\rho(O) := \arg\max_{P \in \mathcal{P}_n} \langle P, O \rangle_F = \arg\max_{P \in \mathcal{P}_n} \text{trace}(P^\top O)
$$

This is solved in $O(n^3)$ by the Hungarian algorithm. Then move $O$ toward $P$ along the geodesic on $\text{SO}(n)$:

$$
\widetilde{O} = P \, \text{expm}(\tau \, \text{logm}(P^\top O)) = P(P^\top O)^\tau
$$

where $\tau \in (0, 1]$ is the temperature parameter.

**Combined transformation:**

$$
\psi_\tau: \text{SO}(n) \to \mathcal{M}_P, \quad O \mapsto \rho(O) D \left( |\rho(O) D|^\top O \right)^\tau D^\top
$$

where $D = I$ for even permutations and $D = \text{diag}(1, \ldots, 1, -1)$ for odd permutations (to handle the $\det = -1$ case).

**Key Properties:**

- $\lim_{\tau \to 0^+} \|\widetilde{O} - P\|_F = 0$ — converges to exact permutation
- The composite mapping $\psi_\tau \circ \phi$ is differentiable, surjective, and injective on each submanifold $\mathcal{S}_P$
- The parameterization does not alter the original problem (surjectivity) and does not introduce spurious local minima (injectivity)

**Re-parameterized gradient estimator (for stochastic optimization):**

$$
P \sim q(P; \theta) \iff P = \rho(\phi(A + B\epsilon)), \quad \epsilon \sim q(\epsilon)
$$

$$
\nabla \mathbb{E}_{\epsilon \sim q(\epsilon)} f(\psi_\tau(\phi(A + B\epsilon))) = \mathbb{E}_{\epsilon \sim q(\epsilon)} \nabla f(\psi_\tau(\phi(A + B\epsilon)))
$$

This enables Monte Carlo gradient estimation for distributions over latent permutations.

## Complexity

| Operation | Sinkhorn (Birkhoff) | OT4P (Orthogonal) |
|-----------|--------------------|--------------------|
| Parameter dimension | $(n-1)^2$ | $n(n-1)/2$ |
| Forward pass | $O(Ln^2)$ ($L$ Sinkhorn iters) | $O(n^3)$ (matrix exp + Hungarian) |
| Backward pass | $O(Ln^2)$ | $O(n^3)$ (eigendecomposition) |
| Local minima | Unreliable (convex hull) | Robust (manifold) |
| Output structure | Doubly stochastic | Orthogonal (near-permutation) |

**Memory:** $O(n^2)$ for the orthogonal matrix, but only $O(n(n-1)/2)$ free parameters.

**Efficient forward/backward:** The matrix power $O^\tau$ is computed via eigendecomposition $O = Q \Lambda Q^{-1}$, then $O^\tau = Q \Lambda^\tau Q^{-1}$ (raising $n$ scalars to power $\tau$). The backward pass through $\psi_\tau$ reduces to a single linear transformation $\widetilde{O} = W_\tau O$, making it efficient.

## Applicability

- **Column-sparse transition matrices (PD-SSM):** Learning the permutation component $P$ in $A = P \cdot D$ can use OT4P instead of Gumbel-Softmax, potentially with better optimization landscape
- **Permutation synchronization:** Finding consistent permutations across multiple objects (graph matching, multi-view correspondence)
- **Latent permutation inference:** Variational inference over permutation-valued latent variables (e.g., neuron identity matching in neuroscience)
- **Mode connectivity:** Finding permutation matrices that align neural network weight spaces for model merging
- **Structured sparsity (PA-DST):** Learning the channel permutation for structured sparse training, as an alternative to the Lipschitz penalty approach

## Limitations

- **$O(n^3)$ cost:** The Hungarian algorithm and matrix exponential/eigendecomposition are $O(n^3)$, which limits scalability to very large permutations (though for SSM state dimensions $N \sim 16$–$256$ this is manageable)
- **Temperature sensitivity:** At $\tau = 0.7$ (too close to 1), convergence to exact permutations can be suboptimal; $\tau \in [0.3, 0.5]$ works best empirically
- **Odd permutation handling:** Requires a sign-flip matrix $D$ to map between $\text{SO}(n)$ (even) and odd permutations, adding a case split
- **Discontinuity at boundaries:** The mapping $\psi_\tau$ is discontinuous at points equidistant from multiple permutations (where $\rho(\cdot)$ switches), though this has zero Lebesgue measure
- **Not directly compatible with straight-through estimator:** Unlike Gumbel-Softmax which naturally pairs with STE, OT4P uses its own re-parameterization gradient approach

## Implementation Notes

```python
import torch
from scipy.optimize import linear_sum_assignment

def ot4p(A, tau=0.5, B=None):
    """
    OT4P: Map unconstrained matrix A to near-permutation matrix.

    Args:
        A: (n, n) unconstrained real matrix
        tau: temperature in (0, 1]
        B: (n, n) optional shift matrix in SO(n)

    Returns:
        O_tilde: (n, n) orthogonal matrix near a permutation
    """
    n = A.shape[0]

    # Step I: Map to SO(n) via Lie exponential
    skew = A - A.T  # Skew-symmetric (Lie algebra so(n))
    O = torch.matrix_exp(skew)  # Orthogonal matrix in SO(n)

    # Optional shift to handle boundary issues
    if B is not None:
        O = B @ O

    # Step II: Find closest permutation (Hungarian algorithm)
    # Solve linear assignment: max trace(P^T O)
    cost = -O.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    P = torch.zeros(n, n, device=A.device)
    P[row_ind, col_ind] = 1.0

    # Handle odd permutations (det(P) = -1)
    if torch.det(P) < 0:
        D = torch.eye(n, device=A.device)
        D[-1, -1] = -1
        P_hat = P @ D
    else:
        D = torch.eye(n, device=A.device)
        P_hat = P

    # Move O toward P along geodesic, controlled by tau
    # O_tilde = P_hat @ (P_hat^T @ O)^tau
    PtO = P_hat.T @ O
    # Matrix power via eigendecomposition
    eigenvalues, Q = torch.linalg.eig(PtO)
    PtO_tau = Q @ torch.diag(eigenvalues ** tau) @ torch.linalg.inv(Q)
    O_tilde = P_hat @ PtO_tau.real

    # Undo odd permutation handling
    O_tilde = O_tilde @ D.T

    return O_tilde
```

## References

- Guo, Y., Zhu, C., Zhu, H., and Wu, T. (2024). OT4P: Unlocking Effective Orthogonal Group Path for Permutation Relaxation. NeurIPS 2024.
- Mena, G. et al. (2018). Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018. (Birkhoff polytope baseline)
- Lezcano-Casado, M. and Martínez-Rubio, D. (2019). Cheap Orthogonal Constraints in Neural Networks. ICML 2019. (Matrix exponential for orthogonal parameterization)
- Kuhn, H. (1955). The Hungarian Method for the Assignment Problem. (Linear assignment solver used in Step II)
