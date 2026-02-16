# 092: Perturb-Then-Diagonalize (PTD)

**Category**: stability
**Gain type**: expressivity
**Source**: Rushton, Erichson, Mahoney (ICLR 2024)
**Documented**: 2026-02-10

## Description

The HiPPO matrices that initialize SSMs are non-normal and exponentially ill-conditioned to diagonalize — naive diagonalization produces eigenvectors with entries of magnitude $\sim 2^N$, making direct computation infeasible. The standard workaround (S4D, S5) simply discards the low-rank component from the DPLR structure, but this abandons the theoretical HiPPO guarantees and creates vulnerability to certain frequency perturbations. Perturb-Then-Diagonalize (PTD) solves this by adding a small controlled perturbation $E$ to the HiPPO matrix before diagonalization, yielding a well-conditioned eigenvector matrix while maintaining provably small backward error. This preserves the HiPPO transfer function properties that S4D/S5 lose.

## Mathematical Form

**Core Operation:**

Given HiPPO matrix $A_H$ with NPLR structure $A_H = V\Lambda V^* - PQ^\top$:

$$
\tilde{A}_H = A_H + E = \tilde{V}_H \tilde{\Lambda}_H \tilde{V}_H^{-1}
$$

**Perturbation Optimization:**

The perturbation $E$ is chosen by solving:

$$
\min_E \Phi(E) = \kappa(\tilde{V}_H) + \gamma \|E\|
$$

where:
- $\kappa(\tilde{V}_H)$ — condition number of the eigenvector matrix
- $\gamma > 0$ — controls tradeoff between conditioning and perturbation magnitude
- $\|E\|$ — perturbation norm

**Key Definitions:**

- $A_H \in \mathbb{R}^{n \times n}$ — HiPPO matrix
- $\kappa(V) = \|V\| \cdot \|V^{-1}\|$ — condition number
- $G(s) = C(sI - A)^{-1}B$ — transfer function

**Error Bound (Theorem 3):**

For $\varepsilon = \|E\|_2$:

$$
|G_{\text{PTD}}(s) - G_{\text{DPLR}}(s)| = (2 \ln(n) + 4)\varepsilon + O(\sqrt{\log(n)} \cdot \varepsilon^2)
$$

The error scales **logarithmically** with state dimension $n$, not exponentially.

**Convergence (Theorem 1):**

For smooth inputs with Fourier decay $|\hat{u}(s)| = O(|s|^{-q})$:

$$
\|y_{\text{DPLR}} - y_{\text{Diag}}\|_{L^2} = O(n^{-1}) \sqrt{\ell}
$$

PTD achieves strong convergence to HiPPO, while S4D/S5 only achieve weak convergence (Theorem 2 shows inputs exist where S4D diverges).

## Complexity

| Operation | Runtime | Notes |
|-----------|---------|-------|
| PTD optimization | One-time | Gradient descent at init |
| Per-step recurrence | $O(N)$ | Same as S4D/S5 |
| **Total** | $O(N)$ per step | Only init changes |

**Memory:** Same as diagonal SSM — $O(N)$

PTD only changes initialization, not runtime computation. The perturbation $E$ is computed once at initialization via gradient descent.

## Applicability

- Direct replacement for S4D and S5 initialization — yields S4-PTD and S5-PTD models
- Any SSM that uses diagonal parameterization from HiPPO matrices
- Particularly important for long-range tasks where HiPPO's memory properties matter
- S5-PTD achieves 87.6% average accuracy on Long Range Arena, with improved robustness to Fourier-mode perturbations

## Limitations

- Only addresses initialization robustness — does not change the model architecture or runtime
- The optimization for $E$ requires solving a non-convex problem (gradient descent, no convergence rate guarantees)
- Perturbation bounds involve constants that may be loose in practice
- Benefits are most pronounced on tasks requiring long-range memory; on short-range tasks, S4D/S5 already perform well
- Does not help with non-HiPPO initializations

## Implementation Notes

```python
# PTD initialization (conceptual)
def ptd_init(hippo_matrix, gamma=0.1):
    # Optimize E to minimize condition number + perturbation
    E = optimize_perturbation(hippo_matrix, gamma)
    A_perturbed = hippo_matrix + E
    eigenvalues, eigenvectors = diagonalize(A_perturbed)
    return eigenvalues  # Well-conditioned diagonal SSM
```

## References

- Rushton, Erichson, Mahoney (2024). Robustifying State-space Models for Long Sequences via Approximate Diagonalization. ICLR.
- Gu, Gupta, Goel, Ré (2022). On the Parameterization and Initialization of Diagonal State Space Models (S4D).
- Smith, Warrington, Linderman (2023). Simplified State Space Layers for Sequence Modeling (S5). ICLR.
- Trefethen & Embree (2005). Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and Operators.
