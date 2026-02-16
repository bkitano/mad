# 006: Birkhoff Extension with Frank-Wolfe Rounding

**Category**: approximation
**Gain type**: flexibility
**Source**: Nerem, Luo, Rafiey & Wang (UCSD, NYU; NeurIPS 2025)
**Paper**: [papers/birkhoff-extension-frank-wolfe-rounding.pdf]
**Documented**: 2026-02-15

## Description

Optimizing a real-valued function $f : \mathcal{P}_n \to \mathbb{R}$ over the discrete space of $n \times n$ permutation matrices is computationally prohibitive ($n!$ candidates). The standard remedy is to relax permutations to doubly stochastic matrices (the Birkhoff polytope $\mathcal{D}_n$) using Sinkhorn normalization and optimize via gradient descent. However, Sinkhorn relaxation provides no formal guarantee that rounding the continuous solution back to a permutation preserves solution quality — the rounded permutation can be arbitrarily worse than the relaxed optimum.

**Birkhoff Extension (BE)** solves this by constructing a continuous, almost-everywhere-differentiable extension $F : \mathcal{D}_n \to \mathbb{R}$ of any function $f$ on permutations such that:

1. **Rounding guarantee:** For any doubly stochastic matrix $A \in \mathcal{D}_n$, the rounded solution satisfies $f(\text{round}_S(A)) \leq F_S(A)$. This means rounding can *only improve* the solution — it never degrades quality.
2. **Minima correspondence:** The global minima of the extension $F$ over $\mathcal{D}_n$ coincide with the global minima of $f$ over $\mathcal{P}_n$.
3. **Approximation transfer:** If $A$ is a $C$-approximation for $\min F_S(A)$, then $\text{round}_S(A)$ is a $C$-approximation for $\min f(P)$.

The construction uses a novel **continuous Birkhoff decomposition**: given a score matrix $S \in \mathbb{R}^{n \times n}$ that induces a total order on permutations, the doubly stochastic matrix $A$ is decomposed as $A = \sum_{k=1}^{M} \alpha_k P_k$ where the coefficients $\alpha_k$ are continuous (Lipschitz) functions of $A$. The extension is then $F_S(A) = \sum_k \alpha_k f(P_k)$.

The key insight for optimization is a **Frank-Wolfe algorithm** over the Birkhoff polytope: instead of taking gradient steps (which may leave $\mathcal{D}_n$) and then projecting back, each step moves toward a permutation matrix (a vertex of $\mathcal{D}_n$), automatically preserving double stochasticity. This avoids the Sinkhorn projection entirely. A **dynamic score** mechanism escapes local minima by periodically updating the score matrix $S$ using the best permutations found so far.

The critical advantage over Sinkhorn-based approaches (used in blockwise channel permutation, Gumbel-Sinkhorn networks, etc.) is the **theoretical guarantee** that any continuous optimization progress translates directly into discrete solution quality. With Sinkhorn, the doubly stochastic optimum may round to a poor permutation; with BE, this cannot happen.

## Mathematical Form

**Birkhoff Decomposition (Theorem 2.1):**

Any doubly stochastic matrix $A \in \mathcal{D}_n$ can be decomposed as:

$$
A = \sum_{k=1}^{M} \alpha_k P_k, \quad M \leq n^2 - n + 1, \quad \alpha_k > 0, \quad \sum_k \alpha_k = 1, \quad P_k \in \mathcal{P}_n
$$

**Score-Induced Ordering (Definition 2.6):**

Given a score matrix $S \in \mathbb{R}^{n \times n}$, the score of a permutation $P$ is:

$$
\langle S, P \rangle = \sum_{i,j} S(i,j) P(i,j)
$$

$S$ is *identifying* if it assigns a unique score to every permutation, inducing a total order on $\mathcal{P}_n$. A simple identifying score matrix is $S(i,j) = 2^{i + nj}$.

**Continuous Birkhoff Decomposition (Definition 2.3):**

Given an enumeration $\{P_\ell\}_{\ell=1}^{n!}$ ordered by decreasing score, the continuous Birkhoff decomposition of $A$ is $(\alpha_\ell, P_\ell)_{\ell=1}^{n!}$ where:

$$
\alpha_\ell = \min_{ij} \left\{ A(i,j) - \sum_{m=1}^{\ell-1} \alpha_m P_m(i,j) \;\middle|\; P_\ell(i,j) = 1 \right\}
$$

**Birkhoff Extension (Definition 2.9):**

For any $f : \mathcal{P}_n \to \mathbb{R}$, the $S$-induced Birkhoff extension $F_S : \mathcal{D}_n \to \mathbb{R}$ is:

$$
F_S(A) = \sum_{k=1}^{M} \alpha_k f(P_k)
$$

where $(\alpha_k, P_k)_{k=1}^{M}$ are the non-zero terms of the continuous Birkhoff decomposition of $A$.

**Key Properties:**

- **Property 1 (Differentiability):** $F_S$ is Lipschitz continuous and a.e. differentiable, with gradient:

$$
\nabla_A F_S(A) = \sum_{\ell \in L_+} (\nabla_A \alpha_\ell) f(P_\ell), \quad L_+ = \{\ell : \alpha_\ell > 0\}
$$

- **Property 3 (Minima correspondence):** $\min_{A \in \mathcal{D}_n} F_S(A) = \min_{P \in \mathcal{P}_n} f(P)$

- **Property 4 (Rounding guarantee):** $f(\text{round}_S(A)) \leq F_S(A)$ for all $A \in \mathcal{D}_n$

**Rounding (Definition 2.10):**

$$
\text{round}_S(A) = \arg\min_{P_k : k \in [M]} f(P_k)
$$

i.e., select the best permutation from the Birkhoff decomposition.

**Frank-Wolfe Algorithm (Algorithm 3):**

At each iteration $t$:

$$
P_t \leftarrow \arg\max_{P \in \mathcal{P}_n} \langle \nabla F_S(A_t), P \rangle
$$

$$
A_{t+1} \leftarrow (1 - \lambda_t) A_t + \lambda_t P_t
$$

Since $P_t \in \mathcal{P}_n \subset \mathcal{D}_n$ and $A_t \in \mathcal{D}_n$, the convex combination $A_{t+1}$ remains in $\mathcal{D}_n$ automatically. The linear subproblem $\arg\max_{P} \langle \nabla F_S(A_t), P \rangle$ is a maximum weight bipartite matching, solvable in $O(n^3)$ by the Hungarian algorithm.

**Dynamic Score Algorithm (Algorithm 4):**

To escape local minima, periodically update the score matrix:

$$
\mathcal{P}^* \leftarrow \mathcal{P}^* \cup \text{Birkhoff}_S(A_t)
$$

$$
S' \leftarrow \frac{1}{2n} Q + \arg\min_{P \in \mathcal{P}^*} f(P)
$$

where $Q$ is a random noise matrix with entries in $[0, 1/n^2]$. The rounding guarantee ensures the quality of the rounded solution never decreases when the score is updated.

## Complexity

| Operation | Sinkhorn Relaxation | Birkhoff Extension (BE) |
|-----------|--------------------|-----------------------|
| Extension computation | $O(Ln^2)$ ($L$ Sinkhorn iters) | $O(n^5)$ (score-induced decomp.) |
| Gradient of extension | Through $L$ unrolled iterations | $\sum_{\ell \in L_+} (\nabla_A \alpha_\ell) f(P_\ell)$ |
| Frank-Wolfe step | N/A (uses projected GD) | $O(n^3)$ (Hungarian algorithm) |
| Rounding guarantee | None | $f(\text{round}(A)) \leq F(A)$ guaranteed |
| Practical truncation | N/A | Use first $k=5$ terms of decomposition |

**Practical speedup:** The $O(n^5)$ theoretical cost is reduced by truncating the Birkhoff decomposition to the first $k = 5$ terms (only 5 matchings computed). In experiments, the number of permutations with non-zero coefficients is usually far fewer than $n^2$.

**Memory:** $O(n^2)$ for the doubly stochastic matrix $A$, plus $O(Mn^2)$ for the $M \leq n^2 - n + 1$ permutations in the decomposition.

**Experimental results (QAP, QAPLIB, $n = 12$–$256$):**

| Method | Avg Gap from Best Known | Runtime (s) |
|--------|------------------------|-------------|
| Gurobi (Kaufman-Broeckx) | 7.67% | 90.3 |
| 2-opt heuristic | 10.38% | 22.82 |
| BE (Random $S$ Init, Alg. 4) | **6.30%** | 23.01 |

## Applicability

- **Blockwise Sinkhorn channel permutation (PermLLM):** BE provides a principled alternative to Sinkhorn relaxation for learning block-diagonal permutations. Within each $B \times B$ block, the permutation objective (minimize output discrepancy) can be extended to $\mathcal{D}_B$ via BE, and the Frank-Wolfe optimizer guarantees that the rounded permutation is at least as good as the continuous solution. However, the $O(B^5)$ decomposition cost limits this to small blocks ($B \leq 30$); for $B = 64$, the truncated variant with $k = 5$ terms is practical.
- **Quadratic assignment problem (QAP):** Primary application in the paper. Outperforms Gurobi and heuristic solvers on QAPLIB instances up to $n = 256$.
- **Neural combinatorial optimization:** BE can serve as a differentiable loss for training neural networks to solve combinatorial problems over permutations (e.g., TSP). The neural network outputs a doubly stochastic matrix, which is rounded via the Birkhoff decomposition.
- **Channel permutation with quality guarantees:** Unlike Sinkhorn + STE (which provides no approximation guarantee), BE ensures that any optimization progress in the continuous space translates to provably better discrete solutions.
- **Local improvement of any existing solution:** Given any approximate permutation $P_\text{approx}$, using it as the score matrix for BE and optimizing yields a solution at least as good as $P_\text{approx}$ (Property 4.2).

## Limitations

- **$O(n^5)$ decomposition cost:** Computing the full score-induced Birkhoff decomposition requires $O(n^2)$ maximum weight matchings, each costing $O(n^3)$. For large $n$ (e.g., $n = 4096$ channels in LLMs), this is prohibitive — only practical with truncation to $k \ll n^2$ terms.
- **Not end-to-end differentiable with neural networks:** While BE is a.e. differentiable, integrating it into a neural network training loop requires computing the extension and its gradient at each step, which is much more expensive per iteration than Sinkhorn.
- **Frank-Wolfe convergence:** The extension $F_S$ is generally non-convex, so Frank-Wolfe may converge to local minima. The dynamic score mechanism mitigates this but adds complexity.
- **Score matrix dependence:** Different score matrices $S$ yield different extensions, and the choice of $S$ affects optimization landscape. Random identifying scores work well empirically but offer no worst-case guarantee on convergence speed.
- **Truncation tradeoff:** Truncating the decomposition to $k = 5$ terms preserves rounding guarantees but reduces the expressiveness of the gradient $\nabla_A F_S$ — fewer permutations contribute to the gradient signal.

## Implementation Notes

```python
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def score_induced_birkhoff_decomposition(A, S, max_terms=5):
    """
    Continuous Birkhoff decomposition of A induced by score matrix S.

    Args:
        A: (n, n) doubly stochastic matrix
        S: (n, n) identifying score matrix
        max_terms: maximum number of decomposition terms

    Returns:
        alphas: list of coefficients
        perms: list of (n, n) permutation matrices
    """
    n = A.shape[0]
    B = A.clone()
    alphas, perms = [], []

    for k in range(max_terms):
        if B.max() < 1e-10:
            break

        # Find maximum score matching of B (score-induced order)
        # Construct bipartite graph: edge (i,j) exists iff B(i,j) > 0
        # Weight of edge = S(i,j)
        cost = -S.numpy() * (B.numpy() > 1e-10).astype(float)
        cost[B.numpy() <= 1e-10] = 1e6  # exclude zero entries
        row_ind, col_ind = linear_sum_assignment(cost)

        P = torch.zeros(n, n)
        P[row_ind, col_ind] = 1.0

        # alpha_k = min B(i,j) where P(i,j) = 1
        alpha = min(B[row_ind[i], col_ind[i]].item()
                     for i in range(n))

        if alpha <= 0:
            break

        alphas.append(alpha)
        perms.append(P)
        B = B - alpha * P

    return alphas, perms

def birkhoff_extension(A, S, f, max_terms=5):
    """
    Compute Birkhoff extension F_S(A) = sum_k alpha_k * f(P_k).

    Args:
        A: (n, n) doubly stochastic matrix
        S: (n, n) identifying score matrix
        f: function from permutation matrices to R
        max_terms: truncation depth

    Returns:
        F_val: scalar extension value
        best_perm: best permutation from decomposition (rounding)
    """
    alphas, perms = score_induced_birkhoff_decomposition(A, S, max_terms)

    F_val = sum(a * f(P) for a, P in zip(alphas, perms))

    # Rounding: select best permutation from decomposition
    best_idx = min(range(len(perms)), key=lambda k: f(perms[k]))
    best_perm = perms[best_idx]

    # Guarantee: f(best_perm) <= F_val
    return F_val, best_perm

def frank_wolfe_birkhoff(f, grad_F, n, S, T=100, lr=0.1):
    """
    Frank-Wolfe optimization of Birkhoff extension.

    Maintains feasibility (A stays in Birkhoff polytope) automatically
    by stepping toward permutation vertices.

    Args:
        f: objective on permutations
        grad_F: gradient of Birkhoff extension
        n: matrix dimension
        S: identifying score matrix
        T: number of iterations
        lr: step size

    Returns:
        best_perm: best permutation found
    """
    # Initialize with random doubly stochastic matrix
    A = torch.ones(n, n) / n  # uniform DSM
    best_perm = None
    best_val = float('inf')

    for t in range(T):
        # Compute gradient of Birkhoff extension
        G = grad_F(A, S)

        # Linear minimization: find permutation most aligned with -gradient
        cost = -G.numpy()  # maximize <gradient, P> for descent
        row_ind, col_ind = linear_sum_assignment(cost)
        P_t = torch.zeros(n, n)
        P_t[row_ind, col_ind] = 1.0

        # Frank-Wolfe step: convex combination stays in D_n
        lambda_t = lr / (t + 1)  # diminishing step size
        A = (1 - lambda_t) * A + lambda_t * P_t

        # Round and check
        _, round_perm = birkhoff_extension(A, S, f, max_terms=5)
        round_val = f(round_perm)
        if round_val < best_val:
            best_val = round_val
            best_perm = round_perm

    return best_perm

# Identifying score matrix (simple exponential)
# S(i,j) = 2^(i + n*j) ensures unique scores for all permutations
def identifying_score(n):
    S = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            S[i, j] = 2.0 ** (i + n * j)
    return S
```

## References

- Nerem, R.R., Luo, Z., Rafiey, A. & Wang, Y. (2025). Differentiable Extensions with Rounding Guarantees for Combinatorial Optimization over Permutations. NeurIPS 2025. arXiv:2411.10707.
- Birkhoff, G. (1946). Three observations on linear algebra. Univ. Nac. Tucuman Revista A., 5:147-151.
- Dufoss, F. & Ucar, B. (2016). Notes on Birkhoff-von Neumann decomposition of doubly stochastic matrices. Linear Algebra and its Applications, 497:108-115.
- Frank, M. & Wolfe, P. (1956). An algorithm for quadratic programming. Naval Research Logistics Quarterly, 3(1-2):95-110.
- Jaggi, M. (2013). Revisiting Frank-Wolfe: Projection-free sparse convex optimization. ICML 2013.
- Mena, G. et al. (2018). Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018.
- Karalias, N. & Loukas, A. (2020). Erdos goes neural: an unsupervised learning framework for combinatorial optimization on graphs. NeurIPS 2020.
