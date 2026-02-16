# 152: CWY Parametrization for Parallelized Orthogonal Optimization

**Category**: decomposition
**Gain type**: efficiency
**Source**: Likhosherstov, Davis, Choromanski, Weller (AISTATS 2021)
**Paper**: [papers/cwy-parametrization-stiefel.pdf]
**Documented**: 2026-02-15

## Description

The CWY (Compact WY) parametrization converts the inherently sequential computation of a product of $L$ Householder reflections into a **parallelizable** form that maps to tensor cores (GPU) and matrix multiply units (TPU). While the naive Householder product $Q = H(v^{(1)}) \cdots H(v^{(L)})$ requires $L$ sequential matrix-vector products, the CWY form $Q = I - US^{-1}U^\top$ precomputes a single $L \times L$ upper-triangular matrix $S$ and then evaluates the product via two BLAS-3 matrix-matrix multiplications and one triangular solve.

The key insight is that Joffrain et al. (2006) showed the compact WY representation satisfies:

$$
S^{-1} + S^{-\top} = U^\top U
$$

where $S$ is upper-triangular, meaning $S$ can be reconstructed from $U^\top U$ (a single matmul) and then inverted via back-substitution. Both operations are highly parallel on GPUs. The resulting algorithm achieves:

- **$O(\log(LN))$ parallel complexity** with $O(L^2 \log L)$ preprocessing — vs $O(L)$ parallel complexity for sequential Householder
- **1–3 orders of magnitude wall-clock speedup** over matrix exponential, Cayley transform, and sequential Householder on GPU/TPU
- **20× speedup** over sequential Householder reflections at $N = 1024$, $L = N$ on TPU

The paper also introduces **T-CWY** (Truncated CWY), extending the approach to the Stiefel manifold $\text{St}(N, M)$ with $M < N$ — parameterizing tall-skinny orthogonal matrices with fewer FLOPs than any competing method.

## Mathematical Form

**Core Theorem (Compact WY for Householder Products):**

Given $L$ nonzero vectors $v^{(1)}, \ldots, v^{(L)} \in \mathbb{R}^N$, the product of Householder reflections:

$$
H(v^{(1)}) \cdots H(v^{(L)}) = I - U S^{-1} U^\top
$$

where:
- $U = \left[\frac{v^{(1)}}{\|v^{(1)}\|_2} \;\cdots\; \frac{v^{(L)}}{\|v^{(L)}\|_2}\right] \in \mathbb{R}^{N \times L}$ — normalized reflection vectors
- $H(v) = I - 2\frac{vv^\top}{\|v\|_2^2}$ — standard Householder reflection
- $S = \frac{1}{2}I + \text{striu}(U^\top U) \in \mathbb{R}^{L \times L}$ — upper-triangular matrix
- $\text{striu}(\cdot)$ extracts the strict upper-triangular part (diagonal and lower zeroed)

**Reconstruction of $S$:**

$$
S = \frac{1}{2}I + \text{striu}(U^\top U)
$$

This follows from the identity $S^{-1} + S^{-\top} = U^\top U$ (Joffrain et al., 2006). Since $S$ is upper-triangular:
- Diagonal: $S_{ii} = \frac{1}{2}(U^\top U)_{ii} = \frac{1}{2}$ (since columns of $U$ are unit-normalized)
- Strict upper triangle: $S_{ij} = (U^\top U)_{ij}$ for $i < j$

**Forward Pass (RNN with CWY transition matrix):**

Given $Q = I - US^{-1}U^\top$ as the transition matrix, the RNN rollout $y_t = Qh_{t-1} + \text{input}$ expands to:

$$
u_t := U^\top h_{t-1}, \quad s_t := S^{-1} u_t, \quad y_t := h_{t-1} - U s_t + \text{input}
$$

The key operations are:
1. $U^\top h_{t-1}$: matrix-vector product, $O(NL)$ — **parallelizable** across $L$
2. $S^{-1} u_t$: triangular solve, $O(L^2)$ — **parallelizable** in $O(L \log L)$ via back-substitution
3. $U s_t$: matrix-vector product, $O(NL)$ — **parallelizable** across $N$

**Precomputation (done once before RNN rollout):**

1. Compute $G = U^\top U$: single matmul, $O(NL^2)$ — tensor-core friendly
2. Extract $S = \frac{1}{2}I + \text{striu}(G)$: $O(L^2)$
3. Invert $S$: back-substitution on $L \times L$ upper-triangular matrix, $O(L^2 \log L)$ parallel, $O(L^3)$ serial

**T-CWY Extension for Stiefel Manifold $\text{St}(N, M)$ with $M < N$:**

Parameterize $\Omega \in \text{St}(N, M)$ by taking the first $M$ columns of a full $N \times N$ CWY-parameterized orthogonal matrix:

$$
\Omega = \gamma_{N,M}(v^{(1)}, \ldots, v^{(M)}) = \begin{bmatrix} I \\ \mathbf{0} \end{bmatrix}^\top - U S^{-1} U_1^\top \in \mathbb{R}^{N \times M}
$$

where $U_1 \in \mathbb{R}^{M \times M}$ is the upper $M \times M$ submatrix of $U$, and $S$ is $M \times M$ upper-triangular. This requires only an $M \times M$ triangular inverse (not $N \times N$), reducing the cost.

**Key Definitions:**

- $N$ — dimension of the orthogonal matrix (hidden state size)
- $L$ — number of Householder reflections ($L \leq N$; $L = N$ covers all of $\mathcal{O}(N)$)
- $\mathcal{O}_L(N) = \{H(v^{(1)}) \cdots H(v^{(L)}) \mid v^{(i)} \in \mathbb{R}^N \setminus \{0\}\}$ — set reachable with $L$ reflections
- $\text{St}(N, M) = \{\Omega \in \mathbb{R}^{N \times M} \mid \Omega^\top \Omega = I_M\}$ — Stiefel manifold

## Complexity

| Operation | Sequential HR | CWY | Cayley/Exp |
|-----------|--------------|-----|------------|
| **Serial time** (forward) | $TLN$ | $TLN + L^2N + L^3$ | $TN^2 + N^3$ |
| **Parallel time** (forward) | $TL \log N$ | $T \log(LN) + L^2 \log L$ | $T \log N + N^2 \log N$ |
| **Preprocessing** | — | $O(NL^2 + L^3)$ | $O(N^3)$ |
| **Mat-vec per step** | $L$ sequential | 2 parallel + 1 tri-solve | 1 dense |
| **Solution domain** | $\mathcal{O}_L(N)$ | $\mathcal{O}_L(N)$ (identical) | $\mathcal{O}^{+1}(N) \setminus \Theta$ |

For **T-CWY on $\text{St}(N, M)$**:

| Approach | FLOPs |
|----------|-------|
| RGD-C-QR | $10NM^2 - 2M^3/3$ |
| RGD-E-QR | $14NM^2 - 2M^3/3$ |
| RGD-C-C | $28NM^2 + 16M^3$ |
| OWN | $4NM^2 + 14M^3/3$ |
| **T-CWY** | $\mathbf{4NM^2 + 7M^3/3}$ |

T-CWY achieves the **smallest FLOPs** among all Stiefel optimization methods when $N \geq M$.

**Memory:** $O(NL)$ for storing $L$ reflection vectors + $O(L^2)$ for $S^{-1}$. No need to materialize the full $N \times N$ matrix.

## Applicability

- **Orthogonal RNNs:** Direct application — parameterize transition matrix $W$ as CWY Householder product. Demonstrated on:
  - Copying task ($T = 1000$, $N = 190$): converges to zero cross-entropy faster than EXPRNN, SCORNN
  - Pixel-by-pixel MNIST ($N = 512$): achieves $> 95\%$ accuracy, matching best baselines
  - Neural machine translation (Tatoeba EN→ES, $N = 1024$): lowest test perplexity (1.41) with fewest parameters (23M) and fastest training (198 min), outperforming LSTM (1.46, 37M, 232 min)
- **Convolutional orthogonal RNNs (ConvNERU):** T-CWY parameterizes convolutional transition kernels on $\text{St}(q^2 f_{\text{out}}, f_{\text{out}})$. Demonstrated on KTH video prediction dataset, outperforming ConvLSTM while using $4.5\times$ fewer parameters and $2.5\times$ less GPU memory
- **DeltaNet / DeltaProduct training:** The UT transform used in DeltaNet is the lower-triangular analog of CWY. CWY provides the same parallelization benefit for models using Householder transition matrices
- **Normalizing flows:** Orthogonal layers parameterized via CWY enable volume-preserving invertible transformations with efficient Jacobian computation
- **Weight orthogonalization in CNNs:** Maintaining orthogonal weights improves Lipschitz bounds and adversarial robustness (Parseval networks)

## Limitations

- **$O(L^2 \log L)$ parallel preprocessing:** The triangular inversion of $S$ adds $L^2 \log L$ parallel time. For $L = N$ (full coverage of $\mathcal{O}(N)$), this is $N^2 \log N$ — still better than $O(N^3)$ for matrix exponent/Cayley, but non-trivial for very large $N$
- **Expressivity vs. speed trade-off:** Smaller $L$ gives faster computation but restricts the reachable subset of $\mathcal{O}(N)$. The sweet spot in the NMT experiments was $L = 128$ for $N = 1024$ (1/8 coverage)
- **Sequential RNN rollout persists:** CWY parallelizes the *per-step* computation of $Qh$, but the RNN rollout across time steps $t = 1, \ldots, T$ remains sequential (unless combined with parallel scan, which requires diagonal or scalar transitions)
- **Not applicable to non-orthogonal transitions:** CWY specifically parameterizes $\mathcal{O}(N)$. For generalized Householder ($\beta \neq 2$), the identity $S^{-1} + S^{-\top} = U^\top U$ does not hold directly, and the UT transform (used in DeltaNet) must be used instead
- **Convergence rate:** SGD on CWY converges at $o(K^{-0.5+\epsilon})$ gradient norm, which is standard but not accelerated
- **Numerical precision:** Mixed-precision training requires careful handling — $S$ is upper-triangular with diagonal entries $= 0.5$, so $S^{-1}$ can amplify errors if $U^\top U$ has large off-diagonal entries

## Implementation Notes

```python
import torch

def cwy_precompute(v_list):
    """
    Precompute CWY representation from L Householder vectors.

    Args:
        v_list: list of L vectors, each (N,) - unnormalized

    Returns:
        U: (N, L) - normalized Householder vectors
        S_inv: (L, L) - inverse of the upper-triangular S matrix
    """
    L = len(v_list)
    N = v_list[0].shape[0]

    # Normalize vectors -> columns of U
    U = torch.stack([v / v.norm() for v in v_list], dim=1)  # (N, L)

    # Compute S = 0.5 * I + striu(U^T U)
    # This is a TENSOR CORE matmul: (L, N) @ (N, L) -> (L, L)
    G = U.T @ U  # (L, L)
    S = 0.5 * torch.eye(L) + torch.triu(G, diagonal=1)

    # Invert S via back-substitution (upper-triangular)
    # O(L^2) serial, O(L^2 log L) parallel
    S_inv = torch.linalg.solve_triangular(S, torch.eye(L), upper=True)

    return U, S_inv


def cwy_matvec(U, S_inv, h):
    """
    Compute Q @ h = (I - U S^{-1} U^T) h

    Args:
        U: (N, L) - normalized Householder vectors
        S_inv: (L, L) - precomputed S^{-1}
        h: (N,) - input vector

    Returns:
        (N,) - Q @ h
    """
    # Step 1: u = U^T h  — parallel across L
    u = U.T @ h  # (L,)

    # Step 2: s = S^{-1} u — triangular solve, O(L^2)
    s = S_inv @ u  # (L,)

    # Step 3: result = h - U s — parallel across N
    return h - U @ s


def cwy_rnn_forward(U, S_inv, V, x_seq, h0=None):
    """
    RNN forward pass with CWY orthogonal transition.

    Q = I - U S^{-1} U^T  (orthogonal transition matrix)
    h_t = sigma(Q h_{t-1} + V x_t + b)

    Args:
        U: (N, L) - Householder vectors
        S_inv: (L, L) - precomputed S^{-1}
        V: (N, K) - input projection
        x_seq: (T, K) - input sequence
        h0: (N,) - initial hidden state
    """
    N = U.shape[0]
    T = x_seq.shape[0]

    h = h0 if h0 is not None else torch.zeros(N)
    outputs = []

    for t in range(T):
        # CWY matrix-vector product (parallelizable per step)
        Qh = cwy_matvec(U, S_inv, h)
        h = torch.tanh(Qh + V @ x_seq[t])
        outputs.append(h)

    return torch.stack(outputs)


# For Stiefel manifold St(N, M) with M < N:
def tcwy_precompute(v_list, M):
    """
    Truncated CWY for Stiefel manifold.
    Returns N x M orthogonal matrix using M Householder vectors.

    Complexity: 4NM^2 + 7M^3/3 FLOPs (smallest among all methods)
    """
    U, S_inv = cwy_precompute(v_list[:M])  # Only M vectors needed

    # Extract first M columns of (I - U S^{-1} U^T)
    U1 = U[:M, :]  # (M, M) upper submatrix of U
    Omega = torch.eye(U.shape[0], M) - U @ S_inv @ U1.T

    return Omega
```

**Timing results (from paper, Figure 2):**

| Method | $L = N = 1024$ | Time (sec/10 epochs) |
|--------|---------------|---------------------|
| Sequential HR | — | 23,180 |
| CWY ($L = 1024$) | — | 1,111 |
| CWY ($L = 512$) | — | 338 |
| CWY ($L = 256$) | — | 213 |
| CWY ($L = 128$) | — | 198 |

**20× speedup** at $L = N = 1024$; **117× speedup** at $L = 128$ (with only modest quality loss).

**GPU efficiency analysis:**
- $U^\top U$ matmul: perfect tensor core utilization ($(L \times N) \times (N \times L)$)
- $S^{-1}$ triangular solve: sequential in $L$, but $L \leq N$ and $L$ is typically small (64–256)
- Per-step matmuls $U^\top h$ and $U s$: map to BLAS-2 (gemv) — efficient when batched across sequences
- Main bottleneck: RNN rollout is still sequential across $T$ time steps

## References

- Likhosherstov, V., Davis, J., Choromanski, K., & Weller, A. (2021). CWY Parametrization: a Solution for Parallelized Optimization of Orthogonal and Stiefel Matrices. AISTATS 2021, PMLR Volume 130. arXiv:2004.08675.
- Joffrain, T., Low, T. M., Quintana-Ortí, E. S., van de Geijn, R. A., & Van Zee, F. G. (2006). Accumulating Householder Transformations, Revisited. ACM Trans. Math. Softw., 32(2), 169–179.
- Mhammedi, Z., Hellicar, A., Rahman, A., & Bailey, J. (2017). Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections. ICML 2017.
- Schreiber, R. & Van Loan, C. (1989). A Storage-Efficient WY Representation for Products of Householder Transformations. SIAM J. Sci. Stat. Comput., 10(1), 53–57.
- Helfrich, K., Willmott, D., & Ye, Q. (2018). Orthogonal Recurrent Neural Networks with Scaled Cayley Transform. ICML 2018.
- Lezcano-Casado, M. & Martínez-Rubio, D. (2019). Cheap Orthogonal Constraints in Neural Networks. ICML 2019.
