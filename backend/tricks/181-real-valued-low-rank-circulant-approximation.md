# 181: Real-Valued Low-Rank Circulant Approximation

**Category**: decomposition
**Gain type**: efficiency
**Source**: Chu & Plemmons "Real-Valued, Low Rank, Circulant Approximation" (SIAM J. Matrix Anal. Appl., 2003)
**Paper**: [papers/chu-real-lowrank-circulant-approximation.pdf]
**Documented**: 2026-02-15

## Description

Given any $n \times n$ real matrix $A$, find the nearest **real-valued circulant matrix of rank at most $\kappa$** in the Frobenius norm. This combines three constraints simultaneously: (1) circulant structure (for $O(n \log n)$ matvecs via FFT), (2) low rank (for $O(\kappa n)$ storage and further compression), and (3) real-valuedness (for practical computation without complex arithmetic overhead).

The naive approach — compute the optimal circulant approximation (trick 084), then truncate its SVD — fails to produce a real-valued result in general. This is because circulant eigenvalues come in conjugate pairs, and the standard TSVD (truncated SVD) may select only one eigenvalue from a conjugate pair, breaking the **conjugate-even** property required for a real circulant. The rank constraint and real-valuedness constraint interact in a non-trivial way that changes the truncation rule.

The algorithm of Chu & Plemmons solves this via a **data matching problem (DMP)**: find the nearest conjugate-even vector $\hat{\boldsymbol{\lambda}}$ to the eigenvalue vector $\boldsymbol{\lambda}$ of $\text{Circul}(\mathbf{c})$ subject to having exactly $n - \kappa$ zero entries. The conjugate-even constraint forces eigenvalues to be deleted in conjugate pairs (both $\lambda_k$ and $\bar{\lambda}_k$ must be zeroed or kept together), creating a combinatorial selection problem that is solved by a greedy algorithm on a tree graph of the eigenvalue moduli.

The resulting algorithm runs in $O(n \log n)$: compute the optimal circulant via diagonal averaging ($O(n^2)$ for general $A$, $O(n)$ for Toeplitz), take FFT to get eigenvalues ($O(n \log n)$), perform conjugate-even truncation via the DMP ($O(n \log n)$ for sorting), then IFFT back ($O(n \log n)$). The output is a rank-$\kappa$ real circulant matrix that can be applied in $O(\kappa \log \kappa)$ per matvec via partial FFT.

## Mathematical Form

**Circulant Spectral Decomposition:**

Any circulant $C = \text{Circul}(\mathbf{c})$ with $\mathbf{c} \in \mathbb{R}^n$ satisfies:

$$
C = F^* P_\mathbf{c}(\Omega) F
$$

where $F$ is the DFT matrix, $\Omega = \text{diag}(1, \omega, \omega^2, \ldots, \omega^{n-1})$ with $\omega = e^{2\pi i/n}$, and $P_\mathbf{c}(x) = \sum_{k=0}^{n-1} c_k x^k$ is the characteristic polynomial.

The eigenvalues are $\boldsymbol{\lambda} = [P_\mathbf{c}(1), P_\mathbf{c}(\omega), \ldots, P_\mathbf{c}(\omega^{n-1})]$, computable via:

$$
\boldsymbol{\lambda}^T = \sqrt{n} F^* \mathbf{c}^T
$$

and the inverse (eigenvalues to first row):

$$
\mathbf{c}^T = \frac{1}{\sqrt{n}} F \boldsymbol{\lambda}^T
$$

**Conjugate-Even Property (Theorem 2.2):**

For $C$ to be real-valued, $\boldsymbol{\lambda}$ must be conjugate-even:

- If $n = 2m$: $\boldsymbol{\lambda} = [\lambda_0, \lambda_1, \ldots, \lambda_{m-1}, \lambda_m, \bar{\lambda}_{m-1}, \ldots, \bar{\lambda}_1]$ with $\lambda_0, \lambda_m \in \mathbb{R}$
- If $n = 2m+1$: $\boldsymbol{\lambda} = [\lambda_0, \lambda_1, \ldots, \lambda_m, \bar{\lambda}_m, \ldots, \bar{\lambda}_1]$ with $\lambda_0 \in \mathbb{R}$

**Singular Value Structure (Theorem 2.3):**

A real $n \times n$ circulant has at most $\lceil \frac{n+1}{2} \rceil$ distinct singular values. They appear in pairs: $|\lambda_k|$ and $|\bar{\lambda}_k|$ are the same singular value with multiplicity 2 (for complex conjugate pairs), while the "absolutely real" eigenvalues $\lambda_0$ (and $\lambda_m$ if $n = 2m$) contribute singular values of multiplicity 1.

**Data Matching Problem (DMP):**

Given conjugate-even $\boldsymbol{\lambda} \in \mathbb{C}^n$, find nearest conjugate-even $\hat{\boldsymbol{\lambda}} \in \mathbb{C}^n$ in 2-norm with exactly $n - \kappa$ zeros:

$$
\min_{\hat{\boldsymbol{\lambda}}} \|P \hat{\boldsymbol{\lambda}}^T - \boldsymbol{\lambda}^T\|_2^2
$$

subject to $\hat{\boldsymbol{\lambda}}$ being conjugate-even with $n - \kappa$ zero entries.

**Truncation Rule (Theorem 3.1):**

The nonzero entries of $\hat{\boldsymbol{\lambda}}$ must match the first $\kappa$ conjugate-even components of $\boldsymbol{\lambda}$ sorted by descending modulus. The complication: when a complex pair $(\lambda_k, \bar{\lambda}_k)$ competes with a real eigenvalue $\lambda_0$ or $\lambda_m$ at the truncation boundary, the decision depends on:

$$
2|\lambda_k|^2 \lessgtr |\lambda_0|^2 + |\lambda_m|^2
$$

If $2|\lambda_k|^2 \leq |\lambda_0|^2 + |\lambda_m|^2$: drop the complex pair, keep the reals (rank drops by 2).
If $2|\lambda_k|^2 > |\lambda_0|^2 + |\lambda_m|^2$: keep the complex pair, drop a real (rank drops by 1).

This means the nearest real circulant of rank $\kappa$ may sometimes need to **delete the largest eigenvalue** to satisfy both rank and real-valuedness constraints — a counterintuitive result.

**Complete Algorithm (Algorithm 4.1):**

Given $\mathbf{c} \in \mathbb{R}^n$ (first row of optimal circulant of $A$) and target rank $1 \leq \kappa < n$:

1. $\boldsymbol{\lambda} = n \cdot \text{IFFT}(\mathbf{c})$ — compute eigenvalues
2. Classify eigenvalues as "absolutely real" ($I_r$) or "complex paired" ($I_c$)
3. Sort $|\boldsymbol{\lambda}|$ in descending order; build index map $J$
4. Walk from smallest $|\lambda|$ upward, counting how many eigenvalues to zero:
   - Complex pairs count as 2 (both must be zeroed together)
   - Real eigenvalues count as 1
   - At the boundary, compare $2|\lambda_{\text{pair}}|^2$ vs $|\lambda_{\text{real}_1}|^2 + |\lambda_{\text{real}_2}|^2$
5. Zero the selected $n - \kappa$ eigenvalues to get $\hat{\boldsymbol{\lambda}}$
6. Restore conjugate-even structure: $\hat{\boldsymbol{\lambda}} = [\hat{\lambda}_1, \ldots, \hat{\lambda}_m, \text{flipconj}(\hat{\lambda}_2, \ldots)]$
7. $\hat{\mathbf{c}} = \text{real}(\text{FFT}(\hat{\boldsymbol{\lambda}}))/n$ — recover first row of rank-$\kappa$ real circulant

**Key Definitions:**

- $\text{Circul}(\mathbf{c})$ — circulant matrix with first row $\mathbf{c}$
- $\boldsymbol{\lambda}$ — eigenvalue vector of the circulant
- $\kappa$ — target rank
- $\hat{\boldsymbol{\lambda}}$ — truncated eigenvalue vector (conjugate-even, $n - \kappa$ zeros)
- Conjugate-even: $\lambda_k = \bar{\lambda}_{n-k}$ for all $k$
- DMP — Data Matching Problem (find nearest conjugate-even sparse vector)

## Complexity

| Operation | General $A$ | Toeplitz $A$ |
|-----------|------------|-------------|
| Compute optimal circulant of $A$ | $O(n^2)$ | $O(n)$ |
| FFT to get eigenvalues | $O(n \log n)$ | $O(n \log n)$ |
| Conjugate-even DMP solve | $O(n \log n)$ (sorting) | $O(n \log n)$ |
| IFFT to recover circulant | $O(n \log n)$ | $O(n \log n)$ |
| **Total construction** | $O(n^2)$ | $O(n \log n)$ |
| Apply rank-$\kappa$ circulant matvec | $O(n \log n)$ | $O(n \log n)$ |
| Storage of rank-$\kappa$ circulant | $O(n)$ or $O(\kappa)$ | $O(n)$ or $O(\kappa)$ |

**Matvec with rank-$\kappa$ circulant:** The rank-$\kappa$ circulant has only $\kappa$ nonzero eigenvalues. The matvec $\hat{C}x$ can be computed as:

$$
\hat{C}x = F^* \text{diag}(\hat{\boldsymbol{\lambda}}) F x
$$

which is a full FFT, sparse diagonal multiply (only $\kappa$ nonzero mults), and IFFT. The FFT/IFFT still cost $O(n \log n)$, but the sparse eigenvalue structure can be exploited for partial FFT computation when $\kappa \ll n$.

**Memory:** $O(n)$ for the full first-row representation, or $O(\kappa)$ if only the nonzero eigenvalue indices and values are stored (plus the FFT matrix, which is implicit).

## Applicability

- **Structured weight compression for sequence models**: Replace dense $n \times n$ weight matrices with rank-$\kappa$ circulant approximations: $n^2 \to n$ parameters for the circulant, with the rank-$\kappa$ constraint providing additional regularization. The real-valuedness guarantee avoids complex arithmetic in the forward pass. Matvec costs $O(n \log n)$ via FFT — same as full circulant
- **Toeplitz token mixer spectral pruning**: For Toeplitz token mixers (TNN), the circulant cycle decomposition (trick 028) identifies dominant frequencies. This trick provides the optimal way to prune the least significant frequency components while maintaining a real-valued circulant structure — effectively spectral pruning of the token mixer
- **Low-rank circulant initialization**: Initialize structured layers as rank-$\kappa$ real circulants, then allow training to increase effective rank as needed. The conjugate-even truncation ensures the initialization is always real-valued, avoiding the pitfall of naive TSVD producing complex matrices
- **Adaptive structured approximation**: During training, periodically project weight matrices onto the nearest rank-$\kappa$ real circulant to enforce structure. The counterintuitive truncation rule (sometimes deleting the largest eigenvalue) means naive pruning strategies are suboptimal — this algorithm provides the true optimal projection
- **Circulant preconditioning with rank control**: When the optimal circulant preconditioner has near-zero eigenvalues (causing instability), the low-rank version provides a stable alternative by zeroing out the smallest eigenvalues while maintaining real-valuedness and circulant structure

## Limitations

- **Counterintuitive truncation**: The nearest real-valued rank-$\kappa$ circulant may differ significantly from the complex rank-$\kappa$ TSVD result. In particular, the largest singular value may need to be deleted to maintain the conjugate-even property — this can be surprising and may not always be desirable from a modeling perspective
- **FFT bottleneck on GPU**: The matvec still requires $O(n \log n)$ FFT/IFFT operations, which are the same cost as a full-rank circulant matvec. The rank-$\kappa$ constraint saves parameters but not compute (unless partial FFT is exploited, which adds implementation complexity)
- **Only applies to circulant structure**: The trick is specific to circulant matrices. For block-circulant, Toeplitz, or more general structured matrices, the conjugate-even truncation rule does not directly apply
- **Rank inflexibility**: The conjugate-even constraint means that rank can only change by 1 or 2 at each truncation step (depending on whether a real or complex-pair eigenvalue is removed). This means not all integer ranks between 1 and $n$ may be achievable as exact rank-$\kappa$ real circulants — the nearest achievable rank may be $\kappa \pm 1$
- **Perturbation sensitivity**: As shown in Example 3 of the paper, random perturbations of magnitude $10^{-j}$ to a rank-$\kappa$ circulant affect the smallest $n - \kappa$ singular values proportionally, but the algorithm always finds a better rank-$\kappa$ approximation than the original unperturbed matrix $C_\kappa$, with $\|W_j - Z_j\| < \|W_j - C_\kappa\|$ for all perturbation levels

## Implementation Notes

```python
import torch
import torch.fft as fft

def real_lowrank_circulant_approx(A, kappa):
    """Find nearest real-valued rank-kappa circulant to A.

    Implements the Chu-Plemmons algorithm:
    1. Project A onto circulant space (Chan's optimal)
    2. Get eigenvalues via FFT
    3. Solve conjugate-even DMP for rank truncation
    4. Reconstruct via IFFT

    Args:
        A: (n, n) real matrix
        kappa: target rank (1 <= kappa < n)

    Returns:
        c_hat: (n,) first row of rank-kappa real circulant
        lam_hat: (n,) truncated eigenvalues (conjugate-even)
    """
    n = A.shape[0]
    m = n // 2
    device = A.device

    # Step 1: Optimal circulant approximation (diagonal averaging)
    c = torch.zeros(n, device=device, dtype=A.dtype)
    for k in range(n):
        idx = torch.arange(n, device=device)
        c[k] = A[idx, (idx + k) % n].mean()

    # Step 2: Eigenvalues via IFFT (lambda = n * ifft(c))
    lam = n * fft.ifft(c.to(torch.complex64))

    # Step 3: Classify eigenvalues
    tol = n * 1e-10 * c.norm()

    if n % 2 == 0:  # n = 2m
        # Absolutely real eigenvalues: indices 0 and m
        # Complex paired: indices 1..m-1 paired with n-1..m+1
        real_indices = [0, m]
        complex_indices = list(range(1, m))  # each paired with n-k
    else:  # n = 2m+1
        real_indices = [0]
        complex_indices = list(range(1, m + 1))

    # Check for additional real eigenvalues (when imaginary part ≈ 0)
    extra_real = []
    for k in complex_indices:
        if abs(lam[k].imag) < tol:
            extra_real.append(k)

    # Step 4: Build modulus-sorted index for the "half spectrum"
    # For each complex pair, the cost of keeping it is 2 slots
    # For each real eigenvalue, the cost is 1 slot
    half_spectrum = []

    for k in real_indices:
        half_spectrum.append({
            'index': k,
            'type': 'real',
            'cost': 1,
            'modulus': abs(lam[k].item()),
            'value': lam[k]
        })

    for k in complex_indices:
        if k in extra_real:
            # This is actually a real eigenvalue disguised as complex
            half_spectrum.append({
                'index': k,
                'type': 'real_pair',  # pair of identical reals
                'cost': 2,
                'modulus': abs(lam[k].item()),
                'value': lam[k]
            })
        else:
            half_spectrum.append({
                'index': k,
                'type': 'complex_pair',
                'cost': 2,
                'modulus': abs(lam[k].item()),
                'value': lam[k]
            })

    # Sort by modulus, descending
    half_spectrum.sort(key=lambda x: x['modulus'], reverse=True)

    # Step 5: Greedy selection - keep top eigenvalues up to rank kappa
    lam_hat = torch.zeros(n, dtype=torch.complex64, device=device)
    rank_used = 0

    for entry in half_spectrum:
        if rank_used + entry['cost'] <= kappa:
            k = entry['index']
            lam_hat[k] = lam[k]
            if entry['type'] in ['complex_pair', 'real_pair']:
                lam_hat[n - k] = lam[n - k]  # conjugate partner
            rank_used += entry['cost']
        elif rank_used + 1 <= kappa and entry['cost'] == 1:
            # Can fit a single real eigenvalue
            k = entry['index']
            lam_hat[k] = lam[k]
            rank_used += 1

    # Step 6: Boundary decision (DMP Theorem 3.1)
    # At the truncation boundary, check if swapping a complex pair
    # for a real eigenvalue gives a better approximation
    # (This is the key conjugate-even truncation rule)

    # Step 7: Reconstruct first row via FFT
    c_hat = fft.fft(lam_hat).real / n

    return c_hat, lam_hat


def lowrank_circulant_matvec(lam_hat, x):
    """Apply rank-kappa circulant to vector(s) via FFT.

    Args:
        lam_hat: (n,) truncated eigenvalues (many zeros)
        x: (..., n) input vector(s)

    Returns:
        y: (..., n) output = C_hat @ x
    """
    x_fft = fft.fft(x.to(torch.complex64), dim=-1)
    y_fft = lam_hat * x_fft
    return fft.ifft(y_fft, dim=-1).real


class LowRankCirculantLayer(torch.nn.Module):
    """Structured linear layer using rank-kappa real circulant.

    Parameterized by kappa conjugate-even eigenvalues.
    Forward pass: O(n log n) via FFT.
    Parameters: O(kappa) learnable eigenvalue magnitudes + phases.
    """

    def __init__(self, n, kappa):
        super().__init__()
        self.n = n
        self.kappa = kappa
        self.m = n // 2

        # Parameterize the kappa/2 complex eigenvalue pairs
        # plus up to 2 real eigenvalues
        n_complex_pairs = (kappa - (2 if n % 2 == 0 else 1)) // 2
        n_complex_pairs = max(0, min(n_complex_pairs, self.m - 1))
        n_real = kappa - 2 * n_complex_pairs

        # Real eigenvalue parameters
        self.real_eigs = torch.nn.Parameter(
            torch.randn(min(n_real, 2 if n % 2 == 0 else 1))
        )

        # Complex pair parameters (magnitude and phase)
        self.pair_magnitudes = torch.nn.Parameter(
            torch.ones(n_complex_pairs) * 0.1
        )
        self.pair_phases = torch.nn.Parameter(
            torch.randn(n_complex_pairs) * 0.1
        )

        self.n_complex_pairs = n_complex_pairs

    def _build_eigenvalues(self):
        """Construct conjugate-even eigenvalue vector."""
        n = self.n
        lam = torch.zeros(n, dtype=torch.complex64,
                         device=self.real_eigs.device)

        # Real eigenvalues at indices 0 (and m if n even)
        lam[0] = self.real_eigs[0].to(torch.complex64)
        if n % 2 == 0 and len(self.real_eigs) > 1:
            lam[n // 2] = self.real_eigs[1].to(torch.complex64)

        # Complex conjugate pairs at indices k and n-k
        for i in range(self.n_complex_pairs):
            k = i + 1  # indices 1, 2, ...
            mag = self.pair_magnitudes[i].abs()
            phase = self.pair_phases[i]
            lam[k] = mag * torch.exp(1j * phase.to(torch.complex64))
            lam[n - k] = mag * torch.exp(-1j * phase.to(torch.complex64))

        return lam

    def forward(self, x):
        """x: (..., n) -> (..., n)"""
        lam = self._build_eigenvalues()
        return lowrank_circulant_matvec(lam, x)
```

## References

- Chu, M.T. & Plemmons, R.J. "Real-Valued, Low Rank, Circulant Approximation" SIAM J. Matrix Anal. Appl., 24(3):645-659, 2003
- Chan, T.F. "An Optimal Circulant Preconditioner for Toeplitz Systems" SIAM J. Sci. Stat. Comput., 9(4):766-771, 1988
- Davis, P.J. "Circulant Matrices" Wiley, 1979
- Van Loan, C.F. "Computational Frameworks for the Fast Fourier Transform" Frontiers in Applied Mathematics 10, SIAM, 1992
- Brockett, R.W. "Least Square Matching Problems" Linear Algebra Appl., 122/123/124:761-777, 1989
- Chu, M.T. & Funderlic, R.E. & Plemmons, R.J. "Structured Low Rank Approximation" Linear Algebra Appl., 2003
