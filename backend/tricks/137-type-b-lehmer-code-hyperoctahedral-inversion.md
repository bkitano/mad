# 137: Type-B Lehmer Code (Hyperoctahedral Inversion Table)

**Category**: algebraic
**Gain type**: efficiency
**Source**: Raharinirina (2016/2022); Reiner (1993); Laisant (1888)
**Paper**: [papers/hyperoctahedral-enumeration-inversion.pdf] (arXiv:1607.08889)
**Documented**: 2026-02-15

## Description

The **type-B Lehmer code** (hyperoctahedral inversion table) is a bijective encoding of signed permutations $\pi \in B_n$ as sequences of bounded integers, generalizing the classical Lehmer code for unsigned permutations. Each signed permutation $\pi$ is uniquely represented by an $n$-tuple $(\text{inv}_1 \pi : \text{inv}_2 \pi : \cdots : \text{inv}_n \pi)$ where $\text{inv}_i \pi$ counts the number of "type-B inversions" at position $i$. This encoding establishes a bijection between $B_n$ (the hyperoctahedral group of order $2^n n!$) and numbers in the **hyperoctahedral number system** — a mixed-radix positional system with bases $(2, 4, 6, \ldots, 2n)$ corresponding to the factors $B_i = 2^i i!$ of the group order.

The key computational trick is that this encoding enables:
1. **$O(n^2)$ rank computation:** Given a signed permutation, compute its rank (position in lexicographic order) without enumerating all $2^n n!$ elements
2. **$O(n^2)$ unranking:** Given a rank $k$, directly construct the $k$-th signed permutation
3. **$O(n)$ Coxeter length computation:** The sum $\sum_i \text{inv}_i \pi$ gives the length of $\pi$ in the Coxeter group $B_n$ (number of generators in a reduced expression)
4. **Separation of sign and permutation components:** The encoding naturally decomposes into a sign vector $\varepsilon \in \{+1, -1\}^n$ and a classical Lehmer code $m \in \prod_{i=0}^{n-1} \{0, \ldots, i\}$

This is relevant to neural network architectures because: (a) it provides an efficient discrete parameterization of the hyperoctahedral group, complementing continuous relaxations like Sinkhorn or OT4P; (b) the Coxeter length (total inversions) gives a natural distance metric on $B_n$ computable in $O(n)$; (c) the mixed-radix encoding can serve as a compact integer representation for signed permutation states in SSMs and automata simulation.

## Mathematical Form

**Type-B Root System and Inversions:**

The hyperoctahedral group $B_n$ is the Coxeter group with root system:

$$
\Phi_n = \{\pm e_i, \pm e_i \pm e_j \mid 1 \leq i \neq j \leq n\}
$$

with positive roots:

$$
\Phi_n^+ = \{e_k, e_i + e_j, e_i - e_j \mid k \in [n], 1 \leq i < j \leq n\}
$$

For each position $i \in [n]$, define the local positive root subset:

$$
\Phi_{n,i}^+ = \{e_i, e_i + e_j, e_i - e_j \mid i < j \leq n\}
$$

**Core Definition — $i$-inversions:**

For $\pi \in B_n$ and $i \in [n]$, the number of $i$-inversions is:

$$
\text{inv}_i \pi = \#\{v \in \Phi_{n,i}^+ \mid \pi^{-1}(v) \in -\Phi_n^+\}
$$

**Explicit Formula (Lemma 3.1):**

For $i \in [n]$ and $\pi \in B_n$:

**(i)** If $\pi(i) = j > 0$ (positive image):

$$
\text{inv}_i \pi = \#\{k \in \{i+1, \ldots, n\} \mid j > |\pi(k)|\}
$$

**(ii)** If $\pi(i) = -j < 0$ (negative image):

$$
\text{inv}_i \pi = 1 + \#\{k \in \{i+1, \ldots, n\} \mid j > |\pi(k)|\} + 2 \cdot \#\{k \in \{i+1, \ldots, n\} \mid j < |\pi(k)|\}
$$

**Range of each digit:**

$$
\text{inv}_i \pi \in \{0, 1, \ldots, 2(n-i)+1\}
$$

So specifically:
- $\text{inv}_1 \pi \in \{0, 1, \ldots, 2n-1\}$
- $\text{inv}_2 \pi \in \{0, 1, \ldots, 2(n-1)+1\}$
- $\vdots$
- $\text{inv}_n \pi \in \{0, 1\}$

**Hyperoctahedral Number System:**

Every non-negative integer $N$ has a unique representation:

$$
N = \sum_{i=0}^{k} d_i \cdot B_i, \quad \text{where } B_i = 2^i \cdot i!, \quad d_i \in \{0, 1, 2, \ldots, 2i+1\}
$$

The basis sequence is $\mathcal{B} = (B_0, B_1, B_2, B_3, \ldots) = (1, 2, 8, 48, 384, \ldots)$.

**Bijection (Signed Permutation $\leftrightarrow$ Hyperoctahedral Number):**

The type-B Lehmer code establishes:

$$
\pi \in B_n \quad \longleftrightarrow \quad (\text{inv}_1 \pi : \text{inv}_2 \pi : \cdots : \text{inv}_n \pi) \quad \longleftrightarrow \quad \text{rank}(\pi) - 1 = \sum_{i=1}^{n} \text{inv}_i \pi \cdot B_{n-i}
$$

where $B_{n-i} = 2^{n-i}(n-i)!$.

**Decomposition into Sign + Permutation:**

Given the hyperoctahedral code $(\gamma_{n-1} : \cdots : \gamma_0)$, apply the mapping $\mathcal{M}_\ell$ for each digit:

$$
\mathcal{M}_\ell(\gamma) = \begin{cases} (\gamma, +1) & \text{if } \gamma \leq \ell \\ (1 + 2\ell - \gamma, -1) & \text{if } \gamma > \ell \end{cases}
$$

This separates each digit into a sign component $\varepsilon_i \in \{+1, -1\}$ and a classical inversion count $m_i \in \{0, 1, \ldots, i\}$. The sign vector $\varepsilon = (\varepsilon_1, \ldots, \varepsilon_n)$ gives the diagonal part $D$ and the classical Lehmer code $(m_{n-1}, \ldots, m_0)$ gives the permutation $\sigma$ of the wreath product decomposition $\pi = D \cdot P_\sigma$.

**Coxeter Length:**

The Coxeter length (minimum number of generators to express $\pi$) equals the total inversions:

$$
\ell(\pi) = \sum_{i=1}^{n} \text{inv}_i \pi
$$

## Complexity

| Operation | Naive | With Type-B Lehmer Code |
|-----------|-------|------------------------|
| Rank $\pi$ (position in lex order) | $O(2^n n!)$ enumerate | $O(n^2)$ compute inversions + $O(n)$ mixed-radix |
| Unrank $k$ (find $k$-th permutation) | $O(2^n n!)$ enumerate | $O(n)$ mixed-radix decode + $O(n^2)$ reconstruct |
| Coxeter length $\ell(\pi)$ | $O(n \cdot |\text{generators}|)$ BFS | $O(n^2)$ sum of inversions |
| Distance $d(\pi, \tau) = \ell(\pi \tau^{-1})$ | $O(n^2)$ compose + length | $O(n^2)$ |
| Storage of $\pi$ | $O(n)$ (index + sign arrays) | $O(n)$ (digit sequence) |
| Encoding $\pi \to$ code | — | $O(n^2)$ |
| Decoding code $\to \pi$ | — | $O(n^2)$ |

**Memory:** $O(n)$ for the Lehmer code (one integer per position, each bounded by $2n$).

## Applicability

- **Compact enumeration for search/sampling:** When exploring signed permutation spaces (e.g., for state-tracking in SSMs, combinatorial optimization), the type-B Lehmer code enables efficient uniform random sampling and systematic enumeration without materializing the full group
- **Coxeter length as distance/loss:** The total inversion count $\ell(\pi) = \sum \text{inv}_i \pi$ provides an efficiently computable, natural metric on $B_n$ for regularization or loss functions when learning signed permutation structures
- **Mixed-radix representation for learned embeddings:** The digit sequence $(\text{inv}_1, \ldots, \text{inv}_n)$ provides a natural factored representation of signed permutations, where each digit independently encodes a local structural property — useful for embedding layers in neural architectures
- **Parallel scan state spaces:** For SSMs with monomial/signed-permutation transition matrices, the Lehmer code can represent the cumulative state transformation compactly, enabling efficient serialization and comparison of scan states
- **Genome rearrangement distance:** Combined with the breakpoint graph framework, the type-B inversion count provides bounds on the reversal distance, connecting to the signed-reversal-sorting trick
- **Differentiable relaxation design:** The mixed-radix structure suggests natural continuous relaxations — each digit can be independently softened, potentially yielding better-behaved gradients than global Sinkhorn relaxation

## Limitations

- The encoding is inherently discrete; computing gradients through the Lehmer code requires Straight-Through Estimators or Gumbel-Softmax-style relaxations
- The $O(n^2)$ encoding/decoding cost comes from computing inversions (analogous to counting inversions in merge sort); this is optimal for comparison-based methods but could be reduced to $O(n \log n)$ with merge-sort-based counting
- The mixed-radix system has variable-width digits ($d_i \in \{0, \ldots, 2i+1\}$), which complicates fixed-width tensor representations compared to uniform binary or factorial representations
- The Coxeter length (total inversions) is just one possible distance metric on $B_n$; it corresponds to the word metric with respect to the Coxeter generators, not reversal distance or other biologically/computationally motivated distances
- For large $n$, the group order $2^n n!$ grows superexponentially, so even compact enumeration is impractical — the encoding's main value is in computing properties of individual elements efficiently

## Implementation Notes

```python
import numpy as np

def signed_perm_to_lehmer_B(pi):
    """
    Compute the type-B Lehmer code (inversion table) of a signed permutation.

    Args:
        pi: list of signed integers, a permutation of [±1, ..., ±n]
             e.g., [1, -3, 4, 2] means π(1)=1, π(2)=-3, π(3)=4, π(4)=2

    Returns:
        code: list of n integers, the type-B Lehmer code
               inv_i(π) ∈ {0, 1, ..., 2(n-i)+1}
    """
    n = len(pi)
    code = []
    for i in range(n):
        val = pi[i]
        abs_val = abs(val)
        sign = 1 if val > 0 else -1

        if sign > 0:
            # Case (i): π(i) = j > 0
            # Count k > i where |π(k)| < j
            inv_i = sum(1 for k in range(i + 1, n) if abs(pi[k]) < abs_val)
        else:
            # Case (ii): π(i) = -j < 0
            # 1 + #{k > i : |π(k)| < j} + 2·#{k > i : |π(k)| > j}
            less_count = sum(1 for k in range(i + 1, n) if abs(pi[k]) < abs_val)
            greater_count = sum(1 for k in range(i + 1, n) if abs(pi[k]) > abs_val)
            inv_i = 1 + less_count + 2 * greater_count

        code.append(inv_i)
    return code


def lehmer_B_to_signed_perm(code):
    """
    Reconstruct a signed permutation from its type-B Lehmer code.

    Args:
        code: list of n integers (the type-B Lehmer code)

    Returns:
        pi: list of signed integers (the signed permutation)
    """
    n = len(code)

    # Step 1: Separate into signs and classical Lehmer code
    signs = []
    classical_code = []
    for i, gamma in enumerate(code):
        ell = n - 1 - i  # available positions for classical part
        if gamma <= ell:
            signs.append(+1)
            classical_code.append(gamma)
        else:
            signs.append(-1)
            classical_code.append(1 + 2 * ell - gamma)
            # Note: the formula maps γ > ℓ to m = 1 + 2ℓ - γ ∈ {0,...,ℓ}

    # Step 2: Reconstruct permutation from classical Lehmer code
    # r_i = 1 + m_{n-1-i} gives the rank in the remaining list
    available = list(range(1, n + 1))
    perm = []
    for i in range(n):
        rank = classical_code[i]  # 0-indexed position in remaining elements
        # rank-th smallest among remaining
        chosen = available[rank]
        perm.append(chosen)
        available.remove(chosen)

    # Step 3: Apply signs
    pi = [signs[i] * perm[i] for i in range(n)]
    return pi


def rank_signed_perm(pi):
    """
    Compute the rank (1-indexed position in lex order) of a signed permutation.

    Uses the hyperoctahedral number system: rank = 1 + Σ inv_i · B_{n-i}
    where B_k = 2^k · k!
    """
    n = len(pi)
    code = signed_perm_to_lehmer_B(pi)

    # Compute B_k = 2^k * k!
    def B(k):
        result = 1
        for j in range(1, k + 1):
            result *= 2 * j
        return result

    rank = 1
    for i in range(n):
        rank += code[i] * B(n - 1 - i)
    return rank


def unrank_signed_perm(rank, n):
    """
    Find the signed permutation at position 'rank' (1-indexed) in B_n.

    Uses the hyperoctahedral number system to decode.
    """
    N = rank - 1

    # Convert to hyperoctahedral number system
    code = []
    for i in range(n):
        base = 2 * (n - i)  # divisor at position i
        d_i = N % base
        N = N // base
        code.append(d_i)

    code.reverse()  # code[0] = inv_1, ..., code[n-1] = inv_n
    return lehmer_B_to_signed_perm(code)


def coxeter_length(pi):
    """
    Compute Coxeter length of signed permutation π in type B.
    Equal to the sum of i-inversions = sum of Lehmer code digits.
    """
    return sum(signed_perm_to_lehmer_B(pi))


# Example usage:
# π = (1, -3, 4, 2) from Example 4 in the paper
pi = [1, -3, 4, 2]
code = signed_perm_to_lehmer_B(pi)
print(f"π = {pi}")
print(f"Type-B Lehmer code: {code}")  # Should be [0, 4, 1, 0]
print(f"Rank: {rank_signed_perm(pi)}")  # Should be 35
print(f"Coxeter length: {coxeter_length(pi)}")  # = 0+4+1+0 = 5
print(f"Reconstructed: {lehmer_B_to_signed_perm(code)}")  # Should be [1, -3, 4, 2]
```

## References

- Raharinirina, I.V. (2016). On Hyperoctahedral Enumeration System, Application to Signed Permutations. arXiv:1607.08889.
- Reiner, V. (1993). Signed Permutation Statistics. *European J. Combinatorics*, 14(6):553–567.
- Laisant, C.-A. (1888). Sur la numération factorielle, application aux permutations. *Bull. Soc. Math. France*, 16:176–183.
- Björner, A. & Brenti, F. (2005). *Combinatorics of Coxeter Groups*. Springer GTM 231.
- Wikipedia: Lehmer code. https://en.wikipedia.org/wiki/Lehmer_code
- Wikipedia: Hyperoctahedral group. https://en.wikipedia.org/wiki/Hyperoctahedral_group
