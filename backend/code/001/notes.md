# Proposal 001: CS-DeltaNet vs. Kimi Delta Attention (KDA)

## Overview

This document compares our CS-DeltaNet (Column-Sparse DeltaNet with input-dependent permutation routing) against Kimi Delta Attention (KDA) from the Kimi Linear paper (arXiv:2510.26692).

---

## Core Update Rules

### CS-DeltaNet (our proposal)

$$\mathbf{S}_t = P(\mathbf{x}_t) \cdot \mathbf{S}_{t-1} + \beta_t \cdot \mathbf{k}_t \otimes (\mathbf{v}_t - \mathbf{S}_{t-1}^\top \mathbf{k}_t)$$

where $P(\mathbf{x}_t)$ is an **input-dependent permutation matrix** (learned via Gumbel-softmax)

### KDA (Kimi)

$$\mathbf{S}_t = (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \cdot \text{Diag}(\boldsymbol{\alpha}_t) \cdot \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top$$

where $\text{Diag}(\boldsymbol{\alpha}_t)$ is a **channel-wise decay** (each dimension has independent forgetting rate)

---

## What Each Does to the State

| Operation | CS-DeltaNet | KDA |
|-----------|-------------|-----|
| **State transformation** | Permutation $P(\mathbf{x}_t)$ - reorders/routes dimensions | Diagonal decay $\text{Diag}(\boldsymbol{\alpha}_t)$ - scales dimensions |
| **Eigenvalue structure** | Permutation matrices have eigenvalues on unit circle ($|\lambda|=1$) | Diagonal has eigenvalues in $(0,1)$ per channel |
| **Information flow** | Routes information between dimensions | Selectively forgets per dimension |
| **Determinant** | $\det(P) = \pm 1$ (volume preserving) | $\det(\text{Diag}(\boldsymbol{\alpha})) < 1$ (contractive) |

---

## Theoretical Expressivity

### CS-DeltaNet's Permutation $P(\mathbf{x}_t)$
- Permutation matrices form the symmetric group $S_n$
- Can represent any reordering of state dimensions
- **Key capability**: routing/copying information between memory slots
- Eigenvalues lie on the unit circle $\rightarrow$ oscillatory dynamics, no decay
- Combined with $\beta \in (0,2)$: can do sign flips AND permutation routing

### KDA's Diagonal Decay $\text{Diag}(\boldsymbol{\alpha}_t)$
- Diagonal matrices are a commutative group
- Can only scale dimensions independently
- **Key capability**: selective forgetting, learnable positional bias
- Eigenvalues in $(0,1)$ $\rightarrow$ contractive, information eventually lost
- No routing between dimensions

---

## The Fundamental Difference

$$\text{CS-DeltaNet: State dimensions can \textbf{talk to each other} via permutation}$$
$$\text{KDA: State dimensions are \textbf{isolated}, only scaled independently}$$

This is a huge distinction:
- **CS-DeltaNet** can move the content of slot $i$ to slot $j$ $\rightarrow$ enables group operations like $D_4$
- **KDA** cannot mix slots, only decay them at different rates $\rightarrow$ better for "forgetting old information gracefully"

---

## Advantages & Disadvantages

| Aspect | CS-DeltaNet | KDA |
|--------|-------------|-----|
| **Group state tracking** | Excellent (permutation = group action) | Poor (diagonal can't permute) |
| **Long-context LM** | Unclear (no forgetting mechanism) | Excellent (selective forgetting) |
| **Hardware efficiency** | Expensive (full permutation matrix per step) | Efficient (diagonal = elementwise multiply) |
| **Parallelization** | Hard (permutation depends on previous state) | Has efficient chunkwise algorithm |
| **Positional encoding** | Not addressed | Replaces RoPE entirely |
| **Parameter cost** | $O(d^2)$ per step for permutation logits | $O(d)$ for decay vector |

---

## CS-DeltaNet Permutation Cost Analysis

### Why It's Expensive (per token, per head, with head_dim $= d$)

| Operation | Cost | Notes |
|-----------|------|-------|
| Permutation logits | $O(d^2)$ params in projection | $d_{\text{model}} \to d \times d$ |
| Gumbel-softmax | $O(d^2)$ compute | $d$ columns $\times$ $d$-way softmax |
| Apply $P \cdot S$ | $O(d^2)$ compute | dense matrix multiply |
| **Parallelization** | **Sequential** | $P(\mathbf{x}_t)$ depends on $\mathbf{x}_t$, can't precompute |

For KDA's typical $d=128$: that's **16,384 logits per head per token** just for the permutation.

---

## Approximation Tricks for Efficient Permutation

### 1. Permutation Vocabulary (MoE-style)

Instead of learning arbitrary $P(\mathbf{x}_t)$, have $K$ fixed/learned permutation matrices and select among them:

```python
# K permutation matrices (learned or fixed)
P_vocab = [P_1, P_2, ..., P_K]  # K × d × d

# Selection logits from input
selector = softmax(W_select @ x_t)  # K-dimensional

# Soft selection (or hard via Gumbel)
P_t = sum(selector[i] * P_vocab[i] for i in range(K))
```

**Cost**: $O(K)$ selection + $O(d^2)$ application, but $K$ can be small (e.g., $K=8$ for $D_4$)

**For $D_4$**: You literally only need 8 permutations! This is perfect.

---

### 2. Factored Permutation (Butterfly/Sorting Networks)

Any permutation can be written as a product of $O(\log d)$ layers of simple swaps:

```python
# Butterfly structure: log(d) layers
# Each layer has d/2 independent 2x2 blocks
def butterfly_permutation(x_t, d):
    P = I
    for layer in range(log2(d)):
        # Each layer: pairs of dimensions can swap or not
        swap_logits = W_layer @ x_t  # d/2 logits
        swap_probs = sigmoid(swap_logits)

        # Build block-diagonal of 2x2 swap matrices
        P_layer = build_butterfly_layer(swap_probs, layer)
        P = P_layer @ P
    return P
```

**Cost**: $O(d \log d)$ parameters, $O(d \log d)$ compute

**Key insight**: This is related to FFT structure - highly optimized on hardware.

---

### 3. Block-Diagonal Permutation

Only allow permutation within blocks of size $b$:

$$P = \begin{bmatrix} P_1 & 0 & 0 \\ 0 & P_2 & 0 \\ 0 & 0 & P_3 \end{bmatrix} \quad \text{where each } P_i \text{ is } b \times b$$

**Cost**: $O(d/b \times b^2) = O(db)$ instead of $O(d^2)$

**Tradeoff**: Can't route between blocks. But if your task has natural block structure (e.g., separate "rotation" and "reflection" coordinates for $D_4$), this might be fine.

---

### 4. Householder Product (Most Elegant)

**Key connection**: DeltaNet already uses Householder-like structure!

The delta update $(\mathbf{I} - \beta \mathbf{k}\mathbf{k}^\top)$ is a Householder reflection when $\beta=2$ and $\|\mathbf{k}\|=1$.

Any orthogonal matrix (including permutations) = product of at most $d$ Householder reflections.

```python
# m Householder reflections (m << d for approximation)
def householder_permutation(x_t, m=4):
    P = I
    for i in range(m):
        u_i = L2_normalize(W_i @ x_t)  # reflection vector
        H_i = I - 2 * outer(u_i, u_i)  # Householder reflection
        P = H_i @ P
    return P
```

**Cost**: $O(md)$ parameters, $O(md)$ compute per reflection, $O(md^2)$ total for $P \cdot S$

**Key insight**: With $m = O(\log d)$, you can approximate any permutation. And this structure is already what DeltaNet uses internally!

---

### 5. Implicit Permutation via Index Routing

Instead of permuting the state matrix, permute the **read/write indices**:

```python
# Learn read and write permutations
read_idx = gumbel_softmax_permutation(W_read @ x_t)   # which slot to read from
write_idx = gumbel_softmax_permutation(W_write @ x_t) # which slot to write to

# Read from permuted indices
retrieved = S[read_idx] @ k_t

# Write to permuted indices
S[write_idx] += beta * outer(k_t, delta)
```

**Advantage**: Permutation and state update decouple - potentially more parallelizable.

---

### 6. For $D_4$ Specifically: Exploit Group Structure (RECOMMENDED)

$D_4$ has only **8 elements**. The state can be encoded in 3 bits.

**Trick**: Use a **direct group representation**

```python
class D4PermutationRouting(nn.Module):
    def __init__(self, d_model, state_dim=8):
        super().__init__()
        # Cayley table for D4 as permutation matrices
        self.register_buffer('cayley_perms', self._build_cayley_table())  # 8 × 8 × 8
        self.selector = nn.Linear(d_model, 8)

    def _build_cayley_table(self):
        # D4 Cayley table as permutation matrices
        # cayley_perms[g] is the permutation induced by left-mult by g
        ...

    def forward(self, x_t, S):
        # Select group element
        g_logits = self.selector(x_t)
        g_probs = F.softmax(g_logits, dim=-1)  # or Gumbel-softmax

        # Apply weighted permutation
        P = torch.einsum('bg,gij->bij', g_probs, self.cayley_perms)

        return P @ S
```

This is:
- **Exact** for $D_4$ (not an approximation)
- $O(8)$ selection cost
- $O(64)$ application cost
- **Interpretable**: you can see which group element is being applied

---

## Recommendations

### For $D_4$ State Tracking
Use **Option 6 (Group Structure)** or **Option 1 (Permutation Vocabulary)**:
- The permutation vocabulary IS the Cayley table
- Only 8 permutations needed
- Trivial computational cost

### For General Permutation Routing at Scale
Use **Option 2 (Butterfly Factorization)**:
- $O(d \log d)$ parameters and compute
- Can approximate any permutation
- Hardware-friendly structure (related to FFT)

### For Combining with KDA
A hybrid could look like:

$$\mathbf{S}_t = (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \cdot P(\mathbf{x}_t) \cdot \text{Diag}(\boldsymbol{\alpha}_t) \cdot \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top$$

This would give:
1. **Permutation routing** (from CS-DeltaNet) - move info between slots
2. **Selective decay** (from KDA) - forget old info per-channel
3. **Delta correction** (shared) - error-correcting updates

**Challenge**: Permutation + diagonal don't commute, and this breaks KDA's efficient chunkwise algorithm.

---

## Key Takeaway

**CS-DeltaNet and KDA solve different problems:**

| | CS-DeltaNet | KDA |
|--|-------------|-----|
| **Problem** | How to route information between memory slots? | How to selectively forget old information? |
| **Solution** | Input-dependent permutation | Per-channel learnable decay |
| **Best for** | Algebraic/group structure tasks | Language modeling, long-context |
| **Weakness** | No forgetting, expensive | Cannot permute/route dimensions |

The Kimi paper doesn't address state tracking - they're optimizing for LM quality + long context. Our CS-DeltaNet addresses a theoretical capability that KDA fundamentally lacks: **inter-dimensional routing**.

---

## Critical Update: DeltaProduct Makes CS-DeltaNet Redundant

**Key insight**: DeltaNet's state transition matrix *already* accumulates Householder reflections over time!

### The "Slow Permutation" Observation

Each DeltaNet update applies $(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$, which is a **generalized Householder reflection** when $\beta \in (0, 2)$ and $\|\mathbf{k}\| = 1$.

After $T$ tokens, the state transition matrix from $t=0$ to $t=T$ is:

$$A_{0 \to T} = \prod_{t=1}^{T} (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$$

By the **Cartan-Dieudonné theorem**: any orthogonal matrix (including any permutation!) is a product of at most $d$ Householder reflections.

**Implication**: DeltaNet is already building permutations "slowly" - one Householder per token.

---

### DeltaProduct: The Formalization

The **DeltaProduct** paper (Schlag et al., ICLR 2025, arXiv:2502.10297) formalizes this exactly:

Instead of 1 Householder step per token, take $n_h$ steps:

$$A(\mathbf{x}_i) = \prod_{j=1}^{n_h} \left( \mathbf{I} - \beta_{i,j} \, \mathbf{k}_{i,j} \mathbf{k}_{i,j}^\top \right)$$

**Key theorem**: With $n_h$ Householder reflections, you can simulate permutations of up to $(n_h + 1)$ elements in a single layer.

| Symmetric Group | Min $n_h$ Required |
|-----------------|-------------------|
| $S_3$ (3 elements) | $n_h = 2$ |
| $S_4$ (4 elements) | $n_h = 2$ (exploits $S_4 \hookrightarrow SO(3)$) |
| $S_5$ (5 elements) | $n_h = 4$ |
| $S_n$ general | $n_h = n - 1$ |

---

### Why This Makes CS-DeltaNet Over-Engineered

| Aspect | CS-DeltaNet | DeltaProduct |
|--------|-------------|--------------|
| **Mechanism** | Explicit permutation $P(\mathbf{x}_t)$ via Gumbel-softmax | Implicit permutation via $n_h$ Householder products |
| **Parameter cost** | $O(d^2)$ for permutation logits | $O(d \cdot n_h)$ for $n_h$ key vectors |
| **Compute cost** | $O(d^2)$ matrix multiply | $O(d \cdot n_h)$ per token |
| **Permutation coverage** | Any permutation (but expensive) | $S_{n_h+1}$ (sufficient for small groups) |
| **Learning** | Must learn discrete permutation structure | Gradient-friendly, continuous |
| **For $D_4$** | Overkill: learning $8 \times 8$ permutation logits | $n_h = 2$ suffices (since $D_4 \subset O(2)$) |

**For $D_4$ specifically**: $D_4$ is a subgroup of $O(2)$ (rotations and reflections in 2D). DeltaProduct with $n_h = 2$ Householder reflections can represent any element of $O(2)$, hence any element of $D_4$.

---

### Conclusion: Use DeltaProduct Instead

CS-DeltaNet's explicit permutation matrix is:
1. **Expensive**: $O(d^2)$ vs $O(d \cdot n_h)$
2. **Redundant**: DeltaNet already builds orthogonal matrices via Householder products
3. **Harder to learn**: Discrete Gumbel-softmax vs continuous gradients

**Recommendation**: For state-tracking tasks, use **NEG-DeltaNet** (to enable true reflections with $\beta \in (0,2)$) with the **DeltaProduct** extension ($n_h > 1$ steps per token).

For $D_4$: NEG-DeltaProduct with $n_h = 2$ is provably sufficient and far more efficient than CS-DeltaNet.

---

### Caveat: DeltaProduct's Parallelization Cost

**Important tradeoff**: DeltaProduct does NOT have a clean parallel form like standard DeltaNet.

**Standard DeltaNet** parallelizes well:
1. Each token contributes a single rank-1 update: $S_t = S_{t-1} + \beta_t k_t \delta_t^\top$
2. WY representation: $S_t = I + W_t Y_t^\top$ (just append columns)
3. Chunkwise scan: compute within chunk assuming boundary = 0, then propagate corrections
4. Maps cleanly to matmuls/tensor cores

**DeltaProduct** complicates this:
1. Each token does $n_h$ **sequential** Householder reflections
2. The product can be written as $A = I + WR$ (identity + rank-$n_h$ perturbation)
3. But computing the WY form requires sequential processing of the $n_h$ reflections
4. State transition per token is a dense-ish matrix, not a simple rank-1 outer product

**Parallelization hierarchy:**

| Model | State Transition | Parallel Scan Type | Hardware Efficiency |
|-------|-----------------|-------------------|---------------------|
| Mamba/S4D | Diagonal $\text{Diag}(\alpha_t)$ | Element-wise parallel scan | **Best** (trivially parallel) |
| KDA | Diagonal + Householder | Element-wise + rank-1 | Good |
| DeltaNet | Rank-1 update | Chunkwise with WY | Good (matmul-friendly) |
| DeltaProduct | Rank-$n_h$ product | Chunkwise with WY (complex) | **Worse** (sequential $n_h$ steps/token) |
| CS-DeltaNet | Full permutation | ??? | **Worst** ($O(d^2)$ per token) |

From the DeltaProduct notes:
> "The WY representation is needed for efficient chunkwise parallel training"
> "Unlike diagonal state-transition matrices, does not admit element-wise parallel scans"

**The fundamental tradeoff**: Expressivity (permutation/state-tracking capability) vs parallelization efficiency.

- DeltaProduct training cost scales **linearly with $n_h$**
- For $D_4$ with $n_h = 2$: ~2× the cost of standard DeltaNet per token
- Still much better than CS-DeltaNet's $O(d^2)$, but worse than diagonal models

**Open question**: Is the state-tracking capability worth the parallelization hit for practical applications?
