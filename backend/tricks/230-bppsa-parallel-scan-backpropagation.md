# 230: BPPSA — Parallel Scan Backpropagation

**Category**: parallelization
**Gain type**: efficiency
**Source**: Wang, Bai & Pekhimenko (2020). BPPSA: Scaling Back-propagation by Parallel Scan Algorithm. MLSys 2020.
**Paper**: papers/bppsa-parallel-scan-backpropagation.pdf
**Documented**: 2026-02-15

## Description

BPPSA reformulates the backpropagation algorithm as a parallel scan (prefix sum) over transposed Jacobian matrices, reducing the gradient computation from $\Theta(n)$ sequential steps to $\Theta(\log n)$ parallel steps under model parallelism. The key insight is that the backward pass through $n$ layers — which chains transposed Jacobian-vector products — is mathematically equivalent to an exclusive scan with a non-commutative associative binary operator defined as matrix-matrix (or matrix-vector) multiplication. By applying a modified Blelloch scan algorithm, the gradient computation across layers can be parallelized across multiple workers (GPU SMs, devices, etc.), breaking the strict sequential dependency of standard backpropagation.

A critical practical contribution is the observation that Jacobians of common neural network operators (convolution, ReLU, max-pooling) are extremely sparse — often >99.9% zeros — and that this sparsity pattern is deterministic (known from the architecture before training). This enables efficient sparse matrix representation (CSR format) that dramatically reduces the per-step complexity of the scan operator.

## Mathematical Form

**Standard Backpropagation (Sequential):**

Given a network $f = f_n \circ f_{n-1} \circ \cdots \circ f_1$ with layer outputs $\vec{x}_i = f_i(\vec{x}_{i-1}; \vec{\theta}_i)$ and loss $l$, the gradient at layer $i$ is:

$$
\nabla_{\vec{x}_i} l \leftarrow \left(\frac{\partial \vec{x}_{i+1}}{\partial \vec{x}_i}\right)^T \nabla_{\vec{x}_{i+1}} l, \quad \forall i \in \{n-1, \ldots, 1\}
$$

This imposes a strong sequential dependency: $\nabla_{\vec{x}_i} l$ cannot be computed until $\nabla_{\vec{x}_{i+1}} l$ is available.

**Reformulation as Scan:**

Define a binary, associative, non-commutative operator $\diamond$ where $A \diamond B = BA$ (i.e., $A$ can be a matrix or vector, $B$ is always a matrix, and the identity is $I$). The full backward pass is:

$$
[\nabla_{\vec{x}_n} l, \; \nabla_{\vec{x}_n} l \diamond \left(\frac{\partial \vec{x}_n}{\partial \vec{x}_{n-1}}\right)^T, \; \nabla_{\vec{x}_n} l \diamond \left(\frac{\partial \vec{x}_n}{\partial \vec{x}_{n-1}}\right)^T \diamond \left(\frac{\partial \vec{x}_{n-1}}{\partial \vec{x}_{n-2}}\right)^T, \; \ldots]
$$

This is exactly the **exclusive scan** of $\diamond$ on the input array:

$$
\left[\nabla_{\vec{x}_n} l, \; \left(\frac{\partial \vec{x}_n}{\partial \vec{x}_{n-1}}\right)^T, \; \left(\frac{\partial \vec{x}_{n-1}}{\partial \vec{x}_{n-2}}\right)^T, \; \ldots, \; \left(\frac{\partial \vec{x}_2}{\partial \vec{x}_1}\right)^T, \; \left(\frac{\partial \vec{x}_1}{\partial \vec{x}_0}\right)^T\right]
$$

**Modified Blelloch Scan for Non-Commutative $\diamond$:**

- **Up-sweep (reduce):** $a[r] \leftarrow a[l] \diamond a[r]$ (standard order)
- **Down-sweep (distribute):** $a[l] \leftarrow a[r]$, then $a[r] \leftarrow a[r] \diamond T$ (reversed operand order vs. standard Blelloch, because $\diamond$ is non-commutative)

where $T$ is the saved left child value. The reversal in the down-sweep is critical for correctness with non-commutative operators.

**Parameter Gradients:**

Once all $\nabla_{\vec{x}_i} l$ are computed via the scan, parameter gradients are computed independently in parallel (no inter-layer dependency):

$$
\nabla_{\vec{\theta}_i} l = \left(\frac{\partial \vec{x}_i}{\partial \vec{\theta}_i}\right)^T \nabla_{\vec{x}_i} l
$$

## Complexity

| Metric | Standard BP | BPPSA (Blelloch Scan) |
|--------|-----------|----------------------|
| Step complexity (parallel depth) | $\Theta(n)$ | $\Theta(\log n)$ when $p > n$; $\Theta(n/p + \log p)$ otherwise |
| Work complexity (total ops) | $\Theta(n)$ | $\Theta(n)$ |
| Per-step cost | $P_{\text{Linear}}$ (dense mat-vec) | $P_{\text{Blelloch}}$ (sparse mat-mat) |
| Total runtime | $\Theta(n) \cdot P_{\text{Linear}}$ | $\Theta(\log n) \cdot P_{\text{Blelloch}}$ |

**Memory:** $O(\max(n/p, 1) \cdot M_{\text{Jacob}})$ per worker, where $M_{\text{Jacob}}$ is the size of one sparse transposed Jacobian. This scales inversely with $p$ (number of workers), unlike pipeline parallelism which grows linearly: $M_{\text{Pipeline}} = O(n/p + p) \cdot M_{\vec{x}}$.

**Sparse Jacobian Sparsity (VGG-11 examples):**

| Operator | Guaranteed Zero Fraction | CSR Speedup vs. Dense |
|----------|-------------------------|-----------------------|
| Convolution | $1 - \frac{h_f w_f}{h_i w_i} \approx 0.99157$ | $8.3 \times 10^3$ |
| ReLU | $1 - \frac{1}{chw} \approx 0.99998$ | $1.2 \times 10^6$ |
| Max-pooling | $1 - \frac{h_f w_f}{c_i h_i w_i} \approx 0.99994$ | $1.5 \times 10^5$ |

## Applicability

- **Deep RNN training**: The primary use case — vanilla RNNs and GRUs where the backward pass through time (BPTT) is a long sequential chain of Jacobian-vector products. BPPSA achieves up to 108x speedup on the backward pass for RNNs with long sequence lengths ($T$ up to 30,000).
- **Very deep feedforward networks**: Models like ResNet-1000+ where layer-sequential backpropagation is a bottleneck under model parallelism.
- **Pruned network retraining**: Pruned weights increase Jacobian sparsity (e.g., 97% weight pruning in VGG-11), reducing $P_{\text{Blelloch}}$ and making the scan-based approach more advantageous.
- **Small-batch training regimes**: When batch size is small (GPU underutilized by data parallelism), BPPSA provides an orthogonal axis of parallelism across layers/timesteps.

## Limitations

- **Per-step cost overhead**: The scan operator involves sparse matrix-matrix multiplication ($P_{\text{Blelloch}}$) which is more expensive than the sequential matrix-vector product ($P_{\text{Linear}}$). The method only wins when $n / \log n > P_{\text{Blelloch}} / P_{\text{Linear}}$, i.e., when the number of layers/timesteps is sufficiently large relative to the cost ratio.
- **Jacobian materialization**: Each layer's transposed Jacobian must be explicitly computed and stored in sparse format. While sparsity makes this feasible, it adds overhead compared to implicit Jacobian-vector products used in standard autograd.
- **Not applicable to attention layers**: Self-attention Jacobians are dense (every output depends on every input), negating the sparsity advantage. Best suited for conv/ReLU/pooling chains and recurrent architectures.
- **GPU arithmetic intensity**: Sparse matrix-matrix multiplication on GPUs has low arithmetic intensity compared to dense GEMM, limiting tensor core utilization. The scan requires $O(\log n)$ sequential sparse matmuls, each of which may underutilize hardware.
- **Numerical precision**: Changing the order of matrix multiplications (scan vs. sequential) introduces different floating-point rounding, though experiments show no impact on convergence.

## Implementation Notes

```python
# BPPSA: Parallel scan over transposed Jacobians for backpropagation
# The key idea: BP gradient chain = exclusive scan with non-commutative operator

def bppsa_backward(grad_output, jacobians_T):
    """
    grad_output: gradient from the loss, shape matching last layer output
    jacobians_T: list of n transposed Jacobian matrices (sparse CSR format)
                 [dxn/dxn-1)^T, (dxn-1/dxn-2)^T, ..., (dx2/dx1)^T, (dx1/dx0)^T]

    Returns: all layer gradients [grad_xn, grad_xn-1, ..., grad_x1]
    """
    # Build scan input array:
    # a = [grad_xn_l, J_n^T, J_{n-1}^T, ..., J_1^T]
    # where grad_xn_l is the loss gradient (treated as a row vector/matrix)

    n = len(jacobians_T)
    a = [grad_output] + jacobians_T  # length n+1

    # Modified Blelloch scan with non-commutative diamond operator:
    # A diamond B = B @ A  (note: reversed order!)

    # Phase 1: Up-sweep (reduce) — standard order
    for d in range(int(np.log2(n + 1))):
        stride = 2 ** (d + 1)
        for i in range(0, n + 1, stride):  # parallel over i
            l_idx = i + 2**d - 1
            r_idx = min(i + stride - 1, n)
            a[r_idx] = a[l_idx] @ a[r_idx]  # a[l] diamond a[r]

    # Phase 2: Down-sweep — REVERSED operand order for diamond
    a[n] = identity_matrix
    for d in range(int(np.log2(n + 1)) - 1, -1, -1):
        stride = 2 ** (d + 1)
        for i in range(0, n + 1, stride):  # parallel over i
            l_idx = i + 2**d - 1
            r_idx = min(i + stride - 1, n)
            T = a[l_idx]
            a[l_idx] = a[r_idx]           # left <- parent
            a[r_idx] = a[r_idx] @ T       # right <- parent diamond saved (REVERSED)

    # a now contains [I, grad_xn, grad_xn-1, ..., grad_x1]
    return a[1:]  # all layer gradients

# On GPU: implement as two CUDA kernels (up-sweep + down-sweep)
# Each thread block handles one diamond operation (sparse matmul)
# Shared memory used for intermediate sparse matrix caching
# Synchronization between levels via kernel launches in same CUDA stream
```

## References

- Wang, S., Bai, Y. & Pekhimenko, G. (2020). BPPSA: Scaling Back-propagation by Parallel Scan Algorithm. *Proceedings of the 3rd MLSys Conference*. arXiv:1907.10134.
- Blelloch, G.E. (1990). Prefix Sums and Their Applications. CMU-CS-90-190.
- Hillis, W.D. & Steele, G.L. (1986). Data Parallel Algorithms. *Communications of the ACM*.
- Rumelhart, D.E., Hinton, G.E. & Williams, R.J. (1988). Learning representations by back-propagating errors. *Nature*, 323, 533-536.
