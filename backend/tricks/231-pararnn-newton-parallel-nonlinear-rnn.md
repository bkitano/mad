# 231: ParaRNN — Newton-Iteration Parallel Nonlinear RNN

**Category**: parallelization
**Gain type**: efficiency
**Source**: Danieli, Rodriguez, Sarabia, Suau & Zappella (2025). ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models. Apple. arXiv:2510.21450.
**Paper**: papers/pararnn-parallel-nonlinear-rnn.pdf
**Documented**: 2026-02-15

## Description

ParaRNN breaks the sequence-parallelization barrier for **nonlinear** RNNs by recasting the sequential application of an RNN cell to a length-$L$ input sequence as the solution of a nonlinear system of $L$ equations, solved via Newton's method with a custom parallel reduction inner solver. This overcomes the fundamental limitation of standard parallel scan approaches (Mamba, S4, linear attention), which require the recurrence to be *linear* in the hidden state to exploit associativity. ParaRNN enables full nonlinearities (sigmoid, tanh gates) in the state transition, recovering the expressivity advantages of classical GRU/LSTM architectures while achieving training throughput competitive with Mamba2.

The method has two nested components: (1) an **outer Newton iteration** that linearizes the nonlinear system at each step, producing a block bi-diagonal linear system, and (2) an **inner parallel reduction** (a variant of parallel scan/cyclic reduction) that solves each linearized system in $O(\log L)$ parallel depth. In practice, only $N_{\text{its}} \approx 3$ Newton iterations suffice for convergence, making the total cost $O(N_{\text{its}} \log L)$ parallel steps — a dramatic improvement over the $O(L)$ sequential baseline.

A critical design choice is constraining the RNN cell's state-to-state weight matrices to be **diagonal** (as in Mamba), so that the per-step Jacobians $J_f|_{h_l}$ are diagonal matrices occupying $O(d_h)$ memory and enabling $O(d_h)$ element-wise parallel multiplication instead of $O(d_h^3)$ dense matmul.

## Mathematical Form

**Nonlinear RNN Recurrence:**

$$
\boldsymbol{h}_l = \boldsymbol{f}(\boldsymbol{h}_{l-1}, \boldsymbol{x}_l), \quad \forall l = 1, \ldots, L
$$

where $\boldsymbol{h}_l \in \mathbb{R}^{d_h}$ is the hidden state, $\boldsymbol{x}_l \in \mathbb{R}^{d_{\text{in}}}$ is the input, and $\boldsymbol{f}$ is a nonlinear cell action (e.g., GRU, LSTM).

**System-of-Equations Reformulation:**

Collating all $L$ recurrence steps into a single system of $L$ equations in $L$ unknowns $[\boldsymbol{h}_l]_{l=1}^L$:

$$
\begin{cases}
\boldsymbol{h}_1 - \boldsymbol{f}(\boldsymbol{0}, \boldsymbol{x}_1) = \boldsymbol{0} \\
\boldsymbol{h}_2 - \boldsymbol{f}(\boldsymbol{h}_1, \boldsymbol{x}_2) = \boldsymbol{0} \\
\vdots \\
\boldsymbol{h}_L - \boldsymbol{f}(\boldsymbol{h}_{L-1}, \boldsymbol{x}_L) = \boldsymbol{0}
\end{cases}
$$

**Newton's Method (Outer Loop):**

Given approximate solution $[\boldsymbol{h}_l^k]_{l=1}^L$ at iteration $k$, solve the linearized block bi-diagonal system:

$$
\begin{bmatrix} I & & \\ -J_f|_{\boldsymbol{h}_1^k} & I & \\ & \ddots & \ddots \\ & & -J_f|_{\boldsymbol{h}_{L-1}^k} & I \end{bmatrix} \begin{bmatrix} \delta\boldsymbol{h}_1^k \\ \delta\boldsymbol{h}_2^k \\ \vdots \\ \delta\boldsymbol{h}_L^k \end{bmatrix} = \begin{bmatrix} \boldsymbol{f}(\boldsymbol{0}, \boldsymbol{x}_1) - \boldsymbol{h}_1^k \\ \boldsymbol{f}(\boldsymbol{h}_1^k, \boldsymbol{x}_2) - \boldsymbol{h}_2^k \\ \vdots \\ \boldsymbol{f}(\boldsymbol{h}_{L-1}^k, \boldsymbol{x}_L) - \boldsymbol{h}_L^k \end{bmatrix}
$$

Update: $\boldsymbol{h}_l^{k+1} = \boldsymbol{h}_l^k + \delta\boldsymbol{h}_l^k$.

Here $J_f|_{\boldsymbol{h}_l^k} = \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{h}}(\boldsymbol{h}_l^k, \boldsymbol{x}_{l+1})$ is the Jacobian of the cell action with respect to the hidden state.

**Parallel Reduction (Inner Solver):**

The linearized system has the form $\delta\boldsymbol{h}_l^k = J_f|_{\boldsymbol{h}_l^k} \delta\boldsymbol{h}_{l-1}^k + \boldsymbol{r}_l$, which unrolls to:

$$
\delta\boldsymbol{h}_l^k = \sum_{s=1}^{l} \prod_{r=0}^{l-s-1} J_f|_{\boldsymbol{h}_{l-r}^k} \; \boldsymbol{r}_s
$$

This is solved via parallel reduction in $O(\log_2 L)$ steps:

$$
\text{Initialize: } A_l = -J_f|_{\boldsymbol{h}_l^k}, \quad \delta\boldsymbol{h}_l = \boldsymbol{r}_l
$$

$$
\text{For } i = 0 \text{ to } \lfloor\log_2 L\rfloor - 1: \quad \text{parfor } l = 2^i \text{ to } L:
$$

$$
\delta\boldsymbol{h}_l \leftarrow \delta\boldsymbol{h}_l - A_l \, \delta\boldsymbol{h}_{l-2^i+1}
$$

$$
A_l \leftarrow -A_l \, A_{l-2^i+1}
$$

**Backward Pass (Gradient Computation):**

The backward pass through the RNN is inherently linear (regardless of forward nonlinearity):

$$
\nabla_{\boldsymbol{h}_{l-1}} \mathcal{L} = J_f|_{\boldsymbol{h}_{l-1}}^\top \nabla_{\boldsymbol{h}_l} \mathcal{L} + \partial_{\boldsymbol{h}_{l-1}} \mathcal{L}, \quad \forall l = L, \ldots, 1
$$

This uses the same parallel reduction algorithm with transposed Jacobians, requiring no additional Newton iterations.

**Diagonal Jacobian Simplification (ParaGRU):**

For efficient parallel scan, state-to-state matrices $A_\star$ and peephole connections $C_\star$ are constrained to be diagonal:

$$
A_\star = \text{diag}(\boldsymbol{a}_\star), \quad C_\star = \text{diag}(\boldsymbol{c}_\star), \quad \boldsymbol{a}_\star, \boldsymbol{c}_\star \in \mathbb{R}^{d_h}
$$

This makes $J_{\text{GRU}}$ a diagonal matrix:

$$
J_{\text{GRU}} = \text{diag}(\boldsymbol{1} - \boldsymbol{z}_l) + \text{diag}((\boldsymbol{c}_l - \boldsymbol{h}_{l-1}) \odot \sigma_g'(\hat{\boldsymbol{z}}_l)) A_z + \text{diag}(\boldsymbol{z}_l \odot \sigma_h'(\hat{\boldsymbol{c}}_l)) \text{diag}(\boldsymbol{r}_l) A_r + \ldots
$$

occupying $O(d_h)$ memory per step, and enabling element-wise parallel products in the reduction.

## Complexity

| Operation | Sequential RNN | ParaRNN |
|-----------|---------------|---------|
| Forward pass depth | $O(L)$ | $O(N_{\text{its}} \log L)$ with $N_{\text{its}} \approx 3$ |
| Backward pass depth | $O(L)$ | $O(\log L)$ (single parallel reduction) |
| Per-step work (diagonal $J$) | $O(d_h)$ | $O(d_h)$ (element-wise) |
| Per-step work (dense $J$) | $O(d_h^2)$ | $O(d_h^3)$ (matmul — infeasible) |
| Total work (diagonal $J$) | $O(L \cdot d_h)$ | $O(N_{\text{its}} \cdot L \cdot d_h \log L)$ |

**Memory:** $O(L \cdot d_h)$ for diagonal Jacobians. Dense Jacobians would require $O(L \cdot d_h^2)$ — prohibitive and the reason for the diagonal constraint.

**Observed Speedups:**

| Configuration | Speedup over Sequential |
|---------------|------------------------|
| Parallel reduction only ($L = 2^9$) | $\sim 447\times$ (ParaGRU), $\sim 599\times$ (ParaLSTM) |
| Full Newton forward pass ($L = 2^9$) | $\sim 2.6\times$ over Mamba (ParaGRU, fused CUDA) |
| End-to-end naively | Up to $665\times$ over sequential |

## Applicability

- **Nonlinear RNN training at scale**: ParaGRU and ParaLSTM achieve competitive perplexity with Mamba2 and Transformers at 7B parameters on language modeling, proving nonlinear RNNs are viable LLM architectures when the parallelization barrier is removed.
- **Expressivity-critical tasks**: Nonlinear RNNs solve synthetic tasks (MQAR, k-hop, Parity) that linear SSMs like Mamba2 fundamentally cannot, while maintaining comparable training throughput.
- **Inference efficiency**: Like all RNNs, ParaGRU/ParaLSTM achieve $O(1)$ per-token inference cost (constant regardless of sequence length), with throughput of $\sim 35$-$37$ tokens/sec vs. Mamba's $\sim 27$ tokens/sec at the 7B scale.
- **Any Markovian RNN cell**: The ParaRNN framework automatically parallelizes any user-defined RNN cell — users only specify the recurrence formula, and the framework handles Jacobian assembly, Newton iteration, and parallel reduction via CUDA kernels.

## Limitations

- **Newton convergence requirement**: The method requires $O(1)$ Newton iterations for practical speedup. For the adapted GRU/LSTM cells, 3 iterations suffice, but new cell designs must be verified for convergence. Convergence in $L$ steps is guaranteed (Gonzalez et al., 2024), but slow convergence negates parallelization benefits.
- **Diagonal Jacobian constraint**: To make parallel reduction efficient ($O(d_h)$ per step instead of $O(d_h^3)$), state-to-state matrices must be diagonal, eliminating cross-dimensional state mixing within the RNN cell. Feature mixing is delegated to subsequent MLP layers (as in Mamba).
- **Work overhead**: Total work is $O(N_{\text{its}} \cdot L \cdot d_h \log L)$ vs. sequential $O(L \cdot d_h)$ — a $\log L$ factor overhead multiplied by Newton iterations. This is offset by massive parallelism on GPUs.
- **Perplexity gap**: At 7B parameters, ParaGRU (9.19 PPL) and ParaLSTM (9.16 PPL) slightly trail Mamba2 (8.62 PPL), suggesting the diagonal constraint may limit model quality.
- **Initial guess sensitivity**: Newton's method requires an initial guess $[\boldsymbol{h}_l^0]$. The paper uses all-zeros, but poor initialization could slow convergence for some cell types.

## Implementation Notes

```python
# ParaRNN: Newton + Parallel Reduction for nonlinear RNN

def pararnn_forward(f, x_seq, n_newton_iters=3):
    """
    f: RNN cell action, h_l = f(h_{l-1}, x_l)
    x_seq: input sequence [x_1, ..., x_L], each in R^{d_in}
    Returns: hidden states [h_1, ..., h_L]
    """
    L = len(x_seq)
    d_h = hidden_dim

    # Initialize: all-zero guess
    h = [zeros(d_h) for _ in range(L)]

    for k in range(n_newton_iters):
        # Step 1: Compute Jacobians and residuals (fully parallel over l)
        J = []  # Jacobians J_f|_{h_l^k}, each diagonal (d_h,)
        r = []  # Residuals f(h_{l-1}^k, x_l) - h_l^k
        for l in range(L):  # parallel
            J_l = jacobian_diag(f, h[l], x_seq[l])  # diagonal Jacobian
            r_l = f(h[max(l-1, 0)], x_seq[l]) - h[l]
            J.append(J_l)
            r.append(r_l)

        # Step 2: Parallel reduction to solve block bi-diagonal system
        # δh_l = J_l * δh_{l-1} + r_l
        delta_h = parallel_reduce(J, r, L)

        # Step 3: Update (parallel over l)
        for l in range(L):  # parallel
            h[l] = h[l] + delta_h[l]

    return h


def parallel_reduce(J_diag, r, L):
    """
    Solve δh_l = J_l * δh_{l-1} + r_l via parallel reduction.
    J_diag: list of diagonal Jacobians (each d_h vector)
    r: list of residual vectors (each d_h)
    O(log L) parallel depth, O(L * d_h) total work per level.
    """
    A = [-j for j in J_diag]  # A_l = -J_l (diagonal, stored as vectors)
    dh = [ri.clone() for ri in r]

    for i in range(int(math.log2(L))):
        stride = 2 ** i
        for l in range(stride, L):  # parallel over l
            # Element-wise ops (diagonal Jacobians → no matmul needed!)
            dh[l] = dh[l] - A[l] * dh[l - stride]  # element-wise
            A[l] = -A[l] * A[l - stride]            # element-wise

    return dh


# GPU Implementation Tiers:
# 1. Pure PyTorch: automatic differentiation, good for prototyping
# 2. CUDA parallel reduction: custom kernel for diagonal/block-diagonal J
#    - Hardware-aware hybrid: forward substitution within warp,
#      parallel reduction across warps/blocks
#    - Maximizes register utilization, minimizes shared/global memory traffic
# 3. Fully-fused CUDA: Newton loop + Jacobian assembly + parallel reduction
#    in a single kernel launch — eliminates all intermediate HBM traffic
#    Achieves ~2.6x speedup over Mamba at L=512

# For linear SSMs (Mamba), J_f = A_l (state matrix), Newton converges
# in exactly 1 iteration, reducing to standard parallel scan.
```

## References

- Danieli, F., Rodriguez, P., Sarabia, M., Suau, X. & Zappella, L. (2025). ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models. arXiv:2510.21450.
- Danieli, F., Sarabia, M., Suau, X., Rodriguez, P. & Zappella, L. (2023). DeepPCR: Parallelizing Sequential Operations in Neural Networks. NeurIPS 2023. arXiv:2309.16318.
- Gonzalez, X., Warrington, A., Smith, J.T.H. & Linderman, S.W. (2024). Towards scalable and stable parallelization of nonlinear RNNs. arXiv:2407.19115.
- Lim, Y.H., Zhu, Q., Selfridge, J. & Kasim, M.F. (2024). Parallelizing non-linear sequential models over the sequence length. arXiv:2309.12252.
- Feng, L., Tung, F., Ahmed, M.O., Bengio, Y. & Hajimirsadeghi, H. (2024). Were RNNs All We Needed? arXiv:2410.01201.
- Hillis, W.D. & Steele, G.L. (1986). Data Parallel Algorithms. *Communications of the ACM*.
- Blelloch, G.E. (1990). Prefix Sums and Their Applications. CMU-CS-90-190.
