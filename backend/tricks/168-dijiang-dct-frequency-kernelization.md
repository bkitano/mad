# 168: DiJiang DCT Frequency Domain Kernelization

**Category**: approximation
**Gain type**: efficiency
**Source**: Chen, Liu, Wang, Tian & Wang (2024), "DiJiang: Efficient Large Language Models through Compact Kernelization" (ICML 2024)
**Paper**: [papers/dijiang-dct-kernelization.pdf]
**Documented**: 2026-02-15

## Description

DiJiang replaces softmax attention with a **Discrete Cosine Transform (DCT)-based kernel** that maps queries and keys into the frequency domain, yielding linear-complexity attention. Unlike FAVOR+ (trick 045), which uses random Fourier/exponential feature maps with Monte Carlo sampling, DiJiang uses the DCT matrix directly as a deterministic, structured projection — eliminating randomness entirely and leveraging the DCT's energy compaction property to produce compact, high-quality feature representations with fewer coefficients.

The key contributions are threefold:

1. **Weighted Quasi-Monte Carlo (WQMC) sampling**: The paper observes that the main source of approximation error in Performer/FAVOR+ comes from Monte Carlo sampling of the kernel integral. By switching to Quasi-Monte Carlo with learned importance weights (the "Weighted Positive Fixed Features" or WPFF map), the approximation error bound improves from $O(1/\sqrt{m})$ (Monte Carlo) to $O(1/m)$ (QMC), where $m$ is the number of features.

2. **DCT as the frequency transform**: Instead of using random projections ($\omega \sim p(\cdot)$) to construct the feature map, DiJiang uses the DCT coefficient matrix $\mathcal{C} \in \mathbb{R}^{d \times d}$ as a fixed, deterministic projection. The DCT operates in the real domain (unlike FFT), avoids the power-of-2 constraint of Hadamard transforms (unlike SORF, trick 155), and has $O(d \log d)$ fast algorithms. Its energy compaction property means a small number of DCT coefficients capture most of the signal energy, making $m = d$ features sufficient.

3. **Pre-trained model conversion**: A vanilla softmax Transformer can be converted to a DiJiang linear-attention model with only ~2% of the original training cost via fine-tuning. DiJiang-7B matches LLaMA2-7B benchmarks using only 40B tokens (vs 2000B for LLaMA2).

The resulting "Frequency Kernel Attention" (FKA) computes $\text{FKA}(Q,K,V) = \phi_{\text{WDCF}}(Q)\phi_{\text{WDCF}}(K)^\top V$ where $\phi_{\text{WDCF}}$ is the Weighted DCT Feature map, achieving $O(nd[\log d + d])$ complexity instead of $O(n^2 d)$.

## Mathematical Form

**Core Operation:**

Standard softmax attention:

$$
o_i = \sum_{j=1}^{n} \frac{e^{q_i \cdot k_j^\top}}{\sum_{j'=1}^{n} e^{q_i \cdot k_{j'}^\top}} v_j
$$

is replaced via kernel trick with feature map $\phi(\cdot) : \mathbb{R}^d \to \mathbb{R}^m$:

$$
o_i = \sum_{j=1}^{n} \frac{\phi(q_i)\phi(k_j)^\top}{\sum_{j'=1}^{n} \phi(q_i)\phi(k_{j'})^\top} v_j
$$

**Key Definitions:**

- $Q, K, V \in \mathbb{R}^{n \times d}$ — query, key, value matrices ($n$ = sequence length, $d$ = head dimension)
- $\mathcal{C} \in \mathbb{R}^{d \times d}$ — DCT coefficient matrix
- $T = \text{diag}(t_1, \ldots, t_m)$ — random diagonal matrix, $t_i$ from inverse CDF sampling
- $D \in \mathbb{R}^m$ — learnable weight vector (optimized per layer)

**Positive Fixed Features (PFF, Theorem 3.2):**

Using Bochner's theorem, the Gaussian kernel $K_G(x,y) = e^{-\|x-y\|^2/2}$ is approximated via QMC:

$$
\varphi_{\text{PFF}}(x) := \frac{e^{-\|x\|^2}}{\sqrt{m}} \left[ e^{\Phi^{-1}(t_1) x^\top v_1}, \ldots, e^{\Phi^{-1}(t_m) x^\top v_m} \right]^\top
$$

where $V = [v_1, \ldots, v_m] \in \mathcal{S}^{d \times m}$ is asymptotically uniformly distributed on the sphere and $t_i \sim U(0,1)$.

**Weighted PFF (WPFF, Theorem 3.3):**

$$
\varphi_{\text{WPFF}}(x) := \frac{D e^{-\|x\|^2}}{\sqrt{m}} \left[ e^{\Phi^{-1}(t_1) x^\top v_1}, \ldots, e^{\Phi^{-1}(t_m) x^\top v_m} \right]^\top
$$

where $D$ is a learnable parameter optimized to minimize the integral estimation error. The upper bound on the WPFF estimation error is no greater than the PFF error bound.

**DCT Coefficient Matrix:**

$$
\mathcal{C}_{j_1 j_2} = s_{j_1} s_{j_2} \sum_{i_1=0}^{n-1} \sum_{i_2=0}^{d-1} \cos\left(\frac{\pi(2i_1+1)j_1}{2d}\right) \cos\left(\frac{\pi(2i_2+1)j_2}{2d}\right)
$$

where $s_j = \sqrt{1/d}$ if $j = 0$ and $s_j = \sqrt{2/d}$ otherwise.

**Weighted DCT Feature Map (WDCF):**

$$
\phi_{\text{WDCF}}(x) = D e^{T \mathcal{C} x^\top}
$$

where $\mathcal{C} \in \mathbb{R}^{m \times d}$ is the DCT coefficient matrix, $D \in \mathbb{R}^m$ is the learnable weight, and $T = \text{diag}(t_1, \ldots, t_m)$ is the random diagonal from inverse CDF sampling.

**Frequency Kernel Attention (FKA):**

$$
\text{FKA}(Q, K, V) = \phi_{\text{WDCF}}(Q) \phi_{\text{WDCF}}(K)^\top V
$$

where $Q, K, V \in \mathbb{R}^{n \times d}$. Setting $m = d$ (the paper's recommended configuration), the feature map dimensionality matches the head dimension.

**QMC Error Bound:**

The QMC approximation achieves $O(1/m)$ convergence (vs $O(1/\sqrt{m})$ for MC):

$$
\left| \hat{K}(x,z) - K(x,z) \right| = O(1/m)
$$

compared to $O(1/m^{0.5})$ for the Monte Carlo method used in FAVOR+.

## Complexity

| Operation | Softmax Attention | FAVOR+ (MC) | DiJiang (WDCF) |
|-----------|------------------|-------------|----------------|
| Attention computation | $O(n^2 d)$ | $O(nmd)$ | $O(nd[\log d + d])$ |
| Memory | $O(n^2)$ | $O(nm)$ | $O(nd)$ |
| Feature map (per token) | N/A | $O(md)$ dense matmul | $O(d \log d)$ via fast DCT |
| Approximation error rate | exact | $O(1/\sqrt{m})$ | $O(1/m)$ |
| Conversion from pretrained | N/A | full retrain | ~2% training cost |

**Empirical results (Table 1, A800 GPU):**

| Model | Training (days) | Inference (tokens/s) |
|-------|----------------|---------------------|
| Pythia-70M | 21.3 | 2037 |
| DiJiang-70M | 1.3 | 2605 |
| Pythia-410M | 105.8 | 203 |
| DiJiang-410M | 6.6 | 787 |
| Pythia-1B | 201.2 | 105 |
| DiJiang-1B | 12.6 | 611 |
| Pythia-2.8B | 593.3 | 34 |
| DiJiang-2.8B | 37.1 | 284 |

**DiJiang-7B vs LLaMA2-7B:** Comparable benchmark performance (0.557 vs 0.565 average) with 40B tokens vs 2000B tokens (~1/50th training cost).

**Memory:** Constant with sequence length (vs quadratic for softmax). At sequence length 8192: ~2.5 GiB (DiJiang) vs ~8 GiB (Transformer) for Pythia-410M.

## Applicability

- **Pre-trained model conversion**: The primary use case — convert an existing softmax Transformer to linear attention with minimal fine-tuning. Demonstrated on Pythia (70M–2.8B), OPT-350M, TinyLLaMA-1.1B, and LLaMA2-7B
- **From-scratch training**: Can also train from scratch with ~10x reduced training cost and comparable quality
- **Causal (decoder-only) LLMs**: Designed specifically for autoregressive language models where FNet-style global frequency transforms fail (each new token would require recomputing the entire frequency representation)
- **GPU-friendly**: The DCT is a real-valued transform (no complex arithmetic like FFT), and has highly optimized CUDA implementations (cuFFT supports DCT). The feature map $\phi_{\text{WDCF}}(x) = D e^{T\mathcal{C}x^\top}$ is: (1) DCT projection via fast DCT in $O(d \log d)$, (2) elementwise diagonal scaling, (3) elementwise exp — all GPU-native operations
- **Combines with gating**: The paper uses RetNet-style learned gating mechanisms alongside the DCT kernelization for best results
- **Tensor core considerations**: The linear attention step $\phi(Q)\phi(K)^\top V$ becomes a standard matmul: $Q' \in \mathbb{R}^{n \times d}$ times $KV \in \mathbb{R}^{d \times d}$ — this is a tensor core-friendly GEMM. The feature map itself (fast DCT) doesn't use tensor cores, but the dominant operation (the linear attention matmul) does

## Limitations

- **Not exact softmax**: Like all linear attention methods, FKA cannot reproduce sharp/sparse attention patterns. The DCT kernel is an approximation of the Gaussian kernel which itself approximates softmax
- **Requires fine-tuning**: Converting a pre-trained model requires ~2% of original training cost, which for LLaMA2-7B scale is still 40B tokens on significant GPU resources
- **Accuracy gap**: DiJiang-7B averages 0.557 across benchmarks vs LLaMA2-7B's 0.565 — a small but nonzero gap, especially on reasoning tasks (BoolQ: 0.346 vs 0.485)
- **$m = d$ constraint**: The paper sets feature count $m = d$ to avoid increasing complexity. For $m > d$, one would need multiple DCT blocks (similar to SORF), potentially losing the energy compaction advantage
- **No power-of-2 requirement** (advantage over SORF): DCT works for any dimension $d$, but fast DCT algorithms are most efficient for certain factorizable sizes
- **Learnable parameters per layer**: The weight $D$ and diagonal $T$ add $O(d)$ trainable parameters per attention layer — negligible but nonzero

## Implementation Notes

```python
import torch
import torch.nn.functional as F
import math

def dct_matrix(d):
    """Construct the d x d Type-II DCT matrix."""
    n = torch.arange(d, dtype=torch.float32)
    k = torch.arange(d, dtype=torch.float32)
    # C[k, n] = s_k * cos(pi * (2n + 1) * k / (2d))
    C = torch.cos(math.pi * k.unsqueeze(1) * (2 * n.unsqueeze(0) + 1) / (2 * d))
    # Normalization: s_0 = sqrt(1/d), s_k = sqrt(2/d) for k > 0
    C[0, :] *= math.sqrt(1.0 / d)
    C[1:, :] *= math.sqrt(2.0 / d)
    return C

def wdcf_feature_map(x, C, T_diag, D_weight):
    """Weighted DCT Feature map.

    Args:
        x: (..., d) input (queries or keys)
        C: (d, d) DCT coefficient matrix
        T_diag: (d,) diagonal scaling from inverse CDF sampling
        D_weight: (d,) learnable weight vector
    Returns:
        phi: (..., d) feature map output
    """
    # Step 1: DCT projection — O(d log d) via fast DCT
    # In practice, use torch.fft.dct or cuFFT DCT
    proj = x @ C.T  # (*, d) — or use fast DCT

    # Step 2: Diagonal scaling
    proj = proj * T_diag  # (*, d) elementwise

    # Step 3: Exponential + learnable weight
    phi = D_weight * torch.exp(proj)  # (*, d)

    return phi

def dijiang_attention(Q, K, V, C, T_diag, D_weight):
    """DiJiang Frequency Kernel Attention.

    Replaces softmax(QK^T)V with phi(Q) @ (phi(K)^T @ V).

    Args:
        Q, K, V: (n, d) query, key, value
        C: (d, d) DCT matrix
        T_diag: (d,) diagonal from inverse CDF
        D_weight: (d,) learnable weight
    Returns:
        output: (n, d)
    """
    # Compute feature maps: O(n * d * log d) via fast DCT
    Q_prime = wdcf_feature_map(Q, C, T_diag, D_weight)  # (n, d)
    K_prime = wdcf_feature_map(K, C, T_diag, D_weight)  # (n, d)

    # Linear attention: O(n * d * d) — tensor core friendly!
    KV = K_prime.T @ V        # (d, d) — one GEMM
    K_sum = K_prime.sum(0)     # (d,) — for normalization

    num = Q_prime @ KV         # (n, d) — one GEMM
    den = Q_prime @ K_sum      # (n,) — one matvec

    return num / den.unsqueeze(-1)

# GPU efficiency analysis:
# 1. Feature map: fast DCT (O(d log d) per token) + elementwise ops
#    - cuFFT provides optimized DCT kernels
#    - d=64 or 128 typical for attention heads
#    - Can be fused: DCT + diag_scale + exp in one kernel
#
# 2. Linear attention: two GEMMs
#    - K'^T @ V: (d, n) @ (n, d) = (d, d) — fits tensor cores
#    - Q' @ KV: (n, d) @ (d, d) = (n, d) — fits tensor cores
#    - These are the dominant operations and are bandwidth-optimal
#
# 3. Memory: O(n*d) for features, O(d^2) for KV accumulator
#    No O(n^2) attention matrix ever materialized
#
# 4. vs SORF (trick 155): SORF uses 3 FWHT passes for the projection
#    DiJiang uses 1 fast DCT pass. DCT doesn't need power-of-2 padding.
#    Both are O(d log d) but DCT may have lower constant factor.
#
# 5. vs FAVOR+ (trick 045): Same linear attention structure, but
#    deterministic DCT replaces random orthogonal projection.
#    QMC sampling gives O(1/m) vs O(1/sqrt(m)) error rate.

# Code: https://github.com/YuchuanTian/DiJiang
```

## References

- Chen, H., Liu, Z., Wang, X., Tian, Y., & Wang, Y. (2024). DiJiang: Efficient Large Language Models through Compact Kernelization. ICML 2024. arXiv:2403.19928.
- Choromanski, K., Likhosherstov, V., et al. (2021). Rethinking Attention with Performers. ICLR 2021.
- Asmussen, S. & Glynn, P. W. (2007). Stochastic Simulation: Algorithms and Analysis. Springer.
- Ahmed, N., Natarajan, T., & Rao, K. R. (1974). Discrete Cosine Transform. IEEE Transactions on Computers.
- Lee-Thorp, J., Ainslie, J., Eckstein, I., & Ontanon, S. (2021). FNet: Mixing Tokens with Fourier Transforms. arXiv:2105.03824.
- Sun, Y., Dong, L., et al. (2023). Retentive Network: A Successor to Transformer for Large Language Models.
