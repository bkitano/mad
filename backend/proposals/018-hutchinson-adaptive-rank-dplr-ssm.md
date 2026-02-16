---
status: ongoing
priority: medium
created: 2026-02-15
based_on: hutchinson-stochastic-trace-estimator, power-series-log-determinant, diagonal-plus-low-rank-(dplr), tangent-space-projection-lrpd, matrix-determinant-lemma, woodbury-resolvent-identity, cauchy-kernel-trick, chunkwise-parallel-scan
experiment_number: 018
experiment_log: experiment-log-018.md
---

# Hutchinson Trace-Guided Adaptive Rank for DPLR SSMs

## Hypothesis

Using cheap Hutchinson trace estimates of $\text{tr}((A - \Lambda)^k)$ (the low-rank correction's contribution to the SSM transfer function) to dynamically allocate rank $r$ across layers and heads will produce DPLR SSMs that are (a) 30–50% more parameter-efficient than fixed-rank baselines at equal quality, and (b) reveal interpretable structure about which layers need dense state coupling vs. simple diagonal decay.

## Background

DPLR (Diagonal Plus Low-Rank) state matrices $A = \Lambda + PQ^*$ are the backbone of S4, S4D-LR, and related SSMs. The diagonal $\Lambda$ handles independent per-channel decay, while the low-rank correction $PQ^*$ (rank $r$) couples channels and enables richer dynamics. The Woodbury resolvent identity makes the transfer function evaluation efficient:

$$
(zI - A)^{-1} = (zI - \Lambda)^{-1} - (zI - \Lambda)^{-1} P \left(I + Q^*(zI - \Lambda)^{-1}P\right)^{-1} Q^* (zI - \Lambda)^{-1}
$$

costing $O(n + r^2)$ instead of $O(n^2)$.

**The problem**: The rank $r$ is a fixed hyperparameter, typically set uniformly across all layers (e.g., $r = 1$ in S4, $r = 2$ in some S4D variants). But there's no reason to believe all layers need the same rank:
- **Early layers** may process local patterns requiring only diagonal dynamics ($r = 0$)
- **Middle layers** may need rich cross-channel coupling ($r = 4$–$8$)
- **Late layers** may specialize in aggregation, needing moderate coupling ($r = 1$–$2$)

**No existing proposal addresses this**. Proposals 003, 004, and 011 all use DPLR with fixed rank. The tangent-space-projection-LRPD trick shows how to efficiently evolve on the LRPD manifold, but doesn't address rank selection. The Hutchinson trace estimator is documented but unused in any SSM proposal.

**The key insight**: The low-rank correction's "importance" at any layer can be measured cheaply via Hutchinson trace estimation of the correction term's contribution to the transfer function. Specifically:

$$
\text{tr}\left((PQ^*)^k\right) = \text{tr}\left(P(Q^*P)^{k-1}Q^*\right) = \text{tr}\left((Q^*P)^k\right) \cdot \frac{1}{1} \quad (\text{matrix trace cycling})
$$

This is an $r \times r$ trace (trivially cheap for small $r$), but for the *resolvent-weighted* version that captures the actual transfer function impact:

$$
\mathcal{I}_{\text{LR}}(\ell) = \frac{1}{|\Omega|} \sum_{\omega \in \Omega} \left\| (zI - \Lambda)^{-1} P \left(I + Q^*(zI - \Lambda)^{-1}P\right)^{-1} Q^* (zI - \Lambda)^{-1} \right\|_F
$$

we need Hutchinson. The Frobenius norm of this correction term tells us how much the low-rank part contributes to the output at layer $\ell$, relative to the diagonal part.

## Mathematical Formulation

**DPLR SSM at layer $\ell$:**

$$
A^{(\ell)} = \Lambda^{(\ell)} + P^{(\ell)} (Q^{(\ell)})^* \in \mathbb{C}^{n \times n}
$$

where $\Lambda^{(\ell)} \in \mathbb{C}^{n \times n}$ is diagonal, $P^{(\ell)}, Q^{(\ell)} \in \mathbb{C}^{n \times r_\ell}$, and $r_\ell$ is the layer-specific rank.

**Low-Rank Importance Score (Hutchinson-estimated):**

Define the resolvent correction at frequency $\omega$:

$$
R_{\text{LR}}^{(\ell)}(\omega) = (i\omega I - \Lambda^{(\ell)})^{-1} P^{(\ell)} \left(I + (Q^{(\ell)})^*(i\omega I - \Lambda^{(\ell)})^{-1}P^{(\ell)}\right)^{-1} (Q^{(\ell)})^* (i\omega I - \Lambda^{(\ell)})^{-1}
$$

The importance score is:

$$
\mathcal{I}^{(\ell)} = \mathbb{E}_\omega\left[\|R_{\text{LR}}^{(\ell)}(\omega)\|_F^2\right] = \mathbb{E}_\omega\left[\text{tr}\left(R_{\text{LR}}^{(\ell)}(\omega)^* R_{\text{LR}}^{(\ell)}(\omega)\right)\right]
$$

Using Hutchinson with $m = 1$ probe vector:

$$
\hat{\mathcal{I}}^{(\ell)} \approx \frac{1}{|\Omega|} \sum_{\omega \in \Omega} g^* R_{\text{LR}}^{(\ell)}(\omega)^* R_{\text{LR}}^{(\ell)}(\omega) g, \quad g \sim \mathcal{N}(0, I)
$$

**Cost**: For each frequency $\omega$, computing $R_{\text{LR}}^{(\ell)}(\omega) g$ costs:
1. $(i\omega I - \Lambda)^{-1} g$: $O(n)$ (diagonal inverse)
2. $(Q^*)^{(\ell)} \cdot \text{result}$: $O(nr)$ (thin matvec)
3. Solve $r \times r$ system: $O(r^3)$
4. $P^{(\ell)} \cdot \text{result}$: $O(nr)$
5. $(i\omega I - \Lambda)^{-1} \cdot \text{result}$: $O(n)$

**Total per frequency**: $O(nr + r^3)$. With $|\Omega| = 16$ sampled frequencies and $m = 1$ probe: $O(16(nr + r^3))$ — negligible compared to a training step.

**Adaptive Rank Allocation:**

Given a total rank budget $R_{\text{total}} = \sum_\ell r_\ell$, allocate ranks proportional to importance:

$$
r_\ell = \text{round}\left(\frac{\mathcal{I}^{(\ell)}}{\sum_j \mathcal{I}^{(j)}} \cdot R_{\text{total}}\right)
$$

with $r_\ell \geq r_{\min} = 1$ (every layer gets at least rank 1).

**Rank Adaptation Schedule:**

- **Phase 1 (warmup, steps 0–$S_1$)**: All layers at $r_{\max}$ (generous rank). Train normally.
- **Phase 2 (measurement, step $S_1$)**: Compute $\hat{\mathcal{I}}^{(\ell)}$ for all layers using Hutchinson.
- **Phase 3 (pruning, steps $S_1$–$S_2$)**: Gradually reduce ranks of low-importance layers via SVD truncation of $PQ^*$:

$$
PQ^* = U \Sigma V^* \approx U_{:r_\ell} \Sigma_{:r_\ell} V_{:r_\ell}^*
$$

Set $P^{(\ell)}_{\text{new}} = U_{:r_\ell} \Sigma_{:r_\ell}^{1/2}$, $Q^{(\ell)}_{\text{new}} = V_{:r_\ell} \Sigma_{:r_\ell}^{1/2}$.

- **Phase 4 (fine-tuning, steps $S_2$–end)**: Train with fixed adapted ranks.

**Connection to Power Series Log-Determinant:**

The matrix determinant lemma gives:

$$
\det(zI - A) = \det(zI - \Lambda) \cdot \det\left(I + Q^*(zI - \Lambda)^{-1}P\right)
$$

The log-determinant of the correction factor:

$$
\ln \det\left(I + Q^*(zI - \Lambda)^{-1}P\right) = \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{k} \text{tr}\left(\left(Q^*(zI - \Lambda)^{-1}P\right)^k\right)
$$

This is a power series of traces of $r \times r$ matrices — each term is $O(r^3)$, no Hutchinson needed! When $\|Q^*(zI - \Lambda)^{-1}P\|_2 < 1$ (which the Cauchy kernel evaluation ensures for frequencies away from eigenvalues), truncating at $k = 4$ terms gives:

$$
\hat{\mathcal{I}}_{\text{logdet}}^{(\ell)} = \mathbb{E}_\omega\left[\left|\sum_{k=1}^{4} \frac{(-1)^{k+1}}{k} \text{tr}\left(\left(Q^*(i\omega I - \Lambda)^{-1}P\right)^k\right)\right|\right]
$$

This is even cheaper than Hutchinson ($O(r^3)$ vs. $O(nr)$) and directly measures how much the low-rank correction affects the transfer function's determinant structure.

**Key Variables:**
- $\Lambda^{(\ell)} \in \mathbb{C}^{n \times n}$ — diagonal eigenvalues at layer $\ell$
- $P^{(\ell)}, Q^{(\ell)} \in \mathbb{C}^{n \times r_\ell}$ — low-rank factors
- $r_\ell$ — adaptive rank at layer $\ell$
- $\mathcal{I}^{(\ell)}$ — low-rank importance score
- $R_{\text{total}}$ — total rank budget across layers

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | DPLR SSM (S4/S5 variant) |
| Layers | $L = 6$–$12$ |
| Hidden dim | $d = 256$–$512$ |
| State dim | $n = 64$ per head |
| Initial rank | $r_{\max} = 8$ (all layers) |
| Rank budget | $R_{\text{total}} = L \times r_{\text{avg}}$ where $r_{\text{avg}} = 4$ |
| Importance estimator | Power-series log-det ($k = 4$ terms, $|\Omega| = 16$ frequencies) |

### Baseline

1. **Fixed rank $r = 4$ everywhere** (standard DPLR): Same total parameters as adaptive budget
2. **Fixed rank $r = 8$ everywhere** (over-parameterized): Upper bound on quality
3. **Fixed rank $r = 1$ everywhere** (minimal): Lower bound on quality (near S4D)
4. **Random rank allocation**: Same budget $R_{\text{total}}$, ranks assigned randomly to layers — ablation for whether the importance-guided allocation matters

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity | $\leq$ fixed $r = 4$ | WikiText-103 validation |
| Parameters | $< 0.8\times$ fixed $r = 8$ | Total trainable params |
| LRA accuracy | $\geq$ fixed $r = 4$ | Long Range Arena benchmark |
| Importance variance | High | $\text{Var}(\mathcal{I}^{(\ell)}) / \text{mean}(\mathcal{I}^{(\ell)})^2$ — coefficient of variation |
| Rank distribution | Non-uniform | Entropy of $\{r_\ell / R_{\text{total}}\}_\ell$ |

### Estimated Compute

**MVE**: < 10 minutes, single GPU
**Phase 1** (LRA evaluation): ~40 GPU-hours on A100
**Phase 2** (language modeling): ~120 GPU-hours on A100
**Total**: ~160 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- Adaptive rank matches or exceeds fixed $r = 4$ quality with the same parameter budget
- Adaptive rank approaches fixed $r = 8$ quality with $\sim 50\%$ fewer low-rank parameters
- The importance scores $\mathcal{I}^{(\ell)}$ show clear structure: early and late layers need less rank, middle layers need more (or some other interpretable pattern)
- Coefficient of variation of importance scores $> 0.5$, confirming that uniform rank is suboptimal

**If hypothesis is wrong:**
- If importance scores are nearly uniform: DPLR SSMs distribute information uniformly across layers, and fixed rank is already optimal
- If adaptive rank hurts quality: the SVD truncation destroys training dynamics (e.g., important gradient pathways through low-singular-value directions), suggesting that rank should be set at initialization, not pruned mid-training
- If random allocation matches importance-guided: the total budget matters more than allocation, suggesting diminishing returns from the low-rank correction beyond a threshold

## Minimum Viable Experiment

### Setup
- **Model**: 4-layer DPLR SSM, $n = 32$, $d = 64$, $r_{\max} = 8$, ~80K params
- **Task**: Sequential CIFAR-10 (sCIFAR) — a standard SSM benchmark that requires moderate long-range modeling
- **Data**: CIFAR-10 flattened to length-1024 sequences
- **Compute**: Single GPU, < 10 minutes

### Procedure
1. Train all 4 layers at $r = 8$ for 2K steps
2. Compute importance scores $\hat{\mathcal{I}}^{(\ell)}$ via power-series log-det method
3. Print the rank distribution across layers
4. Truncate to adaptive ranks with budget $R_{\text{total}} = 16$ (average $r = 4$)
5. Fine-tune for 2K more steps
6. Compare final accuracy against:
   - Baseline A: fixed $r = 4$ trained for 4K steps
   - Baseline B: fixed $r = 8$ trained for 4K steps

### Success Criteria
- Importance scores are non-uniform: $\max_\ell \mathcal{I}^{(\ell)} / \min_\ell \mathcal{I}^{(\ell)} > 2$ (at least $2\times$ variation)
- Adaptive $r$ (budget 16) achieves $\geq 95\%$ of fixed $r = 8$ (budget 32) accuracy
- Adaptive $r$ (budget 16) outperforms fixed $r = 4$ (budget 16) by $> 1\%$ accuracy

### Failure Criteria
- If importance scores are uniform (ratio $< 1.3$): there's no benefit to adaptive allocation for this model/task
- If accuracy degrades significantly after truncation ($> 5\%$ drop): SVD truncation disrupts training

### Why This Test Is Sufficient
- sCIFAR is an established benchmark that exercises long-range dependencies
- The 4-layer model is small enough to train quickly but large enough to show layer-wise differentiation
- If importance varies across just 4 layers, it will certainly vary across 12+ layers in larger models
- The power-series log-det computation is the same at any scale — if it works here, it scales

## Theoretical Analysis

**Importance estimation cost:**

| Operation | Cost per layer | Total ($L$ layers) |
|-----------|---------------|-------------------|
| Power-series log-det ($k = 4$ terms, $|\Omega| = 16$) | $O(16 \cdot 4 \cdot r^3) = O(64 r^3)$ | $O(64 L r^3)$ |
| Hutchinson Frobenius norm ($m = 1$, $|\Omega| = 16$) | $O(16(nr + r^3))$ | $O(16L(nr + r^3))$ |
| SVD truncation (one-time) | $O(nr^2)$ | $O(Lnr^2)$ |

For $L = 8$, $n = 64$, $r = 8$: power-series cost is $64 \times 8 \times 512 = 262K$ FLOPs — negligible vs. a single training step.

**Parameter savings:**

| Configuration | Parameters in low-rank factors | Total model params (relative) |
|---------------|-------------------------------|-------------------------------|
| Fixed $r = 8$ | $2Lnr = 16Ln$ | $1.0\times$ |
| Fixed $r = 4$ | $8Ln$ | $0.5\times$ (low-rank part) |
| Adaptive (avg $r = 4$) | $8Ln$ (same total) | $0.5\times$ but better allocated |

The hypothesis is that adaptive allocation extracts more value per parameter.

**Connection to the LRPD manifold:**

The tangent-space-projection-LRPD trick shows that the manifold of rank-$r$ LRPD matrices is smooth, and projecting onto it preserves dynamics. Our SVD truncation step is a special case: it's the orthogonal projection of $PQ^*$ onto the rank-$r_\ell$ manifold, which the tangent space theory guarantees is well-behaved for small rank reductions.

## Risks & Limitations

1. **Importance measurement timing**: Measuring at step $S_1$ captures the importance at that training stage, which may shift later. Solution: periodic re-measurement (every $N$ steps), though this adds complexity.
2. **SVD truncation shock**: Abruptly reducing rank may cause a training instability "shock." Solution: gradual rank reduction over multiple steps via soft thresholding of singular values.
3. **Task dependence**: The optimal rank distribution likely depends on the task. The importance metric may need to be evaluated on a held-out validation set rather than training data.
4. **Small effect size**: If all layers genuinely need similar rank, the improvement over fixed allocation will be marginal. The experiment may show that DPLR SSMs are already well-calibrated with uniform rank.
5. **Cauchy kernel singularities**: The power-series log-det requires $\|Q^*(zI - \Lambda)^{-1}P\|_2 < 1$. Near eigenvalue resonances, this bound is violated, requiring the Hutchinson Frobenius norm fallback.

## Follow-up Experiments

1. **Input-dependent adaptive rank**: Instead of fixed-per-layer rank, use the importance score to dynamically adjust rank at each time step — layers that need more coupling for specific inputs get higher rank. This connects to proposal 011 (Neumann resolvent) since the Neumann series naturally provides adaptive-order approximation.
2. **Joint importance + Neumann**: Combine this proposal with proposal 011: use the Hutchinson importance score to decide the Neumann truncation order per layer, rather than a global $k$.
3. **Initialization-time rank allocation**: Instead of train-then-prune, use importance scores from a short warmup (100 steps) to set ranks at initialization, avoiding the SVD truncation step entirely.
4. **Cross-architecture transfer**: Measure importance profiles on a small model, transfer the rank distribution to a larger model (same number of layers, larger $n$ and $d$). If importance patterns are architecture-general, this saves the measurement cost at scale.
5. **Extending to Monarch/HSS**: Apply the same importance-guided rank allocation to Monarch-factored SSMs (proposal 006) or HSS linear attention (proposal 005), where the "rank" corresponds to the number of Monarch factors or HSS rank respectively.
