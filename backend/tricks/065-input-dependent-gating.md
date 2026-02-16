# 065: Input-Dependent Gating

**Category**: algebraic
**Gain type**: flexibility
**Source**: Various (Mamba, Gated DeltaNet, etc.)
**Documented**: 2026-02-10

## Description

Make the decay/gate parameters of a recurrence input-dependent instead of fixed. This transforms a linear time-invariant (LTI) system into a linear time-varying (LTV) system, dramatically increasing the set of functions the model can represent.

## Mathematical Form

**Core Operation:**

**LTI (fixed):**
$$
h_t = A h_{t-1} + B x_t \quad \text{($A$ is constant)}
$$

**LTV (gated):**
$$
h_t = A(x_t) h_{t-1} + B(x_t) x_t \quad \text{($A$ depends on input)}
$$

**Key Definitions:**

- $h_t \in \mathbb{R}^n$ — hidden state
- $x_t \in \mathbb{R}^d$ — input
- $A \in \mathbb{R}^{n \times n}$ — state transition (LTI: fixed, LTV: $A(x_t)$)
- $B \in \mathbb{R}^{n \times d}$ — input projection

**Examples:**

| Model | LTI | LTV |
|-------|-----|-----|
| RetNet → Mamba-2 | Fixed decay $\gamma$ | Input-dependent decay $\gamma(x_t)$ |
| DeltaNet → Gated DeltaNet | Fixed update | Input-gated update $\beta(x_t)$ |
| S4 → S6 (Mamba) | Fixed $A$ | Selective $A(x_t)$ |

**Gating Mechanism:**

Typically implemented via sigmoid activation:
$$
\gamma_t = \sigma(W_\gamma x_t + b_\gamma) \in (0, 1)
$$

The state transition becomes:
$$
h_t = \gamma_t \odot h_{t-1} + (1 - \gamma_t) \odot \tilde{h}_t
$$

## Complexity

| Property | LTI | LTV |
|----------|-----|-----|
| Expressivity | Fixed convolutions | Data-dependent computation |
| Training | Convolutional (parallel) | Requires parallel scan tricks |
| Parameters | Fewer (shared $A$) | More (gate projections) |

**Expressivity gain:** LTV systems can represent context-dependent operations that LTI systems cannot, such as selective copying or content-based addressing.

## Applicability

Any linear recurrence. This is the key idea that separates 'old-school' SSMs from modern ones (Mamba, GLA, etc.).

## Limitations

- Input-dependent gates make the system non-linear, which complicates parallel computation
- Requires the chunkwise parallel scan trick for efficient training
- Also adds parameters and may increase overfitting risk
- Loses the ability to use frequency-domain (convolutional) training

## Implementation Notes

```python
# Input-dependent gating (Mamba-style)
def selective_ssm(x, A_base, B, C):
    # x: (T, d) input sequence
    # Compute input-dependent gates
    delta = softplus(linear_delta(x))  # (T, n) - controls decay rate
    A = torch.exp(-delta) * A_base     # (T, n, n) - input-dependent A

    # Run selective scan with input-dependent A
    h = selective_scan(A, B @ x, h0=None)
    return C @ h
```

## References

- Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
- Yang et al. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training.
- Yang et al. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule.
