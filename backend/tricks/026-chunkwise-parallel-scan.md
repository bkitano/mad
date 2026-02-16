# 026: Chunkwise Parallel Scan

**Category**: parallelization
**Gain type**: efficiency
**Source**: Parallel algorithms / GPU computing
**Documented**: 2026-02-10

## Description

Break a sequential recurrence into chunks of size $C$, process each chunk in parallel, then combine results. Within each chunk, the recurrence is unrolled into a matrix form that can leverage GPU parallelism. Between chunks, carry over the accumulated state.

## Mathematical Form

**Core Operation:**

For a linear recurrence $h_t = A_t h_{t-1} + b_t$:

$$
h_t = A_t h_{t-1} + b_t \quad \Rightarrow \quad h_t = \left(\prod_{i=s+1}^{t} A_i\right) h_s + \sum_{j=s+1}^{t} \left(\prod_{i=j+1}^{t} A_i\right) b_j
$$

**Key Definitions:**

- $h_t \in \mathbb{R}^n$ — hidden state at time $t$
- $A_t \in \mathbb{R}^{n \times n}$ — state transition matrix
- $b_t \in \mathbb{R}^n$ — input contribution
- $C$ — chunk size

**Block Algorithm:**

1. **Divide**: Split sequence into chunks of size $C$
2. **Intra-chunk**: Within chunk $j$, compute all $h_{jC+1}, \ldots, h_{(j+1)C}$ assuming $h_{jC} = 0$:
   $$
   \tilde{h}_t^{(j)} = \sum_{i=jC+1}^{t} \left(\prod_{k=i+1}^{t} A_k\right) b_i
   $$
3. **Inter-chunk scan**: Propagate boundary states across chunks via associative scan
4. **Correction**: Add correction from $h_{jC}$ propagated through chunk:
   $$
   h_t = \tilde{h}_t^{(j)} + \left(\prod_{i=jC+1}^{t} A_i\right) h_{jC}
   $$

Step 2 is embarrassingly parallel across positions within each chunk.

**Interpolation:**

$$
C = 1 \Rightarrow \text{fully recurrent}, \quad C = T \Rightarrow \text{fully parallel}
$$

## Complexity

| Operation | Sequential | Chunkwise |
|-----------|------------|-----------|
| Sequential steps | $O(T)$ | $O(T/C)$ |
| Parallel work | $O(1)$ | $O(C)$ per chunk |
| **Total** | $O(T)$ serial | $O(T/C)$ serial + $O(C)$ parallel |

**Memory:** $O(C \cdot n)$ for intermediate results within chunk

## Applicability

Any linear recurrence: DeltaNet, Mamba/S4, RetNet, RWKV, linear attention. The chunk size $C$ is the key knob: small $C$ = low memory, large $C$ = high parallelism.

## Limitations

- Requires the recurrence to be expressible in matrix form (linear or affine)
- Non-linear recurrences (e.g., GRU, LSTM) cannot be directly chunked this way
- Memory scales with $C$ for storing intermediate results
- Optimal $C$ depends on hardware (GPU memory, tensor core tile sizes)

## Implementation Notes

```python
# Chunkwise parallel scan (simplified)
def chunkwise_scan(A, b, chunk_size):
    T = len(b)
    chunks = T // chunk_size

    # Intra-chunk: parallel within each chunk
    chunk_outputs = parallel_map(
        lambda j: compute_chunk(A[j*C:(j+1)*C], b[j*C:(j+1)*C]),
        range(chunks)
    )

    # Inter-chunk: sequential scan over chunk boundaries
    states = sequential_scan(chunk_outputs)

    # Combine: add corrections
    return combine(chunk_outputs, states)
```

## References

- Blelloch (1990). Prefix Sums and Their Applications.
- Martin & Cundy (2018). Parallelizing Linear Recurrent Neural Nets Over Sequence Length.
- Hua et al. (2022). Transformer Quality in Linear Time.
