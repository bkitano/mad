# 047: FlashInfer: Composable Attention JIT Fusion

**Category**: kernel
**Gain type**: efficiency
**Source**: Ye, Chen, Lai, Lin, Zhang, Wang, Chen, Kasikci, Grover, Krishnamurthy, Ceze — "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving" (MLSys 2025)
**Paper**: [papers/flashinfer-composable-attention-engine.pdf]
**Documented**: 2025-06-15

## Description

FlashInfer introduces a **composable JIT-compiled attention fusion** framework that addresses two critical limitations of existing fused attention kernels: (1) the combinatorial explosion of attention variants (RoPE, ALiBi, sliding window, logits soft-cap, GQA, MQA, sigmoid attention, etc.) that each require hand-written CUDA kernels, and (2) the heterogeneity of KV-cache storage formats (dense tensors, page tables, radix trees, sparse masks) that each require different memory access patterns.

The key insight is that all attention variants share a common computational skeleton — the FlashAttention tiled loop with online softmax — and differ only in a small number of **composable functors** that transform queries, keys, values, logits, masks, and outputs. FlashInfer defines a fixed set of functor insertion points within a CUDA/CUTLASS attention template, and uses **Just-In-Time (JIT) compilation** to specialize the kernel at init time by injecting user-defined CUDA code into these slots. This produces a fully fused, hardware-optimized kernel without writing a complete CUDA kernel from scratch.

Simultaneously, FlashInfer unifies all KV-cache storage formats under a single **block-sparse row (BSR)** abstraction. Page tables, radix trees, and sparse masks are all represented as BSR matrices with configurable block sizes $(B_r, B_c)$. This enables a single attention kernel template to handle all storage formats, with only the global-to-shared-memory loading logic varying. Furthermore, FlashInfer introduces **composable formats** — multiple BSR matrices with different block sizes that decompose the KV-cache for shared-prefix scenarios — enabling requests that share a KV-cache prefix to access it cooperatively through high-bandwidth shared memory rather than independently through low-bandwidth global memory.

The combination of JIT-compiled attention variants and unified BSR storage yields 29–69% inter-token latency reduction over Triton backends, 28–30% latency reduction for long-context inference, and 13–17% speedup for parallel generation.

## Mathematical Form

**Attention State Composition (the algebraic foundation):**

For a query $\mathbf{q}$ and key-value index set $\mathcal{I}$, define the **attention state** as:

$$
\text{AttentionState}(\mathcal{I}) = \begin{bmatrix} \mathbf{O}(\mathcal{I}) \\ \text{LSE}(\mathcal{I}) \end{bmatrix}
$$

where the **log-sum-exp scale** is:

$$
\text{LSE}(\mathcal{I}) = \log \sum_{i \in \mathcal{I}} \exp(\mathbf{q} \cdot \mathbf{k}_i)
$$

and the **attention output** is:

$$
\mathbf{O}(\mathcal{I}) = \sum_{i \in \mathcal{I}} \frac{\exp(\mathbf{q} \cdot \mathbf{k}_i)}{\exp(\text{LSE}(\mathcal{I}))} \cdot \mathbf{v}_i
$$

The composition operator $\oplus$ merges two disjoint index sets:

$$
\begin{bmatrix} \mathbf{O}(\mathcal{I} \cup \mathcal{J}) \\ \text{LSE}(\mathcal{I} \cup \mathcal{J}) \end{bmatrix} = \begin{bmatrix} \mathbf{O}(\mathcal{I}) \\ \text{LSE}(\mathcal{I}) \end{bmatrix} \oplus \begin{bmatrix} \mathbf{O}(\mathcal{J}) \\ \text{LSE}(\mathcal{J}) \end{bmatrix}
$$

$$
= \begin{bmatrix} \frac{\exp(\text{LSE}(\mathcal{I})) \cdot \mathbf{O}(\mathcal{I}) + \exp(\text{LSE}(\mathcal{J})) \cdot \mathbf{O}(\mathcal{J})}{\exp(\text{LSE}(\mathcal{I})) + \exp(\text{LSE}(\mathcal{J}))} \\ \log(\exp(\text{LSE}(\mathcal{I})) + \exp(\text{LSE}(\mathcal{J}))) \end{bmatrix}
$$

Since $\oplus$ is **associative and commutative**, attention states from arbitrary partitions of the KV-cache can be composed in any order. This enables:
- Splitting long KV sequences across CTAs (load balancing)
- Composing results from different BSR format blocks (shared prefix + unique suffix)
- Parallel generation with shared-prefix decoupling

**JIT Functor Insertion Points:**

The attention kernel template has the functional form:

$$
\mathbf{O} = f_{\text{epilogue}}\!\left(\text{scan}\!\left(f_{\text{logits}}\!\left(f_q(\mathbf{Q}) \cdot f_k(\mathbf{K})^T\right)\right) \cdot f_v(\mathbf{V})\right)
$$

where each $f$ is a user-definable functor:

- $f_q$: **QueryTransform** — applied to $\mathbf{Q}$ before attention (e.g., RoPE rotation, normalization)
- $f_k$: **KeyTransform** — applied to $\mathbf{K}$ before attention (e.g., RoPE rotation)
- $f_v$: **ValueTransform** — applied to $\mathbf{V}$ before aggregation
- $f_{\text{logits}}$: **LogitsTransform** + **LogitsMask** — applied to $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$ before softmax (e.g., logits soft-cap $\tanh(S/\text{scale}) \cdot \text{scale}$, causal mask, sliding window)
- $f_{\text{epilogue}}$: **OutputTransform** — applied to $\mathbf{O}$ before writing (e.g., projection)

Each functor has the signature:

$$
f: (\text{params}, \text{input}, \text{qo\_idx}, \text{kv\_idx}, \text{qo\_head\_idx}, \text{kv\_head\_idx}) \to \text{output}
$$

**Block-Sparse Row (BSR) Unified Format:**

The KV-cache is stored as a BSR matrix with block size $(B_r, B_c)$:

- $B_r$ = query tile size (rows per block, aligned to query length)
- $B_c$ = KV page size (columns per block, set by serving framework)

For a batch of $R$ requests with KV lengths $\{l_{\text{kv}}(i)\}$, the BSR matrix has:
- Row dimension: $\sum_i \lceil l_{\text{qo}}(i) / B_r \rceil$ blocks
- Column blocks per row: $\lceil l_{\text{kv}}(i) / B_c \rceil$ for request $i$
- Non-zero blocks correspond to KV-cache pages accessed by queries

**Composable Formats for Shared Prefix:**

When requests share a common KV prefix of length $l_{\text{prefix}}$, FlashInfer decomposes the BSR into two matrices:

$$
\text{KV}_{\text{total}} = \text{BSR}(B_r^{\text{shared}}, B_c) \cup \text{BSR}(1, B_c)
$$

where $B_r^{\text{shared}} > 1$ groups queries sharing the prefix (enabling shared memory reuse) and the $(1, B_c)$ format handles unique suffixes (each query accesses its own KV independently). The attention states from both are composed via $\oplus$.

**Load-Balanced Scheduling:**

The cost of a work item (CTA assignment) with query length $l_q$ and KV length $l_{\text{kv}}$ is:

$$
\text{cost}(l_q, l_{\text{kv}}) = \alpha \cdot l_q + \beta \cdot l_{\text{kv}}
$$

The maximum KV chunk size per CTA is:

$$
L_{\text{kv}} \leftarrow \frac{\sum_i \lceil l_{\text{qo}}(i) / T_q \rceil \cdot l_{\text{kv}}(i)}{\#\text{CTA}}
$$

Work items are assigned to CTAs via a priority queue that greedily minimizes the maximum total cost across CTAs.

**Key Definitions:**

- BSR — Block Compressed Sparse Row: a sparse matrix format grouping nonzeros into dense blocks of size $(B_r, B_c)$
- Attention State — The tuple $[\mathbf{O}, \text{LSE}]$ that fully characterizes attention output and can be composed via $\oplus$
- CTA — Cooperative Thread Array (CUDA thread block)
- Functor — A user-defined CUDA function injected into the attention template at compile time via JIT
- Composable Formats — Multiple BSR matrices with different block sizes, enabling heterogeneous KV-cache access patterns within a single attention call

## Complexity

| Metric | Separate Kernels (RoPE + Attention) | FlashInfer JIT-Fused |
|--------|--------------------------------------|---------------------|
| Kernel launches | $\geq 2$ (transform + attention) | $1$ |
| HBM round-trips for Q,K | 2 (write transformed, read for attention) | 1 (transform in registers) |
| KV-cache format kernels needed | 1 per format (dense, paged, sparse) | 1 (unified BSR template) |
| Attention variant kernels needed | 1 per variant (RoPE, ALiBi, softcap, ...) | 1 (JIT-specialized) |

**Performance (kernel-level, H100 80GB SXM):**

| Setting | FlashInfer vs. FlashAttention |
|---------|------------------------------|
| Decode, constant seqlen | Higher bandwidth utilization |
| Decode, uniform seqlen | Higher bandwidth utilization |
| Decode, skewed seqlen | Significantly higher (load balancing) |
| Prefill, GQA | Higher FLOPs utilization |
| Fused RoPE attention | 1.6–3.7× bandwidth utilization vs. unfused |

**End-to-end (SGLang + FlashInfer vs. SGLang + Triton):**

| Metric | Improvement |
|--------|-------------|
| Inter-token latency (ITL) | 29–69% reduction |
| Long-context latency | 28–30% reduction |
| Parallel generation throughput | 13–17% speedup |
| Streaming-LLM ITL | 28–30% latency reduction |

**Memory:** BSR metadata is $O(\text{nnz\_blocks})$ integers. Composable format decomposition adds no extra KV storage — only index pointer arrays.

## Applicability

- **LLM serving with diverse attention variants**: Any production system supporting multiple model families (Llama with RoPE, Gemma with logits soft-cap, GPT with ALiBi, FlashSigmoid without softmax) benefits from a single JIT-compiled template rather than maintaining separate kernel implementations
- **Paged and variable-length KV-cache**: Serving frameworks like vLLM, SGLang, and MLC-Engine that use paged attention benefit from BSR's unified sparse format, which handles page tables, radix trees, and dense tensors through one kernel
- **Shared-prefix scenarios (parallel generation)**: When generating multiple completions for the same prompt, composable formats enable shared KV-cache access through high-bandwidth shared memory, yielding 13–17% throughput improvement
- **Long-context inference (Streaming-LLM)**: Fusing RoPE into the attention kernel eliminates a separate transformation pass, reducing ITL by 28–30% for million-token contexts
- **Speculative decoding and tree attention**: Tree-structured attention patterns (Medusa, SpecInfer) map naturally to sparse BSR masks within the unified format
- **Custom attention research**: Researchers can prototype new attention variants (new score functions, masking patterns, normalization schemes) by writing ~20 lines of CUDA functor code, without touching the FlashAttention mainloop

## Limitations

- **Forward pass only**: FlashInfer currently supports only forward attention; backward pass fusion for training requires separate customizable backward templates (planned future work)
- **JIT compilation latency**: First-time kernel compilation takes seconds; cached for reuse, but cold-start overhead may be noticeable for serving systems with many distinct configurations
- **CUDA-only functors**: User-defined transformations must be written in CUDA (not Python/Triton), requiring some GPU programming expertise
- **NVIDIA-specific**: Templates are CUTLASS-based, targeting sm75 through sm90a (Turing to Hopper). No AMD ROCm support.
- **No mainloop customization**: Functors can only modify pre/post-processing of Q, K, V, logits, and output — the core FlashAttention tiled loop structure is fixed. Fundamentally different attention patterns (e.g., linear attention) cannot be expressed
- **Block size constraints**: BSR block sizes must align with tensor core dimensions (multiples of 16); very fine-grained sparsity patterns (e.g., per-element masks) incur padding overhead
- **CUDAGraph compatibility**: While FlashInfer supports CUDAGraphs, the dynamic scheduler's `plan` phase runs on CPU and cannot be captured, requiring the Inspector-Executor pattern

## Implementation Notes

```python
# FlashInfer JIT-compiled attention variant example
# This shows how to define FlashSigmoid attention with ~20 lines of code

import torch
import flashinfer

# 1. Define the attention variant specification (CUDA functor code)
spec_decl = r"""
template <typename Params_, typename KernelTraits_>
struct FlashSigmoid {
    using Params = typename Params_;
    using KernelTraits = typename KernelTraits_;
    static constexpr bool use_softmax = false;  // Use sigmoid instead!
    float scale;
    float bias;

    // LogitsTransform functor: applied to S = Q @ K^T before normalization
    float LogitsTransform(float const& params, float
        logit, int batch_idx, int qo_idx, int kv_idx,
        int qo_head_idx, int kv_head_idx) {
        return 1. / (1. + exp(-(logit * scale + bias)));
    }
};
"""

# 2. Create attention specification with the custom variant
attn_spec = flashinfer.AttentionSpec(
    "FlashSigmoid",
    dtype_q=torch.float16, dtype_kv=torch.float16,
    head_dim=128, is_sparse=False,
    additional_vars=[("scale", "float"), ("bias", "float")],
    additional_tensors=[],
    spec_decl=spec_decl
)

# 3. JIT-compile and use (compilation cached after first call)
attn = flashinfer.AttentionWrapper(attn_spec, task_info, workspace)
attn.plan(seqlen_info)  # CPU-side scheduling (not captured by CUDAGraph)

# 4. Capture in CUDAGraph for zero-overhead execution
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    attn.run(q, k, v, output)

# 5. Replay at inference time
g.replay()


# Composable format example for shared-prefix parallel generation
# Two BSR matrices with different block sizes

# Shared prefix KV (block size (3,1) — 3 queries share same KV)
shared_wrapper = flashinfer.AttentionWrapper(
    attn_spec, task_info_shared, workspace,
    block_size=(3, 1)  # Larger Br enables shared memory reuse
)

# Unique suffix KV (block size (1,1) — each query has own KV)
unique_wrapper = flashinfer.AttentionWrapper(
    attn_spec, task_info_unique, workspace,
    block_size=(1, 1)  # Independent access per query
)

# Run both and compose attention states
state_shared = shared_wrapper.run(q, k_shared, v_shared)
state_unique = unique_wrapper.run(q, k_unique, v_unique)

# Attention state composition: O_final = compose(O_shared, O_unique)
# Uses the associative ⊕ operator on [O, LSE] pairs
output = flashinfer.compose_attention_states(state_shared, state_unique)


# Load-balanced scheduling (Algorithm 1 from paper)
def load_balanced_schedule(qo_lengths, kv_lengths, T_q, num_CTAs,
                           alpha=1.0, beta=1.0):
    """Assign work items to CTAs minimizing max cost."""
    # Compute max KV chunk size
    total_work = sum(
        ceil(qo_lengths[i] / T_q) * kv_lengths[i]
        for i in range(len(qo_lengths))
    )
    L_kv = total_work // num_CTAs

    # Split each query's KV into chunks of max size L_kv
    work_items = []
    for i, (lq, lkv) in enumerate(zip(qo_lengths, kv_lengths)):
        for chunk_start in range(0, lkv, L_kv):
            chunk_len = min(L_kv, lkv - chunk_start)
            work_items.append((i, chunk_len))

    # Sort by descending KV length (greedy bin-packing)
    work_items.sort(key=lambda w: w[1], reverse=True)

    # Priority queue: assign to least-loaded CTA
    import heapq
    Q = [(0, c) for c in range(num_CTAs)]  # (cost, cta_id)
    assignments = {}
    for w, lkv_w in work_items:
        cost, c = heapq.heappop(Q)
        new_cost = cost + alpha * qo_lengths[w] + beta * lkv_w
        assignments.setdefault(c, []).append(w)
        heapq.heappush(Q, (new_cost, c))

    return assignments
```

## References

- Ye, Z., Chen, L., Lai, R., Lin, W., Zhang, Y., Wang, S., Chen, T., Kasikci, B., Grover, V., Krishnamurthy, A., Ceze, L. "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving." MLSys 2025. arXiv:2501.01005
- Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024. arXiv:2307.08691
- Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., Dao, T. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." 2024. arXiv:2407.08691
- He, H., Guessous, D., Liang, Y., Dong, J. "FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention." 2024. https://pytorch.org/blog/flexattention/
- Liu, H., Abbeel, P. "Blockwise Parallel Transformer for Large Context Models." NeurIPS 2023.
- Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
- FlashInfer source code: https://github.com/flashinfer-ai/flashinfer
