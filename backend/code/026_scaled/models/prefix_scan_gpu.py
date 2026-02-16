"""
Optimized Prefix Scan for GPU (Dense SSM Recurrence)

Hillis-Steele inclusive scan with associative operator:
    (A, b) ⊙ (A', b') = (A @ A', A @ b' + b)

h_t = A_t h_{t-1} + b_t where A_t ∈ R^{n×n}, b_t ∈ R^n

Work: O(Tn³ log T) | Depth: O(log T) | Memory: O(Tn²)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PrefixScanGPU:
    """
    GPU-optimized prefix scan for dense SSM recurrence.

    Features:
    - Mixed precision (FP16/BF16) support
    - In-place updates where possible
    - Batched GEMM optimization
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
    ):
        self.dtype = dtype

    def forward(
        self,
        A: torch.Tensor,  # (T, n, n)
        b: torch.Tensor,  # (T, n)
        h0: Optional[torch.Tensor] = None,  # (n,)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Hillis-Steele inclusive prefix scan.

        Returns:
            h: (T, n) - Hidden states (final scan result)
            stats: dict - Performance statistics
        """
        T, n, _ = A.shape
        device = A.device
        dtype = self.dtype

        # Convert to target dtype
        if A.dtype != dtype:
            A = A.to(dtype)
            b = b.to(dtype)
            if h0 is not None:
                h0 = h0.to(dtype)

        gemm_count = 0

        # Initialize scan arrays
        A_scan = A.clone()
        b_scan = b.clone()

        # Incorporate h0 into the first element
        if h0 is not None:
            b_scan[0] = torch.mv(A_scan[0], h0) + b_scan[0]
            A_scan[0] = torch.eye(n, device=device, dtype=dtype)
            gemm_count += 1

        # Hillis-Steele scan: stride doubling
        stride = 1
        num_levels = 0

        while stride < T:
            num_levels += 1

            # Indices to update: all positions >= stride
            update_count = T - stride
            if update_count <= 0:
                stride *= 2
                continue

            # Indices: stride, stride+1, ..., T-1
            # Source:  0, 1, ..., T-stride-1
            idx_dst = torch.arange(stride, T, device=device, dtype=torch.long)
            idx_src = idx_dst - stride

            # Fetch current and predecessor values
            A_curr = A_scan[idx_dst]  # (update_count, n, n)
            b_curr = b_scan[idx_dst]  # (update_count, n)
            A_pred = A_scan[idx_src]  # (update_count, n, n)
            b_pred = b_scan[idx_src]  # (update_count, n)

            # Associative operator: (A_curr, b_curr) ⊙ (A_pred, b_pred)
            # A_new = A_curr @ A_pred
            A_new = torch.bmm(A_curr, A_pred)  # (update_count, n, n)
            gemm_count += update_count

            # b_new = A_curr @ b_pred + b_curr
            b_new = torch.bmm(A_curr, b_pred.unsqueeze(-1)).squeeze(-1) + b_curr  # (update_count, n)
            gemm_count += update_count

            # Write back (need clone to avoid in-place mutation issues)
            A_scan = A_scan.clone()
            b_scan = b_scan.clone()
            A_scan[idx_dst] = A_new
            b_scan[idx_dst] = b_new

            stride *= 2

        stats = {
            'gemm_count': gemm_count,
            'num_levels': num_levels,
            'work_theoretical': T * n**3 * math.ceil(math.log2(T)),  # O(Tn³ log T)
            'depth_theoretical': math.ceil(math.log2(T)),
        }

        # Final result is in b_scan (inclusive scan output)
        return b_scan, stats


def prefix_scan_gpu(
    A: torch.Tensor,
    b: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float16,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, dict]:
    """
    Functional interface for GPU prefix scan.

    Args:
        A: (T, n, n) - State transition matrices
        b: (T, n) - Input vectors
        h0: (n,) - Initial state (optional)
        dtype: torch dtype for computation
        return_stats: If True, return (h, stats) tuple

    Returns:
        h: (T, n) - Computed hidden states
        stats: (optional) Performance statistics
    """
    scan = PrefixScanGPU(dtype=dtype)
    h, stats = scan.forward(A, b, h0)

    if return_stats:
        return h, stats
    return h


@torch.compile(mode="max-autotune")
def prefix_scan_compiled(
    A: torch.Tensor,
    b: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """torch.compile-optimized prefix scan."""
    return prefix_scan_gpu(A, b, h0, return_stats=False)


def sequential_scan_gpu(
    A: torch.Tensor,
    b: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Sequential scan baseline (O(T) depth, O(Tn²) work).

    This is the work-optimal algorithm but not parallelizable.
    Used as ground truth for correctness testing.
    """
    T, n, _ = A.shape
    device = A.device

    if A.dtype != dtype:
        A = A.to(dtype)
        b = b.to(dtype)
        if h0 is not None:
            h0 = h0.to(dtype)

    h = torch.zeros(T, n, device=device, dtype=dtype)
    h_prev = torch.zeros(n, device=device, dtype=dtype) if h0 is None else h0

    for t in range(T):
        h[t] = torch.mv(A[t], h_prev) + b[t]
        h_prev = h[t]

    return h


if __name__ == "__main__":
    # Test on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    T, n = 1024, 64
    print(f"Testing Prefix Scan GPU on {device}")
    print(f"T={T}, n={n}, dtype={dtype}")

    # Generate test data
    A = torch.randn(T, n, n, device=device, dtype=torch.float32) * (0.95 / math.sqrt(n))
    b = torch.randn(T, n, device=device, dtype=torch.float32)

    # Run scan
    h, stats = prefix_scan_gpu(A, b, dtype=dtype, return_stats=True)

    print(f"\nResults:")
    print(f"  Output shape: {h.shape}")
    print(f"  GEMM count: {stats['gemm_count']}")
    print(f"  Levels: {stats['num_levels']}")
    print(f"  Theoretical work: {stats['work_theoretical']:.2e}")
    print(f"  Output norm: {h.norm().item():.4f}")

    # Verify against sequential
    h_seq = sequential_scan_gpu(A, b, dtype=dtype)
    error = (h - h_seq).abs().max().item()
    print(f"  Max error vs sequential: {error:.2e}")

    # Benchmark
    if device == "cuda":
        import time
        torch.cuda.synchronize()

        # Warmup
        for _ in range(10):
            _ = prefix_scan_gpu(A, b, dtype=dtype)
        torch.cuda.synchronize()

        # Timing
        n_trials = 100
        start = time.perf_counter()
        for _ in range(n_trials):
            _ = prefix_scan_gpu(A, b, dtype=dtype)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"\nBenchmark ({n_trials} trials):")
        print(f"  Time per call: {elapsed/n_trials*1000:.2f} ms")
        print(f"  Throughput: {T*n*n*stats['gemm_count']/(elapsed/n_trials)/1e12:.2f} TFLOP/s")
