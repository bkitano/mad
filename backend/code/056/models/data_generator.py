"""
Synthetic Data Generator for Document-Packed Sequences.

Generates sequences with controlled document boundaries, where alpha=0
at boundary positions creates block-diagonal structure in the gate mask.

The generator produces packed sequences with varying average document lengths
to test the tile-skip efficiency across different document packing regimes.
"""

import torch
import math


def generate_document_packed_data(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    dk: int,
    dv: int,
    avg_doc_len: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic document-packed data for GLA intra-chunk benchmarking.

    Args:
        batch_size: number of sequences in batch
        num_heads: number of attention heads
        seq_len: total packed sequence length (must be divisible by chunk_size)
        dk: key/query dimension
        dv: value dimension
        avg_doc_len: average document length (controls boundary frequency)
        device: computation device
        dtype: data type
        seed: random seed for reproducibility

    Returns:
        dict with:
            Q: (B, H, T, dk) query tensor
            K: (B, H, T, dk) key tensor
            V: (B, H, T, dv) value tensor
            alpha: (B, H, T) gating values (0 at doc boundaries, ~1 elsewhere)
            doc_boundaries: list of boundary positions per (batch, head)
    """
    torch.manual_seed(seed)

    # Generate Q, K, V with random normal initialization
    Q = torch.randn(batch_size, num_heads, seq_len, dk, device=device, dtype=dtype) * (dk ** -0.5)
    K = torch.randn(batch_size, num_heads, seq_len, dk, device=device, dtype=dtype) * (dk ** -0.5)
    V = torch.randn(batch_size, num_heads, seq_len, dv, device=device, dtype=dtype) * 0.1

    # Generate alpha (gating values)
    # Normal positions: alpha ~ Uniform(0.9, 1.0) (close to 1, slight decay)
    # Document boundaries: alpha = 0.0 (full state reset)
    alpha = torch.rand(batch_size, num_heads, seq_len, device=device, dtype=dtype) * 0.1 + 0.9

    # Place document boundaries at random positions with controlled avg spacing
    all_boundaries = []
    for b in range(batch_size):
        batch_boundaries = []
        for h in range(num_heads):
            boundaries = []
            pos = 0
            while pos < seq_len:
                # Sample document length from exponential distribution
                # Exponential distribution with mean = avg_doc_len
                doc_len = max(1, int(torch.empty(1).exponential_(1.0 / avg_doc_len).item()))
                pos += doc_len
                if pos < seq_len:
                    boundaries.append(pos)
                    alpha[b, h, pos] = 0.0  # Document boundary
            batch_boundaries.append(boundaries)
        all_boundaries.append(batch_boundaries)

    return {
        'Q': Q,
        'K': K,
        'V': V,
        'alpha': alpha,
        'doc_boundaries': all_boundaries,
    }


def count_skippable_tiles(
    alpha: torch.Tensor,
    chunk_size: int = 128,
    sub_chunk_size: int = 16,
) -> dict:
    """
    Count the number of tiles that can be skipped with FlashMask tile-skip.

    Returns:
        dict with:
            total_tiles: total number of tiles across all chunks
            causal_skip: tiles skipped by basic causal mask (upper triangular)
            flashmask_skip: additional tiles skipped by FlashMask (cross-document)
            computed_tiles: tiles actually computed
            skip_fraction: total fraction of tiles skipped
    """
    B, H, T = alpha.shape
    num_chunks = T // chunk_size
    Ns = chunk_size // sub_chunk_size

    total_tiles = 0
    causal_skip = 0
    flashmask_skip = 0
    computed_tiles = 0

    for b in range(B):
        for h in range(H):
            for n in range(num_chunks):
                chunk_start = n * chunk_size

                for qi in range(Ns):
                    for kj in range(Ns):
                        total_tiles += 1

                        if kj > qi:
                            # Basic causal skip (upper triangular)
                            causal_skip += 1
                            continue

                        # Check for cross-document skip
                        # A tile (qi, kj) with kj < qi is fully masked if
                        # ALL keys in kj's sub-chunk have their document end
                        # before the start of qi's sub-chunk

                        if kj < qi:
                            qi_start = chunk_start + qi * sub_chunk_size
                            kj_start = chunk_start + kj * sub_chunk_size
                            kj_end = kj_start + sub_chunk_size

                            # Check if there's a boundary between kj and qi sub-chunks
                            # by checking if any alpha in between is 0
                            alpha_between = alpha[b, h, kj_start:qi_start + sub_chunk_size]

                            # For all key positions in kj's sub-chunk, find if their
                            # document ends before qi_start
                            can_skip = True
                            for k_pos in range(kj_start, kj_end):
                                # Find next boundary after k_pos
                                next_boundary = qi_start + sub_chunk_size  # default: no boundary
                                for t in range(k_pos + 1, qi_start + sub_chunk_size):
                                    if t < T and alpha[b, h, t].item() < 1e-6:
                                        next_boundary = t
                                        break
                                if next_boundary > qi_start:
                                    can_skip = False
                                    break

                            if can_skip:
                                flashmask_skip += 1
                                continue

                        computed_tiles += 1

    return {
        'total_tiles': total_tiles,
        'causal_skip': causal_skip,
        'flashmask_skip': flashmask_skip,
        'computed_tiles': computed_tiles,
        'skip_fraction': (causal_skip + flashmask_skip) / max(total_tiles, 1),
        'causal_skip_fraction': causal_skip / max(total_tiles, 1),
        'flashmask_extra_skip_fraction': flashmask_skip / max(total_tiles, 1),
    }
