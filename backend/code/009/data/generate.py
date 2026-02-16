"""
Multi-Query Associative Recall (MQAR) data generator.

MQAR task (from proposal 009 MVE):
  - Store n_kv_pairs key-value associations in a sequence
  - Then query n_queries of those keys and expect the model to output the values

Sequence format:
  [k1, v1, k2, v2, ..., kN, vN, SEP, q1, _, q2, _, ...]

Where:
  - k_i are keys from key vocabulary
  - v_i are values from value vocabulary
  - SEP is a separator token
  - q_i are query keys (subset of {k1, ..., kN})
  - _ are blank tokens (model must predict v_i at these positions)

Vocabulary layout:
  0: PAD
  1: SEP
  2: BLANK (placeholder at answer positions)
  3 .. 3+vocab_size-1: content tokens (used for both keys and values)

The target at answer positions is the value associated with the preceding query key.
All other positions have target = -100 (ignored in loss).

This tests the readout precision of linear attention:
  - The model must store KV pairs in its state S_t
  - At query time, it must extract the correct value from S_t
  - The sigmoid gate should improve extraction precision by providing
    a data-dependent nonlinear filter on the readout
"""

import torch
from torch.utils.data import TensorDataset


def generate_mqar_data(
    n_samples: int = 10000,
    n_kv_pairs: int = 4,
    n_queries: int = 2,
    vocab_size: int = 16,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Generate MQAR dataset.

    Args:
        n_samples: Number of sequences to generate
        n_kv_pairs: Number of key-value pairs to store
        n_queries: Number of queries (must be <= n_kv_pairs)
        vocab_size: Size of the content vocabulary (keys and values drawn from this)
        seed: Random seed for reproducibility

    Returns:
        input_ids: (n_samples, seq_len) integer token indices
        targets: (n_samples, seq_len) target token indices (-100 for non-answer positions)
        total_vocab_size: Total vocabulary size including special tokens
    """
    assert n_queries <= n_kv_pairs, "Can't query more keys than stored"
    assert vocab_size >= n_kv_pairs, "Need enough vocab for unique keys"

    torch.manual_seed(seed)

    # Special tokens
    PAD = 0
    SEP = 1
    BLANK = 2
    CONTENT_OFFSET = 3  # Content tokens start at index 3
    total_vocab_size = CONTENT_OFFSET + vocab_size

    # Sequence layout:
    # [k1, v1, k2, v2, ..., kN, vN, SEP, q1, BLANK, q2, BLANK, ...]
    kv_section_len = 2 * n_kv_pairs  # k1, v1, k2, v2, ...
    query_section_len = 2 * n_queries  # q1, BLANK, q2, BLANK, ...
    seq_len = kv_section_len + 1 + query_section_len  # +1 for SEP

    input_ids = torch.full((n_samples, seq_len), PAD, dtype=torch.long)
    targets = torch.full((n_samples, seq_len), -100, dtype=torch.long)

    for i in range(n_samples):
        # Generate unique keys
        keys = torch.randperm(vocab_size)[:n_kv_pairs] + CONTENT_OFFSET
        # Generate random values (can repeat)
        values = torch.randint(0, vocab_size, (n_kv_pairs,)) + CONTENT_OFFSET

        # Fill KV section: k1, v1, k2, v2, ...
        for j in range(n_kv_pairs):
            input_ids[i, 2 * j] = keys[j]
            input_ids[i, 2 * j + 1] = values[j]

        # SEP token
        input_ids[i, kv_section_len] = SEP

        # Query section: select n_queries random keys to query
        query_indices = torch.randperm(n_kv_pairs)[:n_queries]
        query_keys = keys[query_indices]
        query_values = values[query_indices]

        for j in range(n_queries):
            pos = kv_section_len + 1 + 2 * j
            input_ids[i, pos] = query_keys[j]      # Query key
            input_ids[i, pos + 1] = BLANK           # Blank (model predicts here)
            targets[i, pos + 1] = query_values[j]   # Target is the value

    return input_ids, targets, total_vocab_size


def create_mqar_datasets(
    n_train: int = 10000,
    n_test: int = 2000,
    n_kv_pairs: int = 4,
    n_queries: int = 2,
    vocab_size: int = 16,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset, int]:
    """Create train and test MQAR datasets.

    Returns:
        train_dataset, test_dataset, total_vocab_size
    """
    train_inputs, train_targets, total_vocab = generate_mqar_data(
        n_samples=n_train,
        n_kv_pairs=n_kv_pairs,
        n_queries=n_queries,
        vocab_size=vocab_size,
        seed=seed,
    )

    test_inputs, test_targets, _ = generate_mqar_data(
        n_samples=n_test,
        n_kv_pairs=n_kv_pairs,
        n_queries=n_queries,
        vocab_size=vocab_size,
        seed=seed + 1000,  # Different seed for test set
    )

    train_ds = TensorDataset(train_inputs, train_targets)
    test_ds = TensorDataset(test_inputs, test_targets)

    return train_ds, test_ds, total_vocab


if __name__ == '__main__':
    """Quick sanity check."""
    inputs, targets, vocab_size = generate_mqar_data(
        n_samples=5, n_kv_pairs=4, n_queries=2, vocab_size=16, seed=42
    )

    print(f"Vocab size: {vocab_size}")
    print(f"Sequence length: {inputs.shape[1]}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")

    print("\nExample sequence:")
    print(f"  Input:  {inputs[0].tolist()}")
    print(f"  Target: {targets[0].tolist()}")

    # Verify targets
    answer_positions = (targets[0] != -100).nonzero().squeeze()
    print(f"\n  Answer positions: {answer_positions.tolist()}")
    for pos in answer_positions:
        query_key = inputs[0, pos - 1].item()
        expected_val = targets[0, pos].item()
        # Find the key in the KV section
        for j in range(4):
            if inputs[0, 2 * j].item() == query_key:
                stored_val = inputs[0, 2 * j + 1].item()
                assert stored_val == expected_val, f"Mismatch at pos {pos}"
                print(f"  Query key={query_key} -> value={expected_val} (stored at pos {2*j+1}) OK")
                break

    print("\nAll checks passed!")
