"""
Multi-Query Associative Recall (MQAR) Dataset

Synthetic task for testing key-value association storage and retrieval.

Format of each sequence:
  [KV_PAIRS...] [SEP] [QUERIES...] [SEP] [ANSWERS...]

Example with 3 key-value pairs, vocab_size=16 (keys 0-7, values 8-15):
  Input:  k1 v1 k2 v2 k3 v3 SEP q1 q2 q3 SEP
  Target: -1 -1 -1 -1 -1 -1 -1  -1 -1 -1 -1  a1 a2 a3

Where q_i is one of {k1, k2, k3} and a_i = v_{j} where k_{j} = q_i.

The model must store all key-value pairs during the input phase,
then retrieve the correct value for each query during the output phase.
"""

import torch
from torch.utils.data import Dataset
import random


def generate_mqar_data(
    num_samples: int,
    num_kv_pairs: int,
    vocab_size: int,
    seq_len: int,
    seed: int = 42,
):
    """
    Generate MQAR dataset.

    Args:
        num_samples: Number of sequences to generate
        num_kv_pairs: Number of key-value pairs per sequence
        vocab_size: Total vocabulary size (keys from [0, vocab_size//2),
                    values from [vocab_size//2, vocab_size))
        seq_len: Total sequence length (should be >= 2*num_kv_pairs + 1 + num_kv_pairs + 1 + num_kv_pairs)
        seed: Random seed

    Returns:
        inputs: (num_samples, seq_len) tensor of input tokens
        targets: (num_samples, seq_len) tensor of target tokens (-100 = ignore)
        query_positions: (num_samples, num_kv_pairs) positions where answers should appear
    """
    rng = random.Random(seed)

    # Token allocation:
    # Keys: [0, num_keys)
    # Values: [num_keys, num_keys + num_values)
    # SEP token: vocab_size - 1
    # PAD token: vocab_size - 2
    num_keys = vocab_size // 2 - 1  # Reserve space for special tokens
    num_values = vocab_size // 2 - 1
    SEP = vocab_size - 1
    PAD = vocab_size - 2

    # Sequence structure:
    # [k1 v1 k2 v2 ... kN vN SEP q1 q2 ... qN SEP a1 a2 ... aN PAD...]
    # Minimum length needed: 2*N + 1 + N + 1 + N = 4*N + 2
    min_len = 4 * num_kv_pairs + 2
    assert seq_len >= min_len, f"seq_len {seq_len} too short for {num_kv_pairs} KV pairs (need >= {min_len})"

    inputs = torch.full((num_samples, seq_len), PAD, dtype=torch.long)
    targets = torch.full((num_samples, seq_len), -100, dtype=torch.long)

    for i in range(num_samples):
        # Generate unique keys
        keys = rng.sample(range(num_keys), num_kv_pairs)
        # Generate values (can repeat)
        values = [rng.randint(0, num_values - 1) + num_keys for _ in range(num_kv_pairs)]

        # Build input sequence
        pos = 0
        # KV pairs
        for k, v in zip(keys, values):
            inputs[i, pos] = k
            inputs[i, pos + 1] = v
            pos += 2

        # SEP
        inputs[i, pos] = SEP
        pos += 1

        # Queries (shuffled order of keys)
        query_order = list(range(num_kv_pairs))
        rng.shuffle(query_order)
        for qi in query_order:
            inputs[i, pos] = keys[qi]
            pos += 1

        # SEP
        inputs[i, pos] = SEP
        pos += 1

        # Answers (in same order as queries) — these are the targets
        answer_start = pos
        for qi in query_order:
            inputs[i, pos] = values[qi]
            targets[i, pos] = values[qi]  # Only predict answers
            pos += 1

        # Rest stays as PAD

    return inputs, targets


class MQARDataset(Dataset):
    """MQAR dataset wrapper."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
