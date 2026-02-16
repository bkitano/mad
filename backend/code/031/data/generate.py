"""
Multi-Query Associative Recall (MQAR) Dataset Generator

MQAR tests whether a model can store and retrieve key-value associations.
Each sequence contains:
  1. A set of key-value pairs to memorize
  2. Query keys that the model must map to the corresponding values

Format (proposal: 4 KV pairs, seq_len 64, vocab 16):
  [K1, V1, K2, V2, K3, V3, K4, V4, PAD..., Q1, Q2, Q3, Q4]
  Target:                                   [V1, V2, V3, V4] (at query positions)

Keys are drawn from vocab[0:vocab_size//2], values from vocab[vocab_size//2:vocab_size].
This separation ensures no key-value confusion.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class MQARDataset(Dataset):
    """
    Multi-Query Associative Recall dataset.

    Args:
        num_samples: Number of sequences
        num_kv_pairs: Number of key-value pairs per sequence
        seq_len: Total sequence length
        vocab_size: Vocabulary size (keys from first half, values from second half)
        seed: Random seed
    """

    def __init__(
        self,
        num_samples: int = 10000,
        num_kv_pairs: int = 4,
        seq_len: int = 64,
        vocab_size: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_kv_pairs = num_kv_pairs
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Split vocab: keys from [0, vocab_size//2), values from [vocab_size//2, vocab_size)
        self.num_keys = vocab_size // 2
        self.num_values = vocab_size - self.num_keys

        # Generate all data upfront
        rng = np.random.RandomState(seed)
        self.sequences, self.targets, self.query_positions = self._generate_all(rng)

    def _generate_all(self, rng: np.random.RandomState):
        """Generate all MQAR sequences."""
        sequences = []
        targets = []
        query_positions_list = []

        for _ in range(self.num_samples):
            seq, tgt, qpos = self._generate_one(rng)
            sequences.append(seq)
            targets.append(tgt)
            query_positions_list.append(qpos)

        return (
            torch.tensor(np.array(sequences), dtype=torch.long),
            torch.tensor(np.array(targets), dtype=torch.long),
            torch.tensor(np.array(query_positions_list), dtype=torch.long),
        )

    def _generate_one(self, rng: np.random.RandomState):
        """
        Generate a single MQAR sequence.

        Returns:
            seq: (seq_len,) token indices
            target: (seq_len,) target values (-1 for non-query positions)
            query_positions: (num_kv_pairs,) indices of query positions
        """
        seq = np.zeros(self.seq_len, dtype=np.int64)
        target = np.full(self.seq_len, -1, dtype=np.int64)  # -1 = ignore

        # Generate unique keys and random values
        keys = rng.choice(self.num_keys, size=self.num_kv_pairs, replace=False)
        values = rng.randint(0, self.num_values, size=self.num_kv_pairs) + self.num_keys

        # Place KV pairs at the start: [K1, V1, K2, V2, ...]
        kv_section_len = 2 * self.num_kv_pairs
        for i in range(self.num_kv_pairs):
            seq[2 * i] = keys[i]
            seq[2 * i + 1] = values[i]

        # Fill padding region with random tokens (noise)
        query_section_start = self.seq_len - self.num_kv_pairs
        for i in range(kv_section_len, query_section_start):
            seq[i] = rng.randint(0, self.vocab_size)

        # Place queries at the end (shuffled order)
        query_order = rng.permutation(self.num_kv_pairs)
        query_positions = np.arange(query_section_start, self.seq_len)

        for i, qi in enumerate(query_order):
            seq[query_positions[i]] = keys[qi]
            target[query_positions[i]] = values[qi]

        return seq, target, query_positions

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.query_positions[idx]


def generate_mqar_dataset(
    num_train: int = 8000,
    num_val: int = 1000,
    num_test: int = 1000,
    num_kv_pairs: int = 4,
    seq_len: int = 64,
    vocab_size: int = 16,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Generate MQAR train/val/test dataloaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = MQARDataset(num_train, num_kv_pairs, seq_len, vocab_size, seed=seed)
    val_dataset = MQARDataset(num_val, num_kv_pairs, seq_len, vocab_size, seed=seed + 1)
    test_dataset = MQARDataset(num_test, num_kv_pairs, seq_len, vocab_size, seed=seed + 2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
