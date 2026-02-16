"""
MQAR (Multi-Query Associative Recall) Dataset Generator

Task: Given a sequence of key-value pairs followed by queries, recall the value
associated with each queried key.

Format (for num_kv_pairs=16, seq_len=512, vocab_size=64):
  [K1, V1, K2, V2, ..., K16, V16, NOISE..., Q1, Q2, ..., Q16]

  Target at query positions: [V1, V2, ..., V16]
  Target at non-query positions: -1 (ignored in loss)

Keys: sampled from [1, vocab_size//2)  (no zero, to avoid confusion with padding)
Values: sampled from [vocab_size//2, vocab_size)
Noise: random tokens from [1, vocab_size)
Queries: the keys K1..K16, shuffled
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class MQARDataset(Dataset):
    """Multi-Query Associative Recall Dataset."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        num_kv_pairs: int,
        vocab_size: int,
        seed: int = 42,
    ):
        """
        Args:
            num_samples: Number of sequences to generate
            seq_len: Total sequence length (must be > 2 * num_kv_pairs + num_kv_pairs)
            num_kv_pairs: Number of key-value pairs per sequence
            vocab_size: Total vocabulary size (keys from first half, values from second half)
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_kv_pairs = num_kv_pairs
        self.vocab_size = vocab_size

        assert seq_len >= 3 * num_kv_pairs, (
            f"seq_len ({seq_len}) must be >= 3 * num_kv_pairs ({3 * num_kv_pairs})"
        )

        self.inputs, self.targets, self.query_mask = self._generate(seed)

    def _generate(self, seed: int):
        """Generate all samples."""
        rng = np.random.RandomState(seed)

        half_vocab = self.vocab_size // 2  # Split vocab: keys [1, half), values [half, vocab)

        inputs = np.zeros((self.num_samples, self.seq_len), dtype=np.int64)
        targets = np.full((self.num_samples, self.seq_len), -1, dtype=np.int64)
        query_mask = np.zeros((self.num_samples, self.seq_len), dtype=np.bool_)

        for i in range(self.num_samples):
            # Generate unique keys and random values
            keys = rng.choice(np.arange(1, half_vocab), size=self.num_kv_pairs, replace=False)
            values = rng.randint(half_vocab, self.vocab_size, size=self.num_kv_pairs)

            # Place KV pairs at the start: [K1, V1, K2, V2, ...]
            kv_section_len = 2 * self.num_kv_pairs
            for j in range(self.num_kv_pairs):
                inputs[i, 2 * j] = keys[j]
                inputs[i, 2 * j + 1] = values[j]

            # Fill noise section
            noise_start = kv_section_len
            noise_end = self.seq_len - self.num_kv_pairs
            noise_len = noise_end - noise_start
            if noise_len > 0:
                inputs[i, noise_start:noise_end] = rng.randint(1, self.vocab_size, size=noise_len)

            # Place queries at the end (shuffled order)
            query_order = rng.permutation(self.num_kv_pairs)
            query_start = self.seq_len - self.num_kv_pairs
            for j, q_idx in enumerate(query_order):
                pos = query_start + j
                inputs[i, pos] = keys[q_idx]
                targets[i, pos] = values[q_idx]
                query_mask[i, pos] = True

        return (
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(query_mask),
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.query_mask[idx]


def generate_mqar_data(config: dict, seed: int = 42):
    """Generate train/val/test MQAR datasets from config dict."""
    data_cfg = config["data"]

    train_ds = MQARDataset(
        num_samples=data_cfg["num_train"],
        seq_len=data_cfg["seq_len"],
        num_kv_pairs=data_cfg["num_kv_pairs"],
        vocab_size=data_cfg["vocab_size"],
        seed=seed,
    )
    val_ds = MQARDataset(
        num_samples=data_cfg["num_val"],
        seq_len=data_cfg["seq_len"],
        num_kv_pairs=data_cfg["num_kv_pairs"],
        vocab_size=data_cfg["vocab_size"],
        seed=seed + 1,
    )
    test_ds = MQARDataset(
        num_samples=data_cfg["num_test"],
        seq_len=data_cfg["seq_len"],
        num_kv_pairs=data_cfg["num_kv_pairs"],
        vocab_size=data_cfg["vocab_size"],
        seed=seed + 2,
    )

    # Length generalization test set (T=1024)
    gen_ds = None
    if data_cfg.get("gen_seq_len"):
        gen_ds = MQARDataset(
            num_samples=data_cfg["num_test"],
            seq_len=data_cfg["gen_seq_len"],
            num_kv_pairs=data_cfg["num_kv_pairs"],
            vocab_size=data_cfg["vocab_size"],
            seed=seed + 3,
        )

    return train_ds, val_ds, test_ds, gen_ds
