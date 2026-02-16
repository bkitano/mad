"""
Synthetic associative recall dataset for Experiment 029.

Task: Given (k1, v1, k2, v2, ..., kn, vn, SEP, kq), output vq
where kq matches one of the keys k1..kn.

This is the canonical task for testing attention kernel quality:
it requires precise key-value binding that depends on the kernel's
ability to distinguish similar keys.

From MVE spec:
- 5K sequences of length 64
- Vocabulary size 16
- 8 key-value pairs per sequence

Sequence format:
  [k1, v1, k2, v2, ..., k8, v8, SEP, kq, PAD, PAD, ...]
  Target at query position: vq (the value associated with kq)

Special tokens:
  PAD = 0
  SEP = vocab_size + 1  (index 17)
"""

import torch
from torch.utils.data import Dataset
import random


class AssociativeRecallDataset(Dataset):
    """
    Generates synthetic associative recall sequences.

    Each sequence has n_pairs key-value pairs followed by a query.
    The model must output the value corresponding to the queried key.
    """

    def __init__(self, n_samples: int, seq_len: int, vocab_size: int,
                 n_pairs: int, seed: int = 42):
        """
        Args:
            n_samples: Number of sequences to generate
            seq_len: Total sequence length (padded)
            vocab_size: Number of content tokens (keys and values drawn from 1..vocab_size)
            n_pairs: Number of key-value pairs per sequence
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_pairs = n_pairs

        # Special tokens (outside content vocab range)
        self.pad_token = 0
        self.sep_token = vocab_size + 1

        # Total vocabulary: PAD(0) + content(1..vocab_size) + SEP(vocab_size+1)
        self.total_vocab = vocab_size + 2

        # Minimum sequence length needed: 2*n_pairs (KV pairs) + 1 (SEP) + 1 (query)
        self.min_len = 2 * n_pairs + 2
        assert seq_len >= self.min_len, \
            f"seq_len {seq_len} too short for {n_pairs} pairs (need {self.min_len})"

        # Pre-generate all data
        self.data, self.targets, self.query_positions = self._generate(seed)

    def _generate(self, seed: int):
        """Generate all sequences."""
        rng = random.Random(seed)

        all_seqs = []
        all_targets = []
        all_query_pos = []

        for _ in range(self.n_samples):
            # Sample n_pairs unique keys from content vocab (1..vocab_size)
            keys = rng.sample(range(1, self.vocab_size + 1), self.n_pairs)

            # Sample values from content vocab (can repeat)
            values = [rng.randint(1, self.vocab_size) for _ in range(self.n_pairs)]

            # Pick a random query key
            query_idx = rng.randint(0, self.n_pairs - 1)
            query_key = keys[query_idx]
            target_value = values[query_idx]

            # Build sequence: k1 v1 k2 v2 ... kn vn SEP kq PAD PAD ...
            seq = []
            for k, v in zip(keys, values):
                seq.append(k)
                seq.append(v)
            seq.append(self.sep_token)
            seq.append(query_key)

            query_pos = len(seq) - 1  # Position of the query key

            # Pad to seq_len
            while len(seq) < self.seq_len:
                seq.append(self.pad_token)

            all_seqs.append(seq)
            all_targets.append(target_value)
            all_query_pos.append(query_pos)

        data = torch.tensor(all_seqs, dtype=torch.long)
        targets = torch.tensor(all_targets, dtype=torch.long)
        query_positions = torch.tensor(all_query_pos, dtype=torch.long)

        return data, targets, query_positions

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
            seq: [seq_len] token indices
            target: scalar (the expected value token)
            query_pos: scalar (position of query key in sequence)
        """
        return self.data[idx], self.targets[idx], self.query_positions[idx]


def create_datasets(n_train: int, n_test: int, seq_len: int, vocab_size: int,
                    n_pairs: int) -> tuple:
    """
    Create train and test datasets.

    Returns:
        (train_dataset, test_dataset)
    """
    train_ds = AssociativeRecallDataset(
        n_samples=n_train, seq_len=seq_len, vocab_size=vocab_size,
        n_pairs=n_pairs, seed=42
    )
    test_ds = AssociativeRecallDataset(
        n_samples=n_test, seq_len=seq_len, vocab_size=vocab_size,
        n_pairs=n_pairs, seed=123
    )
    return train_ds, test_ds
