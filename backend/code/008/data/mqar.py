"""
Multi-Query Associative Recall (MQAR) Dataset

Task: Given a sequence of key-value pairs followed by query keys,
recall the associated values.

Format: k1 v1 k2 v2 ... kN vN [SEP] q1 q2 ... qN
Target: -1 -1 -1 -1 ... -1 -1 [SEP] a1 a2 ... aN

Where qi is some key from the input, and ai is its associated value.

This tests the model's ability to store and retrieve key-value associations,
which directly probes state capacity.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class MQARDataset(Dataset):
    """
    Multi-Query Associative Recall dataset.

    Args:
        num_samples: Number of sequences to generate
        num_kv_pairs: Number of key-value pairs (N)
        vocab_size: Size of vocabulary for keys and values
        seq_len: Total sequence length (padded)
        seed: Random seed for reproducibility
    """

    # Special tokens
    SEP_TOKEN = 0  # Separator between KV pairs and queries
    PAD_TOKEN = 1  # Padding token
    IGNORE_INDEX = -100  # For loss masking

    def __init__(
        self,
        num_samples: int = 5000,
        num_kv_pairs: int = 8,
        vocab_size: int = 64,
        seq_len: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_kv_pairs = num_kv_pairs
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        # Vocab layout:
        # 0 = SEP, 1 = PAD, 2..vocab_size-1 = data tokens
        # Keys and values are drawn from [2, vocab_size)
        self.data_vocab_start = 2
        self.data_vocab_size = vocab_size - 2  # Usable tokens for KV data

        # Minimum seq len needed: 2*N (KV pairs) + 1 (SEP) + N (queries) = 3N + 1
        min_seq_len = 3 * num_kv_pairs + 1
        assert seq_len >= min_seq_len, (
            f"seq_len {seq_len} too short for {num_kv_pairs} KV pairs "
            f"(need at least {min_seq_len})"
        )

        # Generate all data upfront
        self.inputs, self.targets = self._generate_data(seed)

    def _generate_data(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all MQAR sequences."""
        rng = np.random.RandomState(seed)

        all_inputs = torch.full(
            (self.num_samples, self.seq_len), self.PAD_TOKEN, dtype=torch.long
        )
        all_targets = torch.full(
            (self.num_samples, self.seq_len), self.IGNORE_INDEX, dtype=torch.long
        )

        for i in range(self.num_samples):
            # Sample unique keys (no repeats in the KV section)
            keys = rng.choice(
                self.data_vocab_size, size=self.num_kv_pairs, replace=False
            )
            keys = keys + self.data_vocab_start  # Shift to data vocab range

            # Sample values (can repeat)
            values = rng.randint(
                self.data_vocab_start,
                self.data_vocab_start + self.data_vocab_size,
                size=self.num_kv_pairs,
            )

            # Build KV section: k1 v1 k2 v2 ... kN vN
            kv_section = np.empty(2 * self.num_kv_pairs, dtype=np.int64)
            kv_section[0::2] = keys
            kv_section[1::2] = values

            # Query section: random permutation of the keys
            query_order = rng.permutation(self.num_kv_pairs)
            query_keys = keys[query_order]
            query_answers = values[query_order]

            # Assemble input: [KV pairs] [SEP] [queries] [PAD...]
            pos = 0
            all_inputs[i, pos : pos + 2 * self.num_kv_pairs] = torch.from_numpy(
                kv_section
            )
            pos += 2 * self.num_kv_pairs

            all_inputs[i, pos] = self.SEP_TOKEN
            pos += 1

            all_inputs[i, pos : pos + self.num_kv_pairs] = torch.from_numpy(
                query_keys
            )

            # Targets: only the query answers are supervised
            target_start = 2 * self.num_kv_pairs + 1  # After SEP
            all_targets[i, target_start : target_start + self.num_kv_pairs] = (
                torch.from_numpy(query_answers)
            )

        return all_inputs, all_targets

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def create_mqar_dataloaders(
    num_kv_pairs: int = 8,
    vocab_size: int = 64,
    seq_len: int = 128,
    num_train: int = 4000,
    num_test: int = 1000,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders for MQAR."""
    train_dataset = MQARDataset(
        num_samples=num_train,
        num_kv_pairs=num_kv_pairs,
        vocab_size=vocab_size,
        seq_len=seq_len,
        seed=seed,
    )
    test_dataset = MQARDataset(
        num_samples=num_test,
        num_kv_pairs=num_kv_pairs,
        vocab_size=vocab_size,
        seq_len=seq_len,
        seed=seed + 1,  # Different seed for test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, test_loader
