"""
Z8 Cyclic Group Composition Dataset

Task: Given a sequence of elements from Z_8 = {0, 1, ..., 7},
compute the cumulative sum modulo 8.

Example:
    Input:  [3, 5, 2, 7, 1]
    Output: [3, 0, 2, 1, 2]  (cumulative sum mod 8)

    3 -> 3
    3+5=8 -> 0
    0+2=2 -> 2
    2+7=9 -> 1
    1+1=2 -> 2

This is the canonical task for circulant matrices because:
- Z_8 is a cyclic group
- Circulant matrices represent cyclic convolutions
- The circulant structure naturally captures modular arithmetic

Token vocabulary:
    0-7: group elements
    8: BOS (beginning of sequence)
    9: PAD

Total vocab size: 10
Number of classes: 8 (the group elements 0-7)
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple


# Special tokens
BOS_TOKEN = 8
PAD_TOKEN = 9
VOCAB_SIZE = 10
NUM_CLASSES = 8
IGNORE_INDEX = -100


class Z8Dataset(Dataset):
    """
    Synthetic dataset for Z_8 cyclic group composition.

    Generates sequences of random Z_8 elements with their cumulative sums.

    Args:
        num_samples: total number of samples to generate
        min_len: minimum sequence length (not counting BOS)
        max_len: maximum sequence length (not counting BOS)
        max_seq_len: maximum padded sequence length
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        num_samples: int = 10000,
        min_len: int = 16,
        max_len: int = 64,
        max_seq_len: int = 66,  # max_len + 1 (BOS) + 1 (buffer)
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.max_seq_len = max_seq_len

        rng = np.random.RandomState(seed)

        self.input_ids = []
        self.labels = []

        for _ in range(num_samples):
            # Random sequence length
            seq_len = rng.randint(min_len, max_len + 1)

            # Random Z_8 elements
            elements = rng.randint(0, 8, size=seq_len)

            # Cumulative sum mod 8
            cumsum = np.cumsum(elements) % 8

            # Build input: [BOS, e1, e2, ..., ek, PAD, PAD, ...]
            input_seq = np.full(max_seq_len, PAD_TOKEN, dtype=np.int64)
            input_seq[0] = BOS_TOKEN
            input_seq[1:1+seq_len] = elements

            # Build labels: [IGNORE, c1, c2, ..., ck, IGNORE, IGNORE, ...]
            label_seq = np.full(max_seq_len, IGNORE_INDEX, dtype=np.int64)
            label_seq[1:1+seq_len] = cumsum

            self.input_ids.append(torch.tensor(input_seq))
            self.labels.append(torch.tensor(label_seq))

        self.input_ids = torch.stack(self.input_ids)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


class Z8FixedLenDataset(Dataset):
    """
    Dataset with fixed sequence length (for controlled evaluation).

    Args:
        num_samples: number of samples
        seq_len: fixed sequence length
        max_seq_len: padded length
        seed: random seed
    """

    def __init__(
        self,
        num_samples: int = 2000,
        seq_len: int = 32,
        max_seq_len: int = 66,
        seed: int = 123,
    ):
        super().__init__()
        self.num_samples = num_samples

        rng = np.random.RandomState(seed)

        self.input_ids = []
        self.labels = []

        for _ in range(num_samples):
            elements = rng.randint(0, 8, size=seq_len)
            cumsum = np.cumsum(elements) % 8

            input_seq = np.full(max_seq_len, PAD_TOKEN, dtype=np.int64)
            input_seq[0] = BOS_TOKEN
            input_seq[1:1+seq_len] = elements

            label_seq = np.full(max_seq_len, IGNORE_INDEX, dtype=np.int64)
            label_seq[1:1+seq_len] = cumsum

            self.input_ids.append(torch.tensor(input_seq))
            self.labels.append(torch.tensor(label_seq))

        self.input_ids = torch.stack(self.input_ids)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def create_z8_dataloaders(
    num_samples: int = 10000,
    min_len: int = 16,
    max_len: int = 64,
    max_seq_len: int = 66,
    batch_size: int = 64,
    test_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Also creates a fixed-length test set at seq_len=32 for the success criterion.

    Returns:
        train_loader, val_loader, test_32_loader
    """
    # Main dataset (variable length)
    dataset = Z8Dataset(
        num_samples=num_samples,
        min_len=min_len,
        max_len=max_len,
        max_seq_len=max_seq_len,
        seed=seed,
    )

    # Split into train/val
    val_size = int(num_samples * test_fraction)
    train_size = num_samples - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Fixed-length test set at seq_len=32 (for success criterion)
    test_32_dataset = Z8FixedLenDataset(
        num_samples=2000,
        seq_len=32,
        max_seq_len=max_seq_len,
        seed=seed + 100,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_32_loader = DataLoader(
        test_32_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_32_loader
