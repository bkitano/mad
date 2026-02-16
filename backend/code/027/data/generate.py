"""
Delayed Copy Task Data Generator

Task description (from proposal 027 MVE):
  Input:  [t_1, t_2, ..., t_k, PAD, PAD, ..., PAD, SEP]
  Target: [IGN, IGN, ..., IGN, IGN, IGN, ..., IGN, t_1, t_2, ..., t_k]

  The model must memorize k tokens, wait T padding steps, then reproduce them.

  - vocab_size: number of content tokens (8 in MVE)
  - k: number of tokens to copy (5 in MVE)
  - T: delay length (50, 100, 200, 500 in MVE)

Token encoding:
  0: PAD token (used during delay)
  1: SEP token (signals "start outputting")
  2..vocab_size+1: content tokens

Sequence structure:
  Input:  [content_0, ..., content_{k-1}, PAD, ..., PAD, SEP, PAD, ..., PAD]
  Target: [IGNORE, ..., IGNORE,           IGNORE,...,IGNORE, IGNORE, content_0, ..., content_{k-1}]

  Total length = k + T + 1 + k = 2k + T + 1
  (k content + T delay + 1 SEP + k output positions)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random


# Special token IDs
PAD_TOKEN = 0
SEP_TOKEN = 1
CONTENT_OFFSET = 2  # Content tokens start at index 2
IGNORE_INDEX = -100  # PyTorch cross_entropy ignore index


class DelayedCopyDataset(Dataset):
    """
    Generates delayed copy task data.

    Each sample:
    - Input: k random content tokens, then T PAD tokens, then SEP, then k PAD tokens
    - Target: IGNORE for first k + T + 1 positions, then the k content tokens

    The model must memorize the k content tokens across T padding steps
    and reproduce them after seeing the SEP token.
    """
    def __init__(
        self,
        num_samples: int,
        vocab_size: int = 8,
        k: int = 5,
        delay: int = 100,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.k = k
        self.delay = delay
        self.total_vocab = vocab_size + CONTENT_OFFSET  # PAD + SEP + content tokens
        self.seq_len = 2 * k + delay + 1  # k content + T delay + 1 SEP + k output

        # Pre-generate all data
        rng = random.Random(seed)
        self.data = []
        for _ in range(num_samples):
            # Random content tokens (k tokens from content vocabulary)
            content = [rng.randint(CONTENT_OFFSET, self.total_vocab - 1) for _ in range(k)]

            # Build input sequence
            # [content_0, ..., content_{k-1}, PAD, ..., PAD, SEP, PAD, ..., PAD]
            input_seq = content + [PAD_TOKEN] * delay + [SEP_TOKEN] + [PAD_TOKEN] * k

            # Build target sequence
            # [IGNORE, ..., IGNORE, content_0, ..., content_{k-1}]
            target_seq = [IGNORE_INDEX] * (k + delay + 1) + content

            assert len(input_seq) == self.seq_len, f"Input length {len(input_seq)} != {self.seq_len}"
            assert len(target_seq) == self.seq_len, f"Target length {len(target_seq)} != {self.seq_len}"

            self.data.append((
                torch.tensor(input_seq, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long),
            ))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloaders(
    num_train: int = 5000,
    num_test: int = 1000,
    vocab_size: int = 8,
    k: int = 5,
    delay: int = 100,
    batch_size: int = 64,
    seed: int = 42,
):
    """Create train and test dataloaders for the delayed copy task."""
    train_ds = DelayedCopyDataset(num_train, vocab_size, k, delay, seed=seed)
    test_ds = DelayedCopyDataset(num_test, vocab_size, k, delay, seed=seed + 1000)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_ds.total_vocab, train_ds.seq_len
