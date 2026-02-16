"""
Selective Copying Task Data Generator.

From proposal 007 MVE:
  "Given an input sequence `a b c d [SEP] _ _ 3 _`, output the token
   at the specified index (3rd token = c). Sequence length 16-32,
   vocabulary size 16."

REVISED Task format (v2):
  Input:  [c0, c1, c2, ..., c_{k-1}, SEP, idx0, idx1, ..., idx_{q-1}]
  Target: [-1, -1, ..., -1, -1, ans0, ans1, ..., ans_{q-1}]

Each idx_j is an index token pointing to a content position.
Each ans_j is the content token at that position.

Multiple queries per sequence (more training signal per example).

This requires input-dependent (selective) dynamics because the model
must route information based on the content of the index token.
An LTI model cannot do this — it applies the same transformation
regardless of input content.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Special tokens
PAD_TOKEN = 0
SEP_TOKEN = 1
CONTENT_OFFSET = 2  # Content tokens start at index 2


class SelectiveCopyingDataset(Dataset):
    """
    Selective copying task dataset (v2 — compact format).

    Each sample consists of:
    - content_len content tokens at positions 0..content_len-1
    - SEP token at position content_len
    - num_queries index tokens at positions content_len+1..content_len+num_queries
    - Targets at each query position = the content token at that index

    This format puts queries right after content (no PAD gap),
    giving the SSM minimal distance to carry information.

    Args:
        num_samples: Number of sequences to generate
        content_len: Number of content tokens before SEP
        num_queries: Number of queries after SEP
        vocab_size: Number of content token types
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        num_samples: int = 10000,
        content_len: int = 8,
        num_queries: int = 4,
        vocab_size: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.content_len = content_len
        self.num_queries = num_queries
        self.vocab_size = vocab_size
        self.num_content_tokens = vocab_size

        # Total sequence length = content + SEP + queries
        self.seq_len = content_len + 1 + num_queries

        # Total vocab: PAD + SEP + content_tokens + index_tokens
        self.num_index_tokens = content_len
        self.total_vocab_size = 2 + self.num_content_tokens + self.num_index_tokens

        # Index tokens start after content tokens
        self.index_offset = 2 + self.num_content_tokens

        self.rng = np.random.RandomState(seed)
        self.data = self._generate()

    def _generate(self):
        """Pre-generate all samples."""
        data = []
        for _ in range(self.num_samples):
            # Generate random content tokens
            content = self.rng.randint(0, self.num_content_tokens, size=self.content_len)
            content_tokens = content + CONTENT_OFFSET  # Shift to avoid PAD/SEP

            # Generate random query indices (with replacement — can query same position twice)
            query_indices = self.rng.randint(0, self.content_len, size=self.num_queries)

            # Build input sequence: [c0, c1, ..., SEP, idx0, idx1, ...]
            input_seq = np.zeros(self.seq_len, dtype=np.int64)
            input_seq[:self.content_len] = content_tokens
            input_seq[self.content_len] = SEP_TOKEN
            for j, qi in enumerate(query_indices):
                input_seq[self.content_len + 1 + j] = self.index_offset + qi

            # Build target sequence: only query positions have valid targets
            target_seq = np.full(self.seq_len, -1, dtype=np.int64)
            for j, qi in enumerate(query_indices):
                target_seq[self.content_len + 1 + j] = content_tokens[qi]

            data.append((input_seq, target_seq))
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
        )


def generate_selective_copying_data(
    num_train: int = 8000,
    num_val: int = 1000,
    num_test: int = 1000,
    content_len: int = 8,
    num_queries: int = 4,
    vocab_size: int = 16,
    batch_size: int = 64,
    **kwargs,  # Ignore extra config keys like seq_len
) -> tuple:
    """
    Generate train/val/test dataloaders for selective copying.

    Returns:
        (train_loader, val_loader, test_loader, dataset_info)
    """
    train_ds = SelectiveCopyingDataset(
        num_samples=num_train,
        content_len=content_len,
        num_queries=num_queries,
        vocab_size=vocab_size,
        seed=42,
    )
    val_ds = SelectiveCopyingDataset(
        num_samples=num_val,
        content_len=content_len,
        num_queries=num_queries,
        vocab_size=vocab_size,
        seed=43,
    )
    test_ds = SelectiveCopyingDataset(
        num_samples=num_test,
        content_len=content_len,
        num_queries=num_queries,
        vocab_size=vocab_size,
        seed=44,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    dataset_info = {
        "total_vocab_size": train_ds.total_vocab_size,
        "num_classes": train_ds.total_vocab_size,
        "seq_len": train_ds.seq_len,
        "content_len": content_len,
        "num_queries": num_queries,
    }

    return train_loader, val_loader, test_loader, dataset_info
