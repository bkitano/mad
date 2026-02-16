"""
Selective Copying Task — the minimal task separating softmax from linear attention.

Task description (from proposal 014 §MVE):
    Given a sequence like [a, _, b, _, c, QUERY:2] → b
    Retrieve the token at the queried position from the memory.

This requires SHARP attention: the model must attend to exactly one position
with high confidence. Linear attention (diffuse) and diagonal SSMs (no content
matching) should fail, while LogSSM (exact softmax via log-semiring) should succeed.

Concrete format:
    - Vocabulary: N_TOKENS content tokens + special tokens (PAD, BLANK, QUERY markers)
    - Sequence: [tok_0, BLANK, tok_1, BLANK, ..., tok_{M-1}, QUERY_j]
    - Target: predict tok_j at the QUERY position (ignore loss at all other positions)

For MVE:
    - 8 tokens to remember (M=8)
    - 1 query per sequence
    - Sequence length 32 (with blanks filling the rest)
    - 5000 training sequences
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random


# Special tokens
PAD_TOKEN = 0
BLANK_TOKEN = 1
QUERY_BASE = 2  # QUERY_j = QUERY_BASE + j for j in [0, M-1]


class SelectiveCopyDataset(Dataset):
    """Selective copying dataset.

    Each sample:
        tokens:  [tok_0, BLANK, tok_1, BLANK, ..., tok_{M-1}, BLANK, ..., QUERY_j, PAD, ...]
        targets: [IGNORE, ..., IGNORE, tok_j, IGNORE, ..., IGNORE]

    The model must learn to retrieve tok_j when seeing QUERY_j.
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        num_samples: int,
        seq_len: int = 32,
        n_memory_tokens: int = 8,
        n_content_vocab: int = 16,
        seed: int = 42,
    ):
        """
        Args:
            num_samples: Number of sequences to generate
            seq_len: Total sequence length
            n_memory_tokens: Number of tokens to remember (M)
            n_content_vocab: Number of distinct content token values
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.n_memory_tokens = n_memory_tokens
        self.n_content_vocab = n_content_vocab

        # Vocabulary layout:
        # [0: PAD, 1: BLANK, 2..2+M-1: QUERY_0..QUERY_{M-1}, 2+M..2+M+V-1: content tokens]
        self.query_tokens = list(range(QUERY_BASE, QUERY_BASE + n_memory_tokens))
        self.content_start = QUERY_BASE + n_memory_tokens
        self.vocab_size = self.content_start + n_content_vocab

        # Number of classes = n_content_vocab (predict which content token)
        self.num_classes = self.vocab_size

        # Pre-generate all samples
        rng = random.Random(seed)
        self.samples = []
        for _ in range(num_samples):
            self.samples.append(self._generate_sample(rng))

    def _generate_sample(self, rng: random.Random):
        """Generate one selective copy sample."""
        tokens = [PAD_TOKEN] * self.seq_len
        targets = [self.IGNORE_INDEX] * self.seq_len

        # Place memory tokens: tok_0, BLANK, tok_1, BLANK, ...
        # Memory tokens are placed at positions 0, 2, 4, ..., 2*(M-1)
        memory_values = []
        for i in range(self.n_memory_tokens):
            val = rng.randint(0, self.n_content_vocab - 1)
            content_token = self.content_start + val
            pos = i * 2
            tokens[pos] = content_token
            memory_values.append(content_token)

            # Fill blank after each memory token
            if pos + 1 < self.seq_len:
                tokens[pos + 1] = BLANK_TOKEN

        # Place query token after the memory region
        # Memory occupies positions 0..2*M-1
        query_pos = self.n_memory_tokens * 2
        query_idx = rng.randint(0, self.n_memory_tokens - 1)
        tokens[query_pos] = self.query_tokens[query_idx]

        # Target: at query_pos, predict the memory token at query_idx
        targets[query_pos] = memory_values[query_idx]

        # Fill remaining positions with BLANK
        for i in range(query_pos + 1, self.seq_len):
            tokens[i] = BLANK_TOKEN

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            query_pos,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens, targets, query_pos = self.samples[idx]
        return tokens, targets, query_pos


def generate_selective_copy_data(
    num_train: int = 5000,
    num_val: int = 500,
    num_test: int = 500,
    seq_len: int = 32,
    n_memory_tokens: int = 8,
    n_content_vocab: int = 16,
    batch_size: int = 256,
):
    """Generate train/val/test dataloaders for selective copying.

    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    train_ds = SelectiveCopyDataset(num_train, seq_len, n_memory_tokens, n_content_vocab, seed=42)
    val_ds = SelectiveCopyDataset(num_val, seq_len, n_memory_tokens, n_content_vocab, seed=43)
    test_ds = SelectiveCopyDataset(num_test, seq_len, n_memory_tokens, n_content_vocab, seed=44)

    def collate_fn(batch):
        tokens = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        query_pos = torch.tensor([b[2] for b in batch])
        return tokens, targets, query_pos

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    dataset_info = {
        "vocab_size": train_ds.vocab_size,
        "num_classes": train_ds.num_classes,
        "seq_len": seq_len,
        "n_memory_tokens": n_memory_tokens,
        "n_content_vocab": n_content_vocab,
    }

    return train_loader, val_loader, test_loader, dataset_info
