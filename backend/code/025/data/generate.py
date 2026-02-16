"""
Synthetic copying task with delay for Nystrom SSM MVE.

From proposal 025:
- Copy a sequence of tokens after a gap of G positions
- Requires state to maintain information across chunk boundaries
- 5K sequences of length 256 (8 chunks of size 32)
- Copy delay G=64 (spanning 2 chunk boundaries)

Task format:
  Input:  [t1, t2, ..., tk, SEP, PAD, PAD, ..., PAD, QUERY, PAD, ..., PAD]
  Target: [IGN, IGN, ..., IGN, IGN, IGN, IGN, ..., IGN, t1,  t2,  ..., tk]

The model must:
1. Read content tokens t1..tk at the beginning
2. Store them in state across the PAD gap
3. After QUERY token, reproduce the content tokens in order

The delay G ensures information must cross chunk boundaries (G=64 spans 2 boundaries
when chunk_size=32).

Special tokens:
  PAD = 0
  SEP = 1
  QUERY = 2
  Content tokens: 3, 4, ..., vocab_size-1
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class DelayedCopyDataset(Dataset):
    """
    Delayed copy task dataset.

    The model sees content tokens, then a gap, then must reproduce them.

    Args:
        n_samples: number of sequences
        seq_len: total sequence length
        n_content: number of content tokens to copy (k)
        delay: gap between content and query (G)
        vocab_size: total vocabulary size (including special tokens)
        seed: random seed
    """

    PAD = 0
    SEP = 1
    QUERY = 2
    CONTENT_OFFSET = 3  # Content tokens start at index 3

    def __init__(
        self,
        n_samples: int = 5000,
        seq_len: int = 256,
        n_content: int = 8,
        delay: int = 64,
        vocab_size: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_content = n_content
        self.delay = delay
        self.vocab_size = vocab_size
        self.n_content_tokens = vocab_size - self.CONTENT_OFFSET

        assert self.n_content_tokens > 0, \
            f"vocab_size ({vocab_size}) must be > {self.CONTENT_OFFSET} for content tokens"
        assert n_content + 1 + delay + n_content <= seq_len, \
            f"Sequence too short: need {n_content + 1 + delay + n_content}, got {seq_len}"

        # Pre-generate all data
        gen = torch.Generator().manual_seed(seed)
        self.inputs, self.targets = self._generate(gen)

    def _generate(self, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all sequences."""
        inputs = torch.full(
            (self.n_samples, self.seq_len),
            self.PAD,
            dtype=torch.long,
        )
        # Use -100 as ignore index for cross-entropy loss
        targets = torch.full(
            (self.n_samples, self.seq_len),
            -100,
            dtype=torch.long,
        )

        for i in range(self.n_samples):
            # Generate random content tokens
            content = torch.randint(
                self.CONTENT_OFFSET,
                self.vocab_size,
                (self.n_content,),
                generator=gen,
            )

            # Place content tokens at start
            inputs[i, :self.n_content] = content

            # Place SEP after content
            sep_pos = self.n_content
            inputs[i, sep_pos] = self.SEP

            # Place QUERY token after delay
            query_pos = sep_pos + 1 + self.delay
            inputs[i, query_pos] = self.QUERY

            # Target: reproduce content tokens after QUERY
            for j in range(self.n_content):
                target_pos = query_pos + 1 + j
                if target_pos < self.seq_len:
                    targets[i, target_pos] = content[j]
                    # Input at target positions: PAD (model must generate from state)

        return inputs, targets

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def create_dataloaders(
    n_train: int = 5000,
    n_test: int = 1000,
    seq_len: int = 256,
    n_content: int = 8,
    delay: int = 64,
    vocab_size: int = 16,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders."""
    train_ds = DelayedCopyDataset(
        n_samples=n_train,
        seq_len=seq_len,
        n_content=n_content,
        delay=delay,
        vocab_size=vocab_size,
        seed=seed,
    )
    test_ds = DelayedCopyDataset(
        n_samples=n_test,
        seq_len=seq_len,
        n_content=n_content,
        delay=delay,
        vocab_size=vocab_size,
        seed=seed + 1000,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
    )

    return train_loader, test_loader


if __name__ == '__main__':
    """Test data generation."""
    ds = DelayedCopyDataset(n_samples=10, seq_len=256, n_content=8, delay=64, vocab_size=16)

    print(f"Dataset size: {len(ds)}")
    print(f"Vocab size: {ds.vocab_size}")
    print(f"Content tokens: {ds.n_content_tokens}")
    print(f"Content offset: {ds.CONTENT_OFFSET}")

    inp, tgt = ds[0]
    print(f"\nSample 0:")
    print(f"  Input shape: {inp.shape}")
    print(f"  Target shape: {tgt.shape}")

    # Show the structure
    content = inp[:8]
    sep = inp[8]
    query_pos = 8 + 1 + 64
    query = inp[query_pos]
    target_tokens = tgt[query_pos+1:query_pos+1+8]

    print(f"  Content tokens: {content.tolist()}")
    print(f"  SEP token (pos {8}): {sep.item()}")
    print(f"  QUERY token (pos {query_pos}): {query.item()}")
    print(f"  Target tokens: {target_tokens.tolist()}")
    print(f"  Content == Target: {(content == target_tokens).all().item()}")

    # Count chunk boundaries crossed
    chunk_size = 32
    content_chunk = 0 // chunk_size  # chunk 0
    query_chunk = query_pos // chunk_size
    print(f"\n  Content in chunk: {content_chunk}")
    print(f"  Query in chunk: {query_chunk}")
    print(f"  Chunk boundaries crossed: {query_chunk - content_chunk}")
