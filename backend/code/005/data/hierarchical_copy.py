"""
Hierarchical Copying Task for MVE 005.

From proposal MVE section:
  - Input: Nested structure [A [B C] D [E [F G] H]] (8 tokens, 3 hierarchy levels)
  - Target: Copy input with delays:
      level-1 (immediate copy at t+0)
      level-2 (copy at t+4)
      level-3 (copy at t+8)
  - Why: Requires multi-scale memory (local + global); HSS structure should excel

Implementation:
  We generate hierarchical sequences where tokens must be remembered and
  recalled at different delays depending on their hierarchy level.

  The task tests whether the model can maintain hierarchical state structure
  in its linear attention state matrix.

Simplified task design:
  - We generate sequences of length T with tokens at 3 hierarchy levels
  - Input phase (tokens 1..8): present the hierarchical structure
  - Output phase: model must reproduce tokens at level-dependent delays
  - Level 1 tokens: recalled after short delay (1-2 steps)
  - Level 2 tokens: recalled after medium delay (3-5 steps)
  - Level 3 tokens: recalled after long delay (6-8 steps)

This creates a multi-scale memory requirement that should benefit from
hierarchical (HSS) state structure.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple, Optional


# Token definitions
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
SEP_TOKEN = 3  # Separator between input and target phases
VOCAB_START = 4  # Actual vocabulary starts here
NUM_CONTENT_TOKENS = 16  # 16 distinct content tokens
VOCAB_SIZE = VOCAB_START + NUM_CONTENT_TOKENS  # = 20 total tokens

# Special indices
IGNORE_INDEX = -100

# Hierarchy levels
LEVEL_1 = 0  # Immediate/local
LEVEL_2 = 1  # Medium range
LEVEL_3 = 2  # Long range


def generate_hierarchical_sequence(
    num_tokens: int = 8,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[list, list, list]:
    """
    Generate a single hierarchical copying example.

    Creates a nested structure where tokens are assigned hierarchy levels:
      Level 1 (local):  tokens at positions [0, 3]  (immediate recall)
      Level 2 (medium): tokens at positions [1, 2, 4, 7] (medium delay recall)
      Level 3 (global): tokens at positions [5, 6] (long delay recall)

    The structure looks like:
      [A [B C] D [E [F G] H]]
       1  2 2  1  2  3 3  2

    Returns:
        tokens: list of token IDs
        levels: list of hierarchy levels (0, 1, 2)
        delays: list of recall delays for each token
    """
    if rng is None:
        rng = np.random.RandomState()

    # Generate random content tokens
    tokens = [rng.randint(VOCAB_START, VOCAB_START + NUM_CONTENT_TOKENS) for _ in range(num_tokens)]

    # Assign hierarchy levels to match the nested structure
    # [A [B C] D [E [F G] H]]
    #  0  1 1  0  1  2 2  1
    levels = [LEVEL_1, LEVEL_2, LEVEL_2, LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_3, LEVEL_2]

    # Delays based on level (proposal: level-1 +0, level-2 +4, level-3 +8)
    level_delays = {LEVEL_1: 0, LEVEL_2: 4, LEVEL_3: 8}
    delays = [level_delays[l] for l in levels]

    return tokens, levels, delays


class HierarchicalCopyDataset(Dataset):
    """
    Dataset for hierarchical copying task.

    Each sample is:
      Input:  [BOS, t1, t2, ..., t8, SEP, PAD, ..., PAD]
      Target: [IGN, IGN, ..., IGN, IGN, target_at_delay_positions]

    The target sequence requires the model to reproduce tokens at positions
    determined by their hierarchy level:
      - Level 1 tokens appear immediately after SEP
      - Level 2 tokens appear 4 steps after their input position + SEP offset
      - Level 3 tokens appear 8 steps after their input position + SEP offset

    Total sequence length: BOS + 8 input + SEP + 16 output + EOS = 27
    (We use a max_seq_len of 32 with padding)
    """

    def __init__(
        self,
        num_samples: int = 5000,
        num_input_tokens: int = 8,
        max_seq_len: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_input_tokens = num_input_tokens
        self.max_seq_len = max_seq_len

        rng = np.random.RandomState(seed)

        self.data = []
        for i in range(num_samples):
            tokens, levels, delays = generate_hierarchical_sequence(num_input_tokens, rng)
            self.data.append((tokens, levels, delays))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_seq: (max_seq_len,) - input token IDs
            target_seq: (max_seq_len,) - target token IDs with IGNORE_INDEX for non-target positions
            level_info: (max_seq_len,) - hierarchy level at each position (-1 for non-content)
        """
        tokens, levels, delays = self.data[idx]

        input_seq = torch.full((self.max_seq_len,), PAD_TOKEN, dtype=torch.long)
        target_seq = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        level_info = torch.full((self.max_seq_len,), -1, dtype=torch.long)

        # Build input: [BOS, t1, ..., t8, SEP, ...]
        pos = 0
        input_seq[pos] = BOS_TOKEN
        pos += 1

        for i, tok in enumerate(tokens):
            input_seq[pos] = tok
            level_info[pos] = levels[i]
            pos += 1

        sep_pos = pos
        input_seq[pos] = SEP_TOKEN
        pos += 1

        # Build target: tokens should appear at output_start + delay
        output_start = sep_pos + 1

        for i, (tok, level, delay) in enumerate(zip(tokens, levels, delays)):
            target_pos = output_start + i + delay
            if target_pos < self.max_seq_len:
                # The input at this position is PAD (model sees PAD, must predict token)
                target_seq[target_pos] = tok
                level_info[target_pos] = level

        # Add EOS at the end of valid region
        last_target = max(output_start + i + d for i, d in enumerate(delays)) + 1
        if last_target < self.max_seq_len:
            input_seq[last_target] = EOS_TOKEN

        return input_seq, target_seq, level_info

    @staticmethod
    def get_vocab_size():
        return VOCAB_SIZE

    @staticmethod
    def get_num_classes():
        return VOCAB_SIZE


def create_dataloaders(
    num_samples: int = 5000,
    batch_size: int = 64,
    test_split: float = 0.2,
    max_seq_len: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset = HierarchicalCopyDataset(
        num_samples=num_samples,
        max_seq_len=max_seq_len,
        seed=seed,
    )

    # Split: 70% train, 15% val, 15% test
    n_total = len(dataset)
    n_test = int(n_total * test_split / 2)
    n_val = int(n_total * test_split / 2)
    n_train = n_total - n_test - n_val

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def verify_dataset():
    """Verify dataset generation with a few examples."""
    ds = HierarchicalCopyDataset(num_samples=5, max_seq_len=32, seed=0)

    print("=== Hierarchical Copy Dataset Verification ===")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Content tokens: {VOCAB_START} to {VOCAB_START + NUM_CONTENT_TOKENS - 1}")
    print()

    token_names = {PAD_TOKEN: "PAD", BOS_TOKEN: "BOS", EOS_TOKEN: "EOS", SEP_TOKEN: "SEP"}
    for i in range(VOCAB_START, VOCAB_SIZE):
        token_names[i] = f"T{i-VOCAB_START}"

    for idx in range(min(3, len(ds))):
        input_seq, target_seq, level_info = ds[idx]
        print(f"Sample {idx}:")
        print(f"  Input:  {[token_names.get(t.item(), '?') for t in input_seq]}")
        print(f"  Target: {['IGN' if t.item() == IGNORE_INDEX else token_names.get(t.item(), '?') for t in target_seq]}")
        print(f"  Levels: {level_info.tolist()}")
        print()


if __name__ == "__main__":
    verify_dataset()
