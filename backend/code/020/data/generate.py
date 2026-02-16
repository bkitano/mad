"""
S3 Permutation Composition Task Data Generator.

From proposal 020 MVE:
  "S3 permutation composition — compose sequences of 3-element permutations,
   predict resulting permutation"

S3 = Symmetric group on 3 elements = all permutations of {1, 2, 3}.
|S3| = 6 elements. S3 is the simplest non-abelian group.

Elements of S3 (as permutations of {0,1,2}):
  e    = (0,1,2)  identity
  (01) = (1,0,2)  swap 0↔1
  (02) = (2,1,0)  swap 0↔2
  (12) = (0,2,1)  swap 1↔2
  (012) = (1,2,0) cyclic rotation
  (021) = (2,0,1) reverse cyclic

Generators: {(01), (012)} generate all of S3.

Task format (scan-style, like code/001 D4 task):
  Input:  [BOS, g1, g2, ..., gT, EOS, PAD, ...]
  Target: [IGN, g1, g1·g2, g1·g2·g3, ..., g1·...·gT, IGN, IGN, ...]

The model must predict the running composition (prefix products) at each position.
This requires non-abelian state-tracking since S3 is non-abelian: (01)·(012) ≠ (012)·(01).
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ============================================================
# S3 Group Implementation
# ============================================================

# S3 elements as permutations of {0, 1, 2}
S3_ELEMENTS = [
    (0, 1, 2),  # 0: e     (identity)
    (1, 0, 2),  # 1: (01)  swap 0,1
    (2, 1, 0),  # 2: (02)  swap 0,2
    (0, 2, 1),  # 3: (12)  swap 1,2
    (1, 2, 0),  # 4: (012) cyclic
    (2, 0, 1),  # 5: (021) reverse cyclic
]

S3_SIZE = len(S3_ELEMENTS)  # 6

# Map permutation tuple → index
_PERM_TO_IDX = {p: i for i, p in enumerate(S3_ELEMENTS)}

# Precompute multiplication table
# mult_table[i][j] = index of S3_ELEMENTS[i] ∘ S3_ELEMENTS[j]
# where (σ ∘ τ)(x) = σ(τ(x))
_MULT_TABLE = np.zeros((S3_SIZE, S3_SIZE), dtype=np.int64)
for i, sigma in enumerate(S3_ELEMENTS):
    for j, tau in enumerate(S3_ELEMENTS):
        # Compose: first apply tau, then sigma
        composed = tuple(sigma[tau[k]] for k in range(3))
        _MULT_TABLE[i, j] = _PERM_TO_IDX[composed]

# Generators of S3: {(01), (012)} — swap and cyclic rotation
# These 2 generators suffice to reach all 6 elements
GENERATOR_INDICES = [1, 4]  # (01) and (012)


def s3_multiply(a_idx: int, b_idx: int) -> int:
    """Compose S3 elements: result = a ∘ b."""
    return int(_MULT_TABLE[a_idx, b_idx])


def s3_scan(indices: list) -> list:
    """Compute prefix products: [g1, g1·g2, g1·g2·g3, ...]."""
    result = 0  # Identity index
    scan = []
    for idx in indices:
        result = s3_multiply(result, idx)
        scan.append(result)
    return scan


# ============================================================
# Special tokens
# ============================================================
BOS_IDX = S3_SIZE      # 6
EOS_IDX = S3_SIZE + 1  # 7
PAD_IDX = S3_SIZE + 2  # 8
TOTAL_VOCAB = S3_SIZE + 3  # 9 tokens: 6 group elements + BOS + EOS + PAD
IGNORE_IDX = -100  # CrossEntropyLoss ignore_index


class S3CompositionDataset(Dataset):
    """
    S3 permutation composition dataset.

    Each sample:
      Input:  [BOS, g1, g2, ..., gT, EOS, PAD...]
      Target: [IGN, g1, g1·g2, ..., g1·...·gT, IGN, IGN...]

    Args:
        num_samples: Number of sequences
        seq_len: Number of group elements per sequence (T)
        max_pad_len: Total padded length (T + 2 for BOS/EOS, then pad)
        use_generators_only: If True, only use generators {(01), (012)}
        seed: Random seed
    """

    def __init__(
        self,
        num_samples: int = 5000,
        seq_len: int = 32,
        max_pad_len: int = 36,
        use_generators_only: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.max_pad_len = max(max_pad_len, seq_len + 2)
        self.use_generators_only = use_generators_only
        self.rng = np.random.RandomState(seed)
        self.data = self._generate()

    def _generate(self):
        """Pre-generate all samples."""
        data = []
        for _ in range(self.num_samples):
            if self.use_generators_only:
                elements = self.rng.choice(GENERATOR_INDICES, size=self.seq_len)
            else:
                elements = self.rng.randint(0, S3_SIZE, size=self.seq_len)

            scan = s3_scan(elements.tolist())

            # Build input: [BOS, g1, g2, ..., gT, EOS, PAD...]
            input_seq = np.full(self.max_pad_len, PAD_IDX, dtype=np.int64)
            input_seq[0] = BOS_IDX
            input_seq[1:1 + self.seq_len] = elements
            input_seq[1 + self.seq_len] = EOS_IDX

            # Build target: [IGN, s1, s2, ..., sT, IGN, IGN...]
            target_seq = np.full(self.max_pad_len, IGNORE_IDX, dtype=np.int64)
            for j in range(self.seq_len):
                target_seq[1 + j] = scan[j]

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


def generate_s3_data(
    num_train: int = 5000,
    num_val: int = 500,
    num_test: int = 500,
    seq_len: int = 32,
    use_generators_only: bool = True,
    batch_size: int = 128,
) -> tuple:
    """
    Generate S3 composition train/val/test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader, dataset_info)
    """
    max_pad_len = seq_len + 4  # BOS + seq + EOS + 1 extra pad

    train_ds = S3CompositionDataset(
        num_samples=num_train, seq_len=seq_len,
        max_pad_len=max_pad_len, use_generators_only=use_generators_only, seed=42,
    )
    val_ds = S3CompositionDataset(
        num_samples=num_val, seq_len=seq_len,
        max_pad_len=max_pad_len, use_generators_only=use_generators_only, seed=43,
    )
    test_ds = S3CompositionDataset(
        num_samples=num_test, seq_len=seq_len,
        max_pad_len=max_pad_len, use_generators_only=use_generators_only, seed=44,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    dataset_info = {
        'total_vocab_size': TOTAL_VOCAB,
        'num_classes': S3_SIZE,  # Predict one of 6 group elements
        'seq_len': max_pad_len,
        'group_size': S3_SIZE,
        'generator_indices': GENERATOR_INDICES,
    }

    return train_loader, val_loader, test_loader, dataset_info
