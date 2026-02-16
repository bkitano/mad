"""
S5 Permutation Composition Task Data Generator.

From proposal 022 MVE:
  "S5 permutation composition — input a sequence of S5 generators,
   output the composed permutation."

The symmetric group S5 consists of all permutations of {0,1,2,3,4}.
We use 2 generators:
  - g0 = (0 1 2 3 4) — cyclic permutation
  - g1 = (0 1)       — transposition

Together these generate all of S5 (|S5| = 120).

Task format:
  Input:  [gen_id_1, gen_id_2, ..., gen_id_T]  (each ∈ {0, 1})
  Target: The composed permutation g_{gen_id_T} ∘ ... ∘ g_{gen_id_2} ∘ g_{gen_id_1}
          represented as a single integer ∈ {0, 1, ..., 119}

The model must learn to track state through sequential composition of
non-commutative group elements — this requires genuine state mixing,
not just element-wise operations.

This is the canonical benchmark for non-abelian state tracking,
used across many SSM proposals (001, 006, 010, 016, 017, 022).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from itertools import permutations


# ============================================================
# S5 Group Utilities
# ============================================================

def compose_perms(p1: tuple, p2: tuple) -> tuple:
    """Compose two permutations: (p2 ∘ p1)(x) = p2(p1(x))."""
    return tuple(p2[p1[i]] for i in range(len(p1)))


def build_s5_lookup():
    """
    Build a lookup table mapping each S5 permutation to an integer ID.

    Returns:
        perm_to_id: dict mapping tuple(perm) -> int
        id_to_perm: dict mapping int -> tuple(perm)
        generators: list of two generator permutations
    """
    # Generators of S5:
    # g0 = (0 1 2 3 4) — cyclic permutation
    # g1 = (0 1)       — transposition swapping 0 and 1
    g0 = (1, 2, 3, 4, 0)  # cyclic: 0->1, 1->2, 2->3, 3->4, 4->0
    g1 = (1, 0, 2, 3, 4)  # transposition: swap 0,1
    generators = [g0, g1]

    # Generate all of S5 via BFS from identity using generators
    identity = (0, 1, 2, 3, 4)
    visited = {identity}
    queue = [identity]

    while queue:
        current = queue.pop(0)
        for g in generators:
            new = compose_perms(current, g)
            if new not in visited:
                visited.add(new)
                queue.append(new)

    assert len(visited) == 120, f"Expected |S5|=120, got {len(visited)}"

    # Sort for deterministic ordering
    all_perms = sorted(visited)
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    id_to_perm = {i: p for i, p in enumerate(all_perms)}

    return perm_to_id, id_to_perm, generators


# Build global lookup tables
PERM_TO_ID, ID_TO_PERM, GENERATORS = build_s5_lookup()
NUM_CLASSES = 120  # |S5|
NUM_GENERATORS = 2  # Two generators


class S5CompositionDataset(Dataset):
    """
    S5 permutation composition dataset.

    Each sample:
      - Input: sequence of generator indices [g_1, g_2, ..., g_T], each ∈ {0, 1}
      - Target: integer ID of composed permutation g_T ∘ ... ∘ g_2 ∘ g_1

    The model must learn the non-abelian group operation.

    Args:
        num_samples: Number of sequences to generate
        seq_len: Length of each generator sequence
        seed: Random seed
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_len: int = 20,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.rng = np.random.RandomState(seed)
        self.data = self._generate()

    def _generate(self):
        """Pre-generate all samples."""
        data = []
        for _ in range(self.num_samples):
            # Random sequence of generator indices
            gen_indices = self.rng.randint(0, NUM_GENERATORS, size=self.seq_len)

            # Compose permutations left-to-right: g_1 then g_2 then ... then g_T
            current = (0, 1, 2, 3, 4)  # identity
            for gi in gen_indices:
                current = compose_perms(current, GENERATORS[gi])

            target = PERM_TO_ID[current]
            data.append((gen_indices, target))

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        gen_indices, target = self.data[idx]
        return (
            torch.tensor(gen_indices, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def generate_s5_data(
    num_train: int = 10000,
    num_val: int = 1000,
    num_test: int = 1000,
    seq_len: int = 20,
    batch_size: int = 128,
) -> tuple:
    """
    Generate train/val/test dataloaders for S5 composition.

    Returns:
        (train_loader, val_loader, test_loader, dataset_info)
    """
    train_ds = S5CompositionDataset(num_samples=num_train, seq_len=seq_len, seed=42)
    val_ds = S5CompositionDataset(num_samples=num_val, seq_len=seq_len, seed=43)
    test_ds = S5CompositionDataset(num_samples=num_test, seq_len=seq_len, seed=44)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    dataset_info = {
        "num_generators": NUM_GENERATORS,
        "num_classes": NUM_CLASSES,  # 120 = |S5|
        "seq_len": seq_len,
        "vocab_size": NUM_GENERATORS,  # Input vocab = {0, 1}
    }

    return train_loader, val_loader, test_loader, dataset_info
