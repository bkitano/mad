"""
Data generators for group state tracking tasks.

Task: Given a sequence of group generators, predict the composed group element
at each position (prefix composition / scan).

Groups implemented:
    S3: Symmetric group on 3 elements (6 elements, 2 generators)
        - Non-abelian, simplest non-commutative group
        - Tests basic state tracking capability

    D4: Dihedral group of order 8 (8 elements, 2 generators)
        - Non-abelian, includes reflections + rotations
        - Tests sign-flipping capability of B_n structure

Token format:
    [g_1, g_2, ..., g_T]

Target format (scan/prefix composition):
    [g_1, g_1*g_2, g_1*g_2*g_3, ..., g_1*...*g_T]

Each target is the index of the composed group element (classification).
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import random


# ============================================================
# S3: Symmetric group on 3 elements
# ============================================================

class S3Group:
    """S3 = {e, (12), (13), (23), (123), (132)} with 6 elements.

    Generators: sigma1 = (1,2), sigma2 = (2,3)
    Represented as permutations of [0, 1, 2].
    """

    def __init__(self):
        # All 6 elements as tuples (permutation of [0,1,2])
        self.elements = [
            (0, 1, 2),  # e (identity)
            (1, 0, 2),  # (0,1)
            (0, 2, 1),  # (1,2)
            (2, 1, 0),  # (0,2)
            (1, 2, 0),  # (0,1,2)
            (2, 0, 1),  # (0,2,1)
        ]
        self.n_elements = 6

        # Generators
        self.generators = [
            (1, 0, 2),  # sigma1 = swap(0,1)
            (0, 2, 1),  # sigma2 = swap(1,2)
        ]
        self.n_generators = 2

        # Map from permutation tuple to index
        self.elem_to_idx = {e: i for i, e in enumerate(self.elements)}

        # Precompute Cayley table
        self.cayley = self._build_cayley_table()

    def _compose(self, p1: tuple, p2: tuple) -> tuple:
        """Compose two permutations: (p1 * p2)(x) = p1(p2(x))."""
        return tuple(p1[p2[i]] for i in range(3))

    def _build_cayley_table(self) -> torch.Tensor:
        """Build N x N Cayley table: cayley[i][j] = idx(elements[i] * elements[j])."""
        table = torch.zeros(self.n_elements, self.n_elements, dtype=torch.long)
        for i, a in enumerate(self.elements):
            for j, b in enumerate(self.elements):
                product = self._compose(a, b)
                table[i, j] = self.elem_to_idx[product]
        return table

    def compose_idx(self, i: int, j: int) -> int:
        """Compose elements by index: returns idx(elements[i] * elements[j])."""
        return self.cayley[i, j].item()


class D4Group:
    """D4 = dihedral group of order 8 (symmetries of a square).

    Elements: {e, r, r^2, r^3, s, sr, sr^2, sr^3}
    where r = 90-degree rotation, s = reflection.

    Generators: r (rotation), s (reflection)

    D4 includes both rotations and reflections, making it a good test
    for the sign-flipping capability of B_n (reflections = sign flips).
    """

    def __init__(self):
        # Represent elements as (rotation_count, is_reflected)
        # rotation_count in {0, 1, 2, 3}, is_reflected in {False, True}
        self.elements = [
            (0, False),  # e
            (1, False),  # r
            (2, False),  # r^2
            (3, False),  # r^3
            (0, True),   # s
            (1, True),   # sr
            (2, True),   # sr^2
            (3, True),   # sr^3
        ]
        self.n_elements = 8

        # Generators: r (rotation by 90), s (reflection)
        self.generators = [
            (1, False),  # r
            (0, True),   # s
        ]
        self.n_generators = 2

        # Map from element to index
        self.elem_to_idx = {e: i for i, e in enumerate(self.elements)}

        # Precompute Cayley table
        self.cayley = self._build_cayley_table()

    def _compose(self, a: tuple, b: tuple) -> tuple:
        """Compose two D4 elements.

        D4 multiplication rules:
            r^a * r^b = r^{(a+b) mod 4}
            r^a * s*r^b = s*r^{(-a+b) mod 4}
            s*r^a * r^b = s*r^{(a+b) mod 4}
            s*r^a * s*r^b = r^{(-a+b) mod 4}
        """
        ra, sa = a  # rotation, is_reflected
        rb, sb = b

        if not sa and not sb:
            # r^a * r^b = r^{a+b}
            return ((ra + rb) % 4, False)
        elif not sa and sb:
            # r^a * s*r^b = s*r^{(-a+b) mod 4}
            return ((-ra + rb) % 4, True)
        elif sa and not sb:
            # s*r^a * r^b = s*r^{a+b}
            return ((ra + rb) % 4, True)
        else:
            # s*r^a * s*r^b = r^{(-a+b) mod 4}
            return ((-ra + rb) % 4, False)

    def _build_cayley_table(self) -> torch.Tensor:
        """Build 8x8 Cayley table."""
        table = torch.zeros(self.n_elements, self.n_elements, dtype=torch.long)
        for i, a in enumerate(self.elements):
            for j, b in enumerate(self.elements):
                product = self._compose(a, b)
                table[i, j] = self.elem_to_idx[product]
        return table

    def compose_idx(self, i: int, j: int) -> int:
        """Compose elements by index."""
        return self.cayley[i, j].item()


# ============================================================
# Dataset generation
# ============================================================

class GroupStateTrackingDataset(Dataset):
    """Dataset for group state tracking (scan/prefix composition).

    Each sample is a sequence of generator tokens, with targets being
    the prefix compositions.

    Token encoding:
        0..n_generators-1 = generator tokens

    Target encoding:
        0..n_elements-1 = group element index (classification)

    Args:
        group_name: "s3" or "d4"
        num_samples: Number of sequences to generate
        seq_len: Length of each sequence
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        group_name: str = "s3",
        num_samples: int = 5000,
        seq_len: int = 32,
        seed: int = 42,
    ):
        self.group_name = group_name
        self.seq_len = seq_len

        if group_name == "s3":
            self.group = S3Group()
        elif group_name == "d4":
            self.group = D4Group()
        else:
            raise ValueError(f"Unknown group: {group_name}")

        self.n_generators = self.group.n_generators
        self.n_elements = self.group.n_elements
        self.vocab_size = self.n_generators  # only generator tokens
        self.num_classes = self.n_elements

        # Generate data
        rng = random.Random(seed)
        self.tokens_list = []
        self.targets_list = []

        for _ in range(num_samples):
            # Random sequence of generators
            gen_indices = [rng.randint(0, self.n_generators - 1) for _ in range(seq_len)]

            # Compute prefix compositions (scan)
            # Map generator index to group element index
            # Generator 0 -> elements index of generators[0]
            # Generator 1 -> elements index of generators[1]
            gen_to_elem = [
                self.group.elem_to_idx[g] for g in self.group.generators
            ]

            compositions = []
            current = gen_to_elem[gen_indices[0]]
            compositions.append(current)

            for t in range(1, seq_len):
                elem_idx = gen_to_elem[gen_indices[t]]
                current = self.group.compose_idx(current, elem_idx)
                compositions.append(current)

            self.tokens_list.append(torch.tensor(gen_indices, dtype=torch.long))
            self.targets_list.append(torch.tensor(compositions, dtype=torch.long))

        self.tokens = torch.stack(self.tokens_list)
        self.targets = torch.stack(self.targets_list)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.targets[idx]


def create_datasets(
    group_name: str = "s3",
    num_train: int = 5000,
    num_test: int = 1000,
    seq_len: int = 32,
    seed: int = 42,
) -> Tuple[GroupStateTrackingDataset, GroupStateTrackingDataset]:
    """Create train and test datasets for group state tracking.

    Args:
        group_name: "s3" or "d4"
        num_train: Number of training sequences
        num_test: Number of test sequences
        seq_len: Sequence length
        seed: Random seed

    Returns:
        (train_dataset, test_dataset)
    """
    train_ds = GroupStateTrackingDataset(
        group_name=group_name,
        num_samples=num_train,
        seq_len=seq_len,
        seed=seed,
    )
    test_ds = GroupStateTrackingDataset(
        group_name=group_name,
        num_samples=num_test,
        seq_len=seq_len,
        seed=seed + 1000,
    )
    return train_ds, test_ds


if __name__ == "__main__":
    # Verify group structures
    s3 = S3Group()
    print(f"S3: {s3.n_elements} elements, {s3.n_generators} generators")
    print(f"S3 Cayley table:\n{s3.cayley}")

    d4 = D4Group()
    print(f"\nD4: {d4.n_elements} elements, {d4.n_generators} generators")
    print(f"D4 Cayley table:\n{d4.cayley}")

    # Verify non-commutativity
    g0, g1 = 0, 1  # generators
    g0_elem = s3.elem_to_idx[s3.generators[0]]
    g1_elem = s3.elem_to_idx[s3.generators[1]]
    prod_01 = s3.compose_idx(g0_elem, g1_elem)
    prod_10 = s3.compose_idx(g1_elem, g0_elem)
    print(f"\nS3 non-commutativity: g0*g1={prod_01}, g1*g0={prod_10}, equal={prod_01==prod_10}")

    # Test dataset generation
    train_ds, test_ds = create_datasets("s3", num_train=100, seq_len=8)
    print(f"\nDataset: {len(train_ds)} train, {len(test_ds)} test")
    tokens, targets = train_ds[0]
    print(f"Sample tokens: {tokens}")
    print(f"Sample targets: {targets}")
