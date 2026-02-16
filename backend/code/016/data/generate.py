"""
S5 Permutation Composition Task Generator

The symmetric group S5 has 120 elements (all permutations of 5 objects).
S5 is the smallest non-solvable group, making it the canonical benchmark for
testing whether SSMs can represent non-abelian group operations.

Diagonal SSMs (abelian) provably cannot represent S5.

Task:
- Input: sequence of S5 generators
- Target: prefix composition at each position (scan-style prediction)
  i.e., at position t, predict g_1 * g_2 * ... * g_t

Generators:
- g0 = (0 1 2 3 4) — 5-cycle (order 5)
- g1 = (0 1) — transposition (order 2)
These two generate all of S5.
"""

import itertools
import torch
from torch.utils.data import Dataset, DataLoader


# ---- S5 Group Implementation ----

def compose_perms(p1, p2):
    """Compose two permutations: (p2 o p1)[i] = p2[p1[i]]."""
    return tuple(p2[p1[i]] for i in range(len(p1)))


def inverse_perm(p):
    """Compute inverse permutation."""
    inv = [0] * len(p)
    for i, v in enumerate(p):
        inv[v] = i
    return tuple(inv)


def perm_to_index(perm, perm_to_idx):
    """Convert permutation tuple to index."""
    return perm_to_idx[perm]


def build_s5():
    """
    Build S5 group: all 120 permutations of 5 elements.

    Returns:
        elements: list of 120 permutation tuples
        perm_to_idx: dict mapping permutation tuple -> index
        generators: list of generator permutation tuples
        gen_names: list of generator name strings
    """
    # Generators
    cycle5 = (1, 2, 3, 4, 0)   # (0 1 2 3 4) — 5-cycle
    swap01 = (1, 0, 2, 3, 4)   # (0 1) — transposition

    generators = [cycle5, swap01]
    gen_names = ["cycle5", "swap01"]

    # BFS to enumerate all S5 elements
    identity = (0, 1, 2, 3, 4)
    elements = set()
    elements.add(identity)
    queue = [identity]

    while queue:
        current = queue.pop(0)
        for gen in generators:
            new = compose_perms(current, gen)
            if new not in elements:
                elements.add(new)
                queue.append(new)

    elements = sorted(elements)  # Canonical ordering
    assert len(elements) == 120, f"Expected 120 S5 elements, got {len(elements)}"

    perm_to_idx = {p: i for i, p in enumerate(elements)}

    return elements, perm_to_idx, generators, gen_names


# ---- Dataset ----

# Special tokens
BOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2
GEN_OFFSET = 3  # generators start at index 3

IGNORE_INDEX = -100


class S5CompositionDataset(Dataset):
    """
    S5 permutation composition dataset.

    Each sample is a sequence of S5 generators. The target at each position
    is the index of the prefix composition up to that point.

    Tokens: BOS(0), EOS(1), PAD(2), gen0(3), gen1(4)
    Targets: IGNORE at BOS, composition index at each gen position, IGNORE at EOS/PAD
    """

    def __init__(self, num_samples, seq_len, max_seq_len=None):
        """
        Args:
            num_samples: number of sequences to generate
            seq_len: number of generator tokens per sequence
            max_seq_len: total padded length (default: seq_len + 2 for BOS/EOS)
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len or (seq_len + 2)

        # Build S5 group
        self.elements, self.perm_to_idx, self.generators, self.gen_names = build_s5()
        self.num_classes = len(self.elements)  # 120
        self.num_generators = len(self.generators)  # 2
        self.num_tokens = GEN_OFFSET + self.num_generators  # 5 total tokens

        # Pre-generate all samples
        self.tokens_list = []
        self.targets_list = []
        self.masks_list = []

        for _ in range(num_samples):
            tokens, targets, mask = self._generate_sample()
            self.tokens_list.append(tokens)
            self.targets_list.append(targets)
            self.masks_list.append(mask)

    def _generate_sample(self):
        """Generate a single training sample."""
        # Random sequence of generator indices
        gen_indices = torch.randint(0, self.num_generators, (self.seq_len,))

        # Build token sequence: BOS gen0 gen1 gen2 ... EOS PAD PAD ...
        tokens = torch.full((self.max_seq_len,), PAD_IDX, dtype=torch.long)
        tokens[0] = BOS_IDX
        for i in range(self.seq_len):
            tokens[i + 1] = gen_indices[i].item() + GEN_OFFSET
        tokens[self.seq_len + 1] = EOS_IDX

        # Build targets: IGNORE at BOS, composition index at each gen position
        targets = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)

        # Compute prefix compositions
        current = (0, 1, 2, 3, 4)  # identity
        for i in range(self.seq_len):
            gen = self.generators[gen_indices[i].item()]
            current = compose_perms(current, gen)
            targets[i + 1] = self.perm_to_idx[current]

        # Mask: 1 for real tokens, 0 for padding
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)
        mask[:self.seq_len + 2] = 1.0  # BOS + generators + EOS

        return tokens, targets, mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.tokens_list[idx], self.targets_list[idx], self.masks_list[idx]


def create_dataloaders(
    num_train=10000,
    num_val=1000,
    num_test=1000,
    seq_len=20,
    batch_size=128,
    max_seq_len=None,
):
    """
    Create train/val/test dataloaders for S5 composition task.

    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    max_seq_len = max_seq_len or (seq_len + 2)

    train_dataset = S5CompositionDataset(num_train, seq_len, max_seq_len)
    val_dataset = S5CompositionDataset(num_val, seq_len, max_seq_len)
    test_dataset = S5CompositionDataset(num_test, seq_len, max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataset_info = {
        "num_tokens": train_dataset.num_tokens,  # 5
        "num_classes": train_dataset.num_classes,  # 120
        "num_generators": train_dataset.num_generators,  # 2
        "seq_len": seq_len,
        "max_seq_len": max_seq_len,
    }

    return train_loader, val_loader, test_loader, dataset_info


if __name__ == "__main__":
    # Quick test
    elements, perm_to_idx, generators, gen_names = build_s5()
    print(f"S5 has {len(elements)} elements")
    print(f"Identity index: {perm_to_idx[(0, 1, 2, 3, 4)]}")
    print(f"Generators: {gen_names}")

    # Test dataset
    dataset = S5CompositionDataset(num_samples=10, seq_len=20)
    tokens, targets, mask = dataset[0]
    print(f"\nSample tokens: {tokens[:25]}")
    print(f"Sample targets: {targets[:25]}")
    print(f"Sample mask: {mask[:25]}")
    print(f"Num tokens: {dataset.num_tokens}, Num classes: {dataset.num_classes}")

    # Verify composition
    identity = (0, 1, 2, 3, 4)
    g0 = generators[0]  # cycle5
    g1 = generators[1]  # swap01
    print(f"\ncycle5: {g0}")
    print(f"swap01: {g1}")
    print(f"cycle5 * swap01: {compose_perms(g0, g1)}")
    print(f"swap01 * cycle5: {compose_perms(g1, g0)}")
    print(f"Non-commutative: {compose_perms(g0, g1) != compose_perms(g1, g0)}")
