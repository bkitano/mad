"""
D4 Dihedral Group Dataset for State Tracking

Implements curriculum learning datasets for the D4 state tracking task.
The task is to predict the cumulative product of a sequence of D4 group elements.

Input: [BOS, g_1, g_2, ..., g_k, EOS, PAD, ...]
Output (scan): [IGNORE, p_1, p_2, ..., p_k, IGNORE, IGNORE, ...] where p_i = g_1 * g_2 * ... * g_i

This tests whether models can track state through:
1. Permutation routing (rotations)
2. Sign-flipping dynamics (reflections)
"""

import random
from typing import Optional

import torch
from torch.utils.data import Dataset

from tasks.d4.tokens import D4TokenSystem


IGNORE_INDEX = -100  # Standard ignore index for CrossEntropyLoss


class D4FixedKDataset(Dataset):
    """
    Dataset with fixed sequence length k.

    Produces sequences in format: [BOS, g_1, g_2, ..., g_k, EOS, PAD, ...]
    """

    def __init__(
        self,
        token_system: D4TokenSystem,
        k: int,
        num_samples: int,
        max_seq_len: int,
        use_generators_only: bool = False,
    ):
        self.token_system = token_system
        self.k = k
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.use_generators_only = use_generators_only
        self.data = self._generate()

        assert max_seq_len >= k + 2, f"max_seq_len ({max_seq_len}) must be >= k + 2 ({k + 2})"

    def _generate(self) -> list[tuple[list[int], list[int]]]:
        """Generate all samples: (list of elements, scan of prefix products)."""
        data = []
        for _ in range(self.num_samples):
            if self.use_generators_only:
                # Only use r and s as inputs (generates all of D4)
                elements = [self.token_system.get_generator_index() for _ in range(self.k)]
            else:
                # Use all 8 elements uniformly
                elements = [self.token_system.get_random_index() for _ in range(self.k)]
            scan = self.token_system.scan_sequence(elements)
            data.append((elements, scan))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            tokens: (max_seq_len,) tensor: [BOS, g_1, ..., g_k, EOS, PAD, ...]
            target: (max_seq_len,) tensor with scan values, IGNORE_INDEX at BOS/EOS/PAD
            mask: (max_seq_len,) tensor (1 for real tokens incl BOS/EOS, 0 for PAD)
            k: the number of group elements (not counting BOS/EOS)
        """
        elements, scan = self.data[idx]

        # Initialize with PAD tokens
        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        target = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        # BOS
        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0
        # target[0] = IGNORE (no prediction at BOS)

        # Group elements with scan targets
        for i, (elem, scan_val) in enumerate(zip(elements, scan)):
            tokens[i + 1] = elem
            target[i + 1] = scan_val  # Predict prefix product
            mask[i + 1] = 1.0

        # EOS
        eos_pos = len(elements) + 1
        tokens[eos_pos] = self.token_system.EOS_IDX
        mask[eos_pos] = 1.0
        # target[eos_pos] = IGNORE (no prediction at EOS)

        return tokens, target, mask, self.k


class _D4StageDataset(Dataset):
    """Internal dataset for a single curriculum stage."""

    def __init__(
        self,
        token_system: D4TokenSystem,
        max_seq_len: int,
        data: list[tuple[list[int], list[int], int]],  # (elements, scan, k)
    ):
        self.token_system = token_system
        self.max_seq_len = max_seq_len
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        elements, scan, k = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        target = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        for i, (elem, scan_val) in enumerate(zip(elements, scan)):
            tokens[i + 1] = elem
            target[i + 1] = scan_val
            mask[i + 1] = 1.0

        eos_pos = len(elements) + 1
        tokens[eos_pos] = self.token_system.EOS_IDX
        mask[eos_pos] = 1.0

        return tokens, target, mask, k


class D4CurriculumWrapper:
    """
    Curriculum learning wrapper for D4 state tracking.

    Provides get_stage(k) interface for staged curriculum training.
    Stage k includes all samples from k=1 to k=stage_k.
    """

    def __init__(
        self,
        token_system: D4TokenSystem,
        max_k: int,
        samples_per_k: int,
        max_seq_len: int,
        test_size: float = 0.2,
        use_generators_only: bool = False,
        fixed_k: Optional[int] = None,
    ):
        self.token_system = token_system
        self.max_k = max_k
        self.samples_per_k = samples_per_k
        self.max_seq_len = max_seq_len
        self.test_size = test_size
        self.use_generators_only = use_generators_only
        self.fixed_k = fixed_k
        self._data_by_k = self._generate_all_data()

    def _generate_all_data(self) -> dict[int, list[tuple[list[int], list[int]]]]:
        """Generate and store data for each k value."""
        data_by_k = {}

        # If fixed_k is set, only generate data for that k
        if self.fixed_k is not None:
            k_values = [self.fixed_k]
        else:
            k_values = range(1, self.max_k + 1)

        for k in k_values:
            k_data = []
            for _ in range(self.samples_per_k):
                if self.use_generators_only:
                    elements = [self.token_system.get_generator_index() for _ in range(k)]
                else:
                    elements = [self.token_system.get_random_index() for _ in range(k)]
                scan = self.token_system.scan_sequence(elements)
                k_data.append((elements, scan))
            data_by_k[k] = k_data
        return data_by_k

    def get_stage(self, stage_k: int) -> tuple[_D4StageDataset, _D4StageDataset]:
        """
        Get train/test datasets for curriculum stage k.

        Stage k includes all samples with k values from 1 to stage_k.
        """
        assert 1 <= stage_k <= self.max_k, f"stage_k must be in [1, {self.max_k}]"

        train_data = []
        test_data = []

        for k in range(1, stage_k + 1):
            k_data = self._data_by_k[k]
            n_test = int(len(k_data) * self.test_size)

            indices = list(range(len(k_data)))
            random.shuffle(indices)

            test_indices = indices[:n_test]
            train_indices = indices[n_test:]

            for idx in train_indices:
                elements, scan = k_data[idx]
                train_data.append((elements, scan, k))
            for idx in test_indices:
                elements, scan = k_data[idx]
                test_data.append((elements, scan, k))

        random.shuffle(train_data)
        random.shuffle(test_data)

        train_ds = _D4StageDataset(self.token_system, self.max_seq_len, train_data)
        test_ds = _D4StageDataset(self.token_system, self.max_seq_len, test_data)

        return train_ds, test_ds

    def get_fixed_k(self, k: int) -> tuple[_D4StageDataset, _D4StageDataset]:
        """
        Get train/test datasets for a fixed k value only (non-curriculum mode).
        """
        assert k in self._data_by_k, f"k={k} not in generated data"

        k_data = self._data_by_k[k]
        n_test = int(len(k_data) * self.test_size)

        indices = list(range(len(k_data)))
        random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        train_data = [(k_data[idx][0], k_data[idx][1], k) for idx in train_indices]
        test_data = [(k_data[idx][0], k_data[idx][1], k) for idx in test_indices]

        random.shuffle(train_data)
        random.shuffle(test_data)

        train_ds = _D4StageDataset(self.token_system, self.max_seq_len, train_data)
        test_ds = _D4StageDataset(self.token_system, self.max_seq_len, test_data)

        return train_ds, test_ds

    def num_stages(self) -> int:
        """Return the number of curriculum stages (equal to max_k)."""
        return self.max_k


if __name__ == "__main__":
    from tasks.d4.tokens import D4TokenSystem

    token_system = D4TokenSystem()

    print("D4 Dataset Test")
    print("=" * 40)

    # Test fixed-k dataset
    dataset = D4FixedKDataset(
        token_system=token_system,
        k=5,
        num_samples=100,
        max_seq_len=12,
        use_generators_only=True,
    )

    print(f"\nFixed k=5 dataset: {len(dataset)} samples")
    tokens, target, mask, k = dataset[0]
    print(f"Sample 0:")
    print(f"  tokens: {tokens.tolist()}")
    print(f"  target: {target.tolist()}")
    print(f"  mask: {mask.tolist()}")
    print(f"  k: {k}")

    # Decode sample
    print(f"  Decoded:")
    for i, (tok, tgt) in enumerate(zip(tokens.tolist(), target.tolist())):
        if tok != token_system.PAD_IDX:
            tgt_str = token_system.token_string(tgt) if tgt != IGNORE_INDEX else "IGNORE"
            print(f"    pos {i}: {token_system.token_string(tok)} -> {tgt_str}")

    # Test curriculum wrapper
    curriculum = D4CurriculumWrapper(
        token_system=token_system,
        max_k=5,
        samples_per_k=100,
        max_seq_len=12,
        test_size=0.2,
        use_generators_only=True,
    )

    print(f"\nCurriculum: {curriculum.num_stages()} stages")
    for stage in range(1, curriculum.num_stages() + 1):
        train_ds, test_ds = curriculum.get_stage(stage)
        print(f"  Stage {stage}: {len(train_ds)} train, {len(test_ds)} test")
