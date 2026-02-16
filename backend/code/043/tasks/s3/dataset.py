"""
S3 Symmetric Group Dataset for State Tracking

Input: [BOS, g_1, g_2, ..., g_k, EOS, PAD, ...]
Output (scan): [IGNORE, p_1, p_2, ..., p_k, IGNORE, IGNORE, ...]
where p_i = g_1 * g_2 * ... * g_i

Tests whether models can track non-abelian group state (permutation composition).
"""

import random
from typing import Optional

import torch
from torch.utils.data import Dataset

from tasks.s3.tokens import S3TokenSystem


IGNORE_INDEX = -100


class _S3StageDataset(Dataset):
    """Internal dataset for a single curriculum stage."""

    def __init__(
        self,
        token_system: S3TokenSystem,
        max_seq_len: int,
        data: list,  # list of (elements, scan, k)
    ):
        self.token_system = token_system
        self.max_seq_len = max_seq_len
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        elements, scan, k = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        target = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        # BOS
        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        # Group elements with scan targets
        for i, (elem, scan_val) in enumerate(zip(elements, scan)):
            tokens[i + 1] = elem
            target[i + 1] = scan_val
            mask[i + 1] = 1.0

        # EOS
        eos_pos = len(elements) + 1
        tokens[eos_pos] = self.token_system.EOS_IDX
        mask[eos_pos] = 1.0

        return tokens, target, mask, k


class S3CurriculumWrapper:
    """
    Curriculum learning wrapper for S3 state tracking.

    Stage k includes all samples from k=1 to k=stage_k.
    """

    def __init__(
        self,
        token_system: S3TokenSystem,
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

    def _generate_all_data(self) -> dict:
        """Generate and store data for each k value."""
        data_by_k = {}
        k_values = [self.fixed_k] if self.fixed_k is not None else range(1, self.max_k + 1)

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

    def get_stage(self, stage_k: int):
        """Get train/test datasets for curriculum stage k."""
        assert 1 <= stage_k <= self.max_k

        train_data = []
        test_data = []

        for k in range(1, stage_k + 1):
            k_data = self._data_by_k[k]
            n_test = int(len(k_data) * self.test_size)

            indices = list(range(len(k_data)))
            random.shuffle(indices)

            for idx in indices[n_test:]:
                elements, scan = k_data[idx]
                train_data.append((elements, scan, k))
            for idx in indices[:n_test]:
                elements, scan = k_data[idx]
                test_data.append((elements, scan, k))

        random.shuffle(train_data)
        random.shuffle(test_data)

        train_ds = _S3StageDataset(self.token_system, self.max_seq_len, train_data)
        test_ds = _S3StageDataset(self.token_system, self.max_seq_len, test_data)
        return train_ds, test_ds

    def get_fixed_k(self, k: int):
        """Get train/test datasets for a fixed k value only."""
        assert k in self._data_by_k

        k_data = self._data_by_k[k]
        n_test = int(len(k_data) * self.test_size)

        indices = list(range(len(k_data)))
        random.shuffle(indices)

        train_data = [(k_data[idx][0], k_data[idx][1], k) for idx in indices[n_test:]]
        test_data = [(k_data[idx][0], k_data[idx][1], k) for idx in indices[:n_test]]

        random.shuffle(train_data)
        random.shuffle(test_data)

        train_ds = _S3StageDataset(self.token_system, self.max_seq_len, train_data)
        test_ds = _S3StageDataset(self.token_system, self.max_seq_len, test_data)
        return train_ds, test_ds

    def num_stages(self) -> int:
        return self.max_k
