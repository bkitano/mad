"""
MQAR (Multi-Query Associative Recall) data generator for Experiment 006.

Task: Given key-value pairs followed by query keys, recall the associated value.
Format: [k1, v1, k2, v2, ..., kN, vN, SEP, q1, q2, ..., qM]
Target at query positions: val(qi) for each query qi.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

PAD_IDX = 0
SEP_IDX = 1
SPECIAL_TOKENS = 2


class MQARDataset(Dataset):
    def __init__(self, num_samples, num_kv_pairs=8, num_queries=8,
                 vocab_size=64, seed=42):
        assert num_queries <= num_kv_pairs
        self.num_samples = num_samples
        self.num_kv_pairs = num_kv_pairs
        self.num_queries = num_queries
        self.vocab_size = vocab_size
        self.total_vocab = vocab_size + SPECIAL_TOKENS
        self.seq_len = 2 * num_kv_pairs + 1 + num_queries

        rng = np.random.RandomState(seed)
        self.inputs, self.targets = self._generate(rng)

    def _generate(self, rng):
        inputs = np.zeros((self.num_samples, self.seq_len), dtype=np.int64)
        targets = np.full((self.num_samples, self.seq_len), -1, dtype=np.int64)

        for i in range(self.num_samples):
            keys = rng.choice(self.vocab_size, size=self.num_kv_pairs, replace=False) + SPECIAL_TOKENS
            values = rng.randint(0, self.vocab_size, size=self.num_kv_pairs) + SPECIAL_TOKENS

            for j in range(self.num_kv_pairs):
                inputs[i, 2 * j] = keys[j]
                inputs[i, 2 * j + 1] = values[j]

            sep_pos = 2 * self.num_kv_pairs
            inputs[i, sep_pos] = SEP_IDX

            query_indices = rng.choice(self.num_kv_pairs, size=self.num_queries, replace=False)
            for j, qi in enumerate(query_indices):
                query_pos = sep_pos + 1 + j
                inputs[i, query_pos] = keys[qi]
                targets[i, query_pos] = values[qi]

        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def generate_mqar_data(num_train=10000, num_val=1000, num_test=1000,
                       num_kv_pairs=8, num_queries=8, vocab_size=64,
                       batch_size=128, seed=42):
    train_ds = MQARDataset(num_train, num_kv_pairs, num_queries, vocab_size, seed=seed)
    val_ds = MQARDataset(num_val, num_kv_pairs, num_queries, vocab_size, seed=seed + 1)
    test_ds = MQARDataset(num_test, num_kv_pairs, num_queries, vocab_size, seed=seed + 2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    dataset_info = {
        'total_vocab_size': train_ds.total_vocab,
        'seq_len': train_ds.seq_len,
        'num_kv_pairs': num_kv_pairs,
        'num_queries': num_queries,
        'vocab_size': vocab_size,
    }
    return train_loader, val_loader, test_loader, dataset_info


def generate_mqar_data_length(num_samples=1000, num_kv_pairs=32, num_queries=8,
                               vocab_size=64, batch_size=128, seed=42):
    ds = MQARDataset(num_samples, num_kv_pairs, num_queries, vocab_size, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    dataset_info = {
        'total_vocab_size': ds.total_vocab,
        'seq_len': ds.seq_len,
        'num_kv_pairs': num_kv_pairs,
        'num_queries': num_queries,
    }
    return loader, dataset_info
