"""
S3 Symmetric Group Token System

The symmetric group S3 (permutations of 3 elements) has 6 elements:
- e: identity (0,1,2)
- (12): swap 0 and 1 -> (1,0,2)
- (13): swap 0 and 2 -> (2,1,0)
- (23): swap 1 and 2 -> (0,2,1)
- (123): cycle 0->1->2->0 -> (1,2,0)
- (132): cycle 0->2->1->0 -> (2,0,1)

This is the smallest non-abelian group (6 elements).
Used to test state tracking: can a model compose permutations sequentially?

Token indices 0-5 map to group elements, 6-8 are special tokens.
"""

import random
from typing import Tuple


# Represent permutations as tuples: (a,b,c) means 0->a, 1->b, 2->c
S3_ELEMENTS = [
    (0, 1, 2),  # e (identity)
    (1, 0, 2),  # (12)
    (2, 1, 0),  # (13)
    (0, 2, 1),  # (23)
    (1, 2, 0),  # (123)
    (2, 0, 1),  # (132)
]

S3_NAMES = ["e", "(12)", "(13)", "(23)", "(123)", "(132)"]

# Generators of S3: (12) and (123) generate all of S3
S3_GENERATOR_INDICES = [1, 4]  # (12) and (123)


def s3_multiply(g1: tuple, g2: tuple) -> tuple:
    """
    Compose two permutations: g1 * g2 (apply g2 first, then g1).

    If g1 = (a,b,c) and g2 = (d,e,f), then:
    g1 * g2 maps i -> g1(g2(i))
    """
    return tuple(g1[g2[i]] for i in range(3))


def build_multiplication_table() -> dict:
    """Build lookup table for all S3 products."""
    elem_to_idx = {elem: i for i, elem in enumerate(S3_ELEMENTS)}
    table = {}
    for i, g1 in enumerate(S3_ELEMENTS):
        for j, g2 in enumerate(S3_ELEMENTS):
            product = s3_multiply(g1, g2)
            table[(i, j)] = elem_to_idx[product]
    return table


MULT_TABLE = build_multiplication_table()


class S3TokenSystem:
    """
    Token system for S3 symmetric group.

    Token layout:
    - 0-5: Group elements (e, (12), (13), (23), (123), (132))
    - 6: BOS (begin-of-sequence)
    - 7: EOS (end-of-sequence)
    - 8: PAD (padding)
    """

    BOS_IDX = 6
    EOS_IDX = 7
    PAD_IDX = 8

    def __init__(self):
        self.all_elements = S3_ELEMENTS
        self.num_group_elements = 6
        self.num_tokens = 9  # 6 group elements + BOS + EOS + PAD
        self.identity_idx = 0
        self.generators = S3_GENERATOR_INDICES

    def compose_indices(self, i: int, j: int) -> int:
        """Compose two group elements given their indices."""
        return MULT_TABLE[(i, j)]

    def compose_sequence(self, indices: list) -> int:
        """Compose a sequence of group elements (left to right)."""
        result = self.identity_idx
        for idx in indices:
            result = self.compose_indices(result, idx)
        return result

    def scan_sequence(self, indices: list) -> list:
        """Return all prefix compositions: [g1, g1*g2, g1*g2*g3, ...]."""
        result = self.identity_idx
        scan = []
        for idx in indices:
            result = self.compose_indices(result, idx)
            scan.append(result)
        return scan

    def token_string(self, idx: int) -> str:
        """Convert token index to string representation."""
        if idx == self.BOS_IDX:
            return "<BOS>"
        elif idx == self.EOS_IDX:
            return "<EOS>"
        elif idx == self.PAD_IDX:
            return "<PAD>"
        else:
            return S3_NAMES[idx]

    def get_random_index(self) -> int:
        """Get a random group element index (0-5)."""
        return random.randint(0, self.num_group_elements - 1)

    def get_generator_index(self) -> int:
        """Get a random generator index."""
        return random.choice(self.generators)


if __name__ == "__main__":
    ts = S3TokenSystem()
    print("S3 Token System")
    print("=" * 40)
    print(f"Elements: {ts.num_group_elements}")
    print(f"Tokens: {ts.num_tokens}")

    print("\nElement mapping:")
    for i in range(ts.num_group_elements):
        print(f"  {i}: {ts.token_string(i)} = {S3_ELEMENTS[i]}")

    print("\nMultiplication table:")
    header = "     " + " ".join(f"{S3_NAMES[j]:>5}" for j in range(6))
    print(header)
    for i in range(6):
        row = f"{S3_NAMES[i]:>5}"
        for j in range(6):
            prod = ts.compose_indices(i, j)
            row += f" {S3_NAMES[prod]:>5}"
        print(row)

    print("\nScan example: (12), (123), (13)")
    seq = [1, 4, 2]
    scan = ts.scan_sequence(seq)
    for i, (tok, prefix) in enumerate(zip(seq, scan)):
        print(f"  After {i+1}: {ts.token_string(prefix)}")
