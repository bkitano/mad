"""
B3 Hyperoctahedral Group Task.

The hyperoctahedral group B_n = Z_2^n ⋊ S_n consists of signed permutations.
For n=3, |B_3| = 2^3 * 3! = 48 elements.

Each element is a pair (signs, perm) where:
  - signs ∈ {-1, +1}^3: coordinate sign flips
  - perm ∈ S_3: permutation of coordinates

Composition: (s1, p1) * (s2, p2) = (s1 * s2[p1^{-1}], p1 ∘ p2)
  i.e., first permute by p2, then p1, and combine signs with permutation.

Generators of B_3:
  - σ_1 = (12) transposition: swap positions 0,1 (signs all +1)
  - σ_2 = (23) transposition: swap positions 1,2 (signs all +1)
  - τ = sign flip on coord 0: signs = [-1, +1, +1], perm = identity

These 3 generators produce all 48 elements of B_3.

Task: Given a sequence of generators [g_1, g_2, ..., g_k], compute the
composed element g_1 * g_2 * ... * g_k and classify it (48 classes).
"""

import itertools
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple


class B3Element:
    """An element of B_3 = Z_2^3 ⋊ S_3."""

    def __init__(self, signs: Tuple[int, ...], perm: Tuple[int, ...]):
        """
        Args:
            signs: tuple of 3 values in {-1, +1}
            perm: tuple of 3 values, a permutation of (0, 1, 2)
        """
        assert len(signs) == 3 and all(s in (-1, 1) for s in signs)
        assert sorted(perm) == [0, 1, 2]
        self.signs = tuple(signs)
        self.perm = tuple(perm)

    def compose(self, other: 'B3Element') -> 'B3Element':
        """
        Compute self * other in B_3.

        If self = (s1, p1) and other = (s2, p2), then:
          self * other = (s_new, p1 ∘ p2)
        where s_new[i] = s1[i] * s2[p1^{-1}(i)]

        Derivation: The signed permutation matrix M = D(s) P(π) acts as:
          M x = s ⊙ x[π]
        So M1 (M2 x) = s1 ⊙ (s2 ⊙ x[π2])[π1] = s1 ⊙ s2[π1] ⊙ x[π2[π1]]
        Combined: signs = s1 * s2[π1], perm = π2 ∘ π1 (apply π1 first, then π2)

        Wait, let's be careful. In our convention:
        - "compose" means apply self first, then other? Or other first?
        - We follow left-to-right: (g1 * g2)(x) means "apply g1, then g2"

        For left-to-right composition:
          (s1, p1) * (s2, p2):
          First apply (s1, p1): y = s1 ⊙ x[p1]
          Then apply (s2, p2): z = s2 ⊙ y[p2] = s2 ⊙ (s1 ⊙ x[p1])[p2]
                               = s2 ⊙ s1[p2] ⊙ x[p1[p2]]
          So: new_signs[i] = s2[i] * s1[p2[i]], new_perm = p1 ∘ p2 (i.e., p1[p2[i]])
        """
        # p1 ∘ p2: apply p2 first (index), then p1
        new_perm = tuple(self.perm[other.perm[i]] for i in range(3))
        # signs: s2[i] * s1[p2[i]]
        new_signs = tuple(other.signs[i] * self.signs[other.perm[i]] for i in range(3))
        return B3Element(new_signs, new_perm)

    def __eq__(self, other):
        return self.signs == other.signs and self.perm == other.perm

    def __hash__(self):
        return hash((self.signs, self.perm))

    def __repr__(self):
        return f"B3({self.signs}, {self.perm})"

    def to_index(self, element_list: List['B3Element']) -> int:
        """Get the index of this element in the enumeration."""
        for i, e in enumerate(element_list):
            if self == e:
                return i
        raise ValueError(f"Element {self} not found in list")


class B3Group:
    """The hyperoctahedral group B_3 with 48 elements."""

    def __init__(self):
        # Enumerate all 48 elements
        self.elements = []
        for signs in itertools.product((-1, 1), repeat=3):
            for perm in itertools.permutations(range(3)):
                self.elements.append(B3Element(signs, perm))

        assert len(self.elements) == 48, f"Expected 48 elements, got {len(self.elements)}"

        # Create lookup dict for fast indexing
        self._index = {e: i for i, e in enumerate(self.elements)}

        # Define generators
        self.identity = B3Element((1, 1, 1), (0, 1, 2))

        # σ_1: transposition (0,1), no sign flips
        self.sigma1 = B3Element((1, 1, 1), (1, 0, 2))

        # σ_2: transposition (1,2), no sign flips
        self.sigma2 = B3Element((1, 1, 1), (0, 2, 1))

        # τ: sign flip on coordinate 0, identity permutation
        self.tau = B3Element((-1, 1, 1), (0, 1, 2))

        self.generators = [self.sigma1, self.sigma2, self.tau]
        self.num_generators = len(self.generators)

        # Verify generators produce all 48 elements
        self._verify_generators()

    def _verify_generators(self):
        """Verify that our generators produce all 48 elements."""
        generated = {self.identity}
        frontier = [self.identity]

        while frontier:
            new_frontier = []
            for elem in frontier:
                for gen in self.generators:
                    # Try both left and right multiplication
                    for product in [elem.compose(gen), gen.compose(elem)]:
                        if product not in generated:
                            generated.add(product)
                            new_frontier.append(product)
            frontier = new_frontier

        assert len(generated) == 48, f"Generators only produce {len(generated)} elements"

    def element_index(self, elem: B3Element) -> int:
        """Get the index (0-47) of a B3 element."""
        return self._index[elem]

    def compose_sequence(self, gen_indices: List[int]) -> B3Element:
        """Compose a sequence of generators (left-to-right)."""
        result = self.identity
        for idx in gen_indices:
            result = result.compose(self.generators[idx])
        return result

    def apply_to_vector(self, elem: B3Element, vec: np.ndarray) -> np.ndarray:
        """Apply signed permutation to a vector: result[i] = signs[i] * vec[perm[i]]."""
        result = np.zeros_like(vec)
        for i in range(3):
            result[i] = elem.signs[i] * vec[elem.perm[i]]
        return result


class B3CompositionDataset(Dataset):
    """
    Dataset for B3 composition task.

    Each sample: a sequence of generator indices -> class label (0-47).
    The class label is the index of the composed B3 element.

    Tokens:
      0, 1, 2: generators (σ_1, σ_2, τ)
      3: BOS
      4: EOS
      5: PAD

    Input format: [BOS, g_1, g_2, ..., g_k, EOS, PAD, ...]
    Target: predict the composition at each position (scan-style)
      [IGNORE, p_1, p_1*p_2, ..., p_1*...*p_k, IGNORE, IGNORE, ...]

    This matches the code/001 pattern of predicting prefix compositions.
    """

    GENERATOR_TOKENS = [0, 1, 2]  # σ_1, σ_2, τ
    BOS = 3
    EOS = 4
    PAD = 5
    NUM_TOKENS = 6
    NUM_CLASSES = 48  # |B_3|
    IGNORE_INDEX = -100

    def __init__(
        self,
        group: B3Group,
        num_samples: int = 10000,
        min_k: int = 8,
        max_k: int = 16,
        max_seq_len: int = 20,
        seed: int = 42,
    ):
        self.group = group
        self.max_seq_len = max_seq_len

        rng = np.random.RandomState(seed)

        self.inputs = []
        self.targets = []
        self.ks = []

        for _ in range(num_samples):
            k = rng.randint(min_k, max_k + 1)
            gen_indices = rng.randint(0, group.num_generators, size=k).tolist()

            # Build input sequence: [BOS, g_1, ..., g_k, EOS, PAD, ...]
            input_seq = [self.BOS] + gen_indices + [self.EOS]

            # Compute prefix compositions for targets
            target_seq = [self.IGNORE_INDEX]  # BOS position
            current = group.identity
            for idx in gen_indices:
                current = current.compose(group.generators[idx])
                target_seq.append(group.element_index(current))
            target_seq.append(self.IGNORE_INDEX)  # EOS position

            # Pad to max_seq_len
            pad_len = max_seq_len - len(input_seq)
            input_seq += [self.PAD] * pad_len
            target_seq += [self.IGNORE_INDEX] * pad_len

            # Truncate if needed
            input_seq = input_seq[:max_seq_len]
            target_seq = target_seq[:max_seq_len]

            self.inputs.append(input_seq)
            self.targets.append(target_seq)
            self.ks.append(k)

        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        self.ks = torch.tensor(self.ks, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.ks[idx]


def test_b3_group():
    """Verify B3 group operations."""
    group = B3Group()

    # Test identity
    for gen in group.generators:
        assert group.identity.compose(gen) == gen
        assert gen.compose(group.identity) == gen

    # Test that σ_1^2 = identity
    s1s1 = group.sigma1.compose(group.sigma1)
    assert s1s1 == group.identity, f"σ_1^2 = {s1s1}, expected identity"

    # Test that τ^2 = identity
    tt = group.tau.compose(group.tau)
    assert tt == group.identity, f"τ^2 = {tt}, expected identity"

    # Test non-commutativity: σ_1 * τ ≠ τ * σ_1
    s1t = group.sigma1.compose(group.tau)
    ts1 = group.tau.compose(group.sigma1)
    assert s1t != ts1, "σ_1 and τ should not commute"

    # Test vector application
    vec = np.array([1.0, 2.0, 3.0])

    # σ_1 should swap positions 0,1: [2, 1, 3]
    result = group.apply_to_vector(group.sigma1, vec)
    np.testing.assert_array_equal(result, [2.0, 1.0, 3.0])

    # τ should negate position 0: [-1, 2, 3]
    result = group.apply_to_vector(group.tau, vec)
    np.testing.assert_array_equal(result, [-1.0, 2.0, 3.0])

    # Test composition
    dataset = B3CompositionDataset(group, num_samples=100, min_k=4, max_k=8, max_seq_len=12)
    assert len(dataset) == 100
    inp, tgt, k = dataset[0]
    assert inp[0] == B3CompositionDataset.BOS

    print("All B3 group tests passed!")
    return group


if __name__ == "__main__":
    test_b3_group()
