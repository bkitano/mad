"""
Base classes for token systems.

Provides a common interface for different group-based and synthetic
token systems so they can be used interchangeably with models.
"""

from abc import ABC, abstractmethod


class TokenSystem(ABC):
    """
    Abstract base class for token systems.

    All token systems must provide:
    - num_tokens: total vocabulary size (group elements + special tokens)
    - num_group_elements: number of group elements (also num_classes for output)
    - BOS_IDX, EOS_IDX, PAD_IDX: special token indices
    - identity_idx: index of the identity element
    """

    num_tokens: int
    num_group_elements: int
    BOS_IDX: int
    EOS_IDX: int
    PAD_IDX: int
    identity_idx: int

    @property
    def num_classes(self) -> int:
        """Alias for num_group_elements (for model compatibility)."""
        return self.num_group_elements

    @abstractmethod
    def get_random_index(self) -> int:
        """Get a random group element index."""
        pass

    @abstractmethod
    def token_string(self, idx: int) -> str:
        """Convert token index to string representation."""
        pass
