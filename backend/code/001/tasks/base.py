"""Base token system protocol."""

from typing import Protocol


class TokenSystem(Protocol):
    """Protocol for token systems (D4, S5, Zn, etc.)."""

    num_tokens: int
    num_group_elements: int
    EOS_IDX: int
    PAD_IDX: int
    BOS_IDX: int
    identity_idx: int

    def get_random_index(self) -> int:
        """Get a random group element index."""
        ...

    def compose_sequence(self, indices: list[int]) -> int:
        """Compose a sequence of group elements."""
        ...

    def scan_sequence(self, indices: list[int]) -> list[int]:
        """Return all prefix compositions."""
        ...

    def token_string(self, idx: int) -> str:
        """Convert token index to string representation."""
        ...
