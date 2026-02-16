"""D4 Dihedral Group Task - Symmetries of a Square.

The dihedral group D4 has 8 elements:
- e: identity
- r, r², r³: rotations by 90°, 180°, 270°
- s: reflection across vertical axis
- sr, sr², sr³: reflection composed with rotations

This is the smallest non-abelian group requiring both:
1. Permutation routing (rotations swap corners)
2. Sign-flipping dynamics (reflections change orientation)

Group relations:
- r⁴ = e (rotation order 4)
- s² = e (reflection order 2)
- srs = r⁻¹ (conjugation relation)
"""

from .tokens import D4TokenSystem
from .dataset import D4CurriculumWrapper, D4FixedKDataset

__all__ = ["D4TokenSystem", "D4CurriculumWrapper", "D4FixedKDataset"]
