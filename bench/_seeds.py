"""Fixed seeds for reproducible benchmarks.

Every probe is run on the same four seeds in every benchmark run. The
goal is dual: (1) stability — random noise on a single seed can hide a
real change, (2) reproducibility — anyone can regenerate exactly the
same case set.

The seed tuple is intentionally short. A larger set would slow the suite
down without adding much variance information; the four-point sample
shows whether a result is seed-sensitive without becoming the dominant
runtime cost of a run.
"""
from __future__ import annotations

from typing import Final

SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3)
"""The seed tuple every probe is run on."""


def is_canonical_seed(seed: int) -> bool:
    """Return whether ``seed`` is one of the canonical bench seeds."""
    return seed in SEEDS
