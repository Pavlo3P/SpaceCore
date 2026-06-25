from __future__ import annotations

import numpy as np


DEFAULT_SEED = 20240612


def seeded_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return an independent NumPy generator initialized from ``seed``."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}.")
    return np.random.default_rng(seed)


def resolve_rng(
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.random.Generator:
    """Resolve the common generator convention without hidden global randomness."""
    if seed is not None and rng is not None:
        raise TypeError("Pass either seed or rng, not both.")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng must be numpy.random.Generator, got {type(rng).__name__}.")
        return rng
    return seeded_rng(DEFAULT_SEED if seed is None else seed)
