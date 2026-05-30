from __future__ import annotations

from typing import Any


def _check_batched(space: Any, xs: Any) -> None:
    """Raise if ``xs`` does not have ``space.shape`` as trailing dimensions."""
    base = tuple(space.shape)
    shape = tuple(getattr(xs, "shape", ()))
    if base and shape[-len(base):] != base:
        raise ValueError(
            f"Batched value trailing shape must be {base}, got {getattr(xs, 'shape', None)}."
        )
