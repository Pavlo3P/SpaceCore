from __future__ import annotations

from typing import Any


def _check_batched(space: Any, xs: Any) -> None:
    """Raise if ``xs`` does not have ``space.shape`` as trailing dimensions."""
    parts = getattr(space, "spaces", None)
    if parts is not None:
        if not isinstance(xs, tuple) or len(xs) != len(parts):
            raise ValueError(
                f"Batched product value must be a tuple of length {len(parts)}, got {type(xs).__name__}."
            )
        for part, xi in zip(parts, xs):
            _check_batched(part, xi)
        return
    base = tuple(space.shape)
    shape = tuple(getattr(xs, "shape", ()))
    if base and shape[-len(base):] != base:
        raise ValueError(
            f"Batched value trailing shape must be {base}, got {getattr(xs, 'shape', None)}."
        )
