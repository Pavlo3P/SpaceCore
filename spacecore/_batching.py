from __future__ import annotations

from typing import Any

from .space._checks import _run_checks


def _check_batched(space: Any, xs: Any) -> None:
    """Raise if ``xs`` does not have ``space.shape`` as trailing dimensions."""
    _run_checks(space, xs, allow_leading=True)


def _batched_inner(space: Any, xs: Any, ys: Any) -> Any:
    """Return ``space.inner(xs[i], ys[i])`` for a leading-axis batch."""
    xs_flat = space.flatten_batch(xs)
    ys_dual = ys if space.is_euclidean else space.riesz(ys)
    ys_flat = space.flatten_batch(ys_dual)
    return space.ops.sum(space.ops.conj(xs_flat) * ys_flat, axis=1)
