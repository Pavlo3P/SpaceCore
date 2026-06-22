from __future__ import annotations

import warnings
from typing import Any

from .space.checks import _run_checks

# Shared batched-evaluation helpers used by both LinOp and Functional. They live
# here (rather than in functional/_base) so the linop and functional batched
# methods share one implementation and one warn-once registry.
_VMAP_FALLBACK_WARNED: set[tuple[type, str]] = set()
_VMAP_FALLBACK_WARN_BATCH = 32


def _check_batched(space: Any, xs: Any) -> None:
    """Raise if ``xs`` does not have ``space.shape`` as trailing dimensions."""
    _run_checks(space, xs, allow_leading=True)


def _check_scalar_shape(values: Any, shape: tuple[int, ...]) -> None:
    """Raise if scalar output does not have ``shape``."""
    value_shape = tuple(getattr(values, "shape", ()))
    if value_shape != shape:
        raise ValueError(f"Expected scalar batch output with shape {shape}, got {value_shape}.")


def _leading_batch_size(space: Any, xs: Any) -> int:
    """Return the leading batch size for dense-array batches."""
    if isinstance(xs, tuple) and xs:
        return _leading_batch_size(getattr(space, "spaces", (space,))[0], xs[0])
    shape = tuple(getattr(xs, "shape", ()))
    if not shape:
        return 0
    return int(shape[0])


def _warn_vmap_fallback_once(obj: Any, method: str, batch_size: int) -> None:
    """Warn once per class/method for NumPy-style Python-loop batched fallback."""
    if batch_size <= _VMAP_FALLBACK_WARN_BATCH or obj.ops.has_native_vmap:
        return
    key = (type(obj), method)
    if key in _VMAP_FALLBACK_WARNED:
        return
    _VMAP_FALLBACK_WARNED.add(key)
    warnings.warn(
        f"{type(obj).__name__}.{method} falls back to a Python loop on this backend "
        "(no native vmap); this is O(batch). Provide a vectorized batched override, "
        "or use JAX/Torch.",
        RuntimeWarning,
        stacklevel=3,
    )


def _batched_inner(space: Any, xs: Any, ys: Any) -> Any:
    """Return ``space.inner(xs[i], ys[i])`` for a leading-axis batch."""
    xs_flat = space.flatten_batch(xs)
    ys_dual = ys if space.is_euclidean else space.riesz(ys)
    ys_flat = space.flatten_batch(ys_dual)
    return space.ops.sum(space.ops.conj(xs_flat) * ys_flat, axis=1)
