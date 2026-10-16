from __future__ import annotations

import warnings
from typing import Any

from .space.checks import _run_checks
from .space._capabilities import _CAP_BATCH, _CAP_INNER, require

# Shared batched-evaluation helpers used by both LinOp and Functional. They live
# here (rather than in functional/_base) so the linop and functional batched
# methods share one implementation and one warn-once registry.
_VMAP_FALLBACK_WARNED: set[tuple[type, str]] = set()
_VMAP_FALLBACK_WARN_BATCH = 32


def _check_batched(space: Any, xs: Any) -> None:
    """Raise if ``xs`` does not have ``space.shape`` as trailing dimensions."""
    _run_checks(space, xs, allow_leading=True)


def _check_scalar_shape(values: Any, shape: tuple[int, ...]) -> None:
    """Raise if scalar output does not have ``shape``.

    ``shape`` is ``()`` for a single evaluation and the leading batch shape for a
    batched one; the message distinguishes the two, since this now guards every
    ``Functional.value``/``vvalue`` rather than only the batched paths.
    """
    value_shape = tuple(getattr(values, "shape", ()))
    if value_shape != shape:
        kind = "scalar output" if shape == () else "scalar batch output"
        raise ValueError(f"Expected {kind} with shape {shape}, got {value_shape}.")


def _leading_batch_size(space: Any, xs: Any) -> int:
    """Return the leading batch size for dense-array batches."""
    # Tuple batches take the size of their first component.
    if isinstance(xs, tuple) and xs:
        return _leading_batch_size(getattr(space, "spaces", (space,))[0], xs[0])
    shape = tuple(getattr(xs, "shape", ()))
    return int(shape[0]) if shape else 0


def _warn_vmap_fallback_once(obj: Any, method: str, batch_size: int) -> None:
    """Warn once per class/method for NumPy-style Python-loop batched fallback."""
    # Warn once per (class, method) when a large batch loops in Python.
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


def _batched_real_inner(space: Any, xs: Any, ys: Any) -> Any:
    """Return ``space.real_inner(xs[i], ys[i])`` for a leading-axis batch."""
    return space.ops.real(_batched_inner(space, xs, ys))


def _batched_real_pairing(space: Any, c: Any, xs: Any) -> Any:
    """Return ``space.real_inner(c, xs[i])`` for one element ``c`` and a leading-axis batch."""
    # Re <c, x_i>_X = pair(R c, x_i).
    return space._vpair_core(c if space.is_euclidean else space._riesz_core(c), xs)


def _batched_inner(space: Any, xs: Any, ys: Any) -> Any:
    """Return ``space.inner(xs[i], ys[i])`` for a leading-axis batch."""
    require(space, _CAP_BATCH, "_batched_inner")
    require(space, _CAP_INNER, "_batched_inner")
    # Row-wise <x_i, y_i> = conj(x_i) . R y_i on flat coordinates.
    xs_flat = space._flatten_batch_core(xs)
    ys_dual = ys if space.is_euclidean else space._riesz_core(ys)
    ys_flat = space._flatten_batch_core(ys_dual)
    return space.ops.sum(space.ops.conj(xs_flat) * ys_flat, axis=1)
