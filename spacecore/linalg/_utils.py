from __future__ import annotations

from math import prod
from typing import Any

from ..linop import LinOp

DEFAULT_CONVERGENCE_CHECK_INTERVAL = 64


def require_linop(A: Any) -> LinOp:
    """Return ``A`` as a ``LinOp`` or raise a clear type error."""
    if not isinstance(A, LinOp):
        raise TypeError(f"A must be a LinOp, got {type(A).__name__}.")
    return A


def require_square(A: LinOp, name: str) -> None:
    """Raise if ``A`` is not a square operator."""
    if A.domain != A.codomain:
        raise ValueError(f"{name} requires a square LinOp; got {A.domain!r} -> {A.codomain!r}.")


def default_maxiter(A: LinOp) -> int:
    """Return the default Krylov iteration count for ``A``."""
    return max(1, prod(A.domain.shape))


def check_maxiter(maxiter: int | None, A: LinOp) -> int:
    """Validate an optional iteration count."""
    if maxiter is None:
        return default_maxiter(A)
    maxiter = int(maxiter)
    if maxiter < 0:
        raise ValueError("maxiter must be nonnegative.")
    return maxiter


def check_interval(interval: int) -> int:
    """Validate a convergence-check interval."""
    interval = int(interval)
    if interval < 1:
        raise ValueError("check_every must be positive.")
    return interval


def should_check_iteration(k: Any, maxiter: int, interval: int) -> Any:
    """Return whether iteration ``k`` should refresh convergence diagnostics."""
    return (k >= maxiter) | ((k % interval) == 0)


def threshold(norm_b: Any, tol: float, atol: float) -> Any:
    """Return the absolute-plus-relative convergence threshold."""
    return max(float(atol), 0.0) + max(float(tol), 0.0) * norm_b


def real_inner(space: Any, x: Any, y: Any) -> Any:
    """Return the real part of ``space.inner(x, y)``."""
    return space.ops.real(space.inner(x, y))


def is_converged(residual_norm: Any, threshold_value: Any) -> Any:
    """Return backend-compatible convergence predicate."""
    return residual_norm <= threshold_value


def safe_inverse(ops: Any, value: Any) -> Any:
    """Return ``1 / value`` where positive and zero otherwise."""
    positive = value > 0
    safe_value = ops.where(positive, value, ops.ones_like(value))
    return ops.where(positive, 1.0 / safe_value, ops.zeros_like(value))


def normalize(space: Any, x: Any) -> tuple[Any, Any]:
    """Normalize a space member and return ``(unit, norm)``."""
    norm = space.norm(x)
    return space.scale(safe_inverse(space.ops, norm), x), norm


def default_initial_vector(A: LinOp) -> Any:
    """Return a deterministic nonzero initial vector for ``A.domain``."""
    size = prod(A.domain.shape)
    flat = A.ops.ones((size,), dtype=A.dtype) / A.ops.sqrt(A.ops.asarray(float(size)))
    return A.domain.unflatten(flat)
