from __future__ import annotations

from math import prod
from typing import Any, NamedTuple

from ..linop import LinOp
from .._repr import summarize_value

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


def require_strict_cg_preconditions(A: LinOp) -> None:
    """Run expensive development-time Hermitian and positive-curvature probes."""
    # Strict only: probe positive curvature <x, A x> > 0 on one vector.
    if not A._checks_at_least("strict"):
        return
    if A.is_hermitian() is False:
        raise ValueError("cg strict checks require A to be Hermitian/self-adjoint.")

    x = default_initial_vector(A)
    curvature = A.domain.real_inner(x, A.apply(x))
    try:
        positive = bool(curvature > 0)
    except Exception as exc:
        raise ValueError("cg strict positive-definiteness probe could not be evaluated.") from exc
    if not positive:
        raise ValueError("cg strict checks require positive curvature on the probe vector.")


def default_maxiter(A: LinOp) -> int:
    """Return the default Krylov iteration count for ``A``."""
    from ..space import CoordinateSpace
    from .._errors import CapabilityError

    if not isinstance(A.domain, CoordinateSpace):
        raise CapabilityError("Default iteration count requires CoordinateSpace; supply maxiter explicitly.")
    return max(1, prod(A.domain.shape))


def check_maxiter(maxiter: int | None, A: LinOp) -> int:
    """Validate an optional iteration count."""
    if maxiter is None:
        return default_maxiter(A)
    maxiter = int(maxiter)
    if maxiter < 0:
        raise ValueError("maxiter must be nonnegative.")
    return maxiter


def check_interval(interval: int, name: str = "check_every") -> int:
    """Validate a positive integer parameter, such as a convergence-check interval."""
    interval = int(interval)
    if interval < 1:
        raise ValueError(f"{name} must be positive.")
    return interval


def require_hermitian_square(A: Any, name: str, message: str | None = None) -> Any:
    """Return ``A`` as a square LinOp, raising if it is provably not self-adjoint."""
    A = require_linop(A)
    require_square(A, name)
    if A._checks_at_least("standard") and A.is_hermitian() is False:
        raise ValueError(message or f"{name} requires A to be Hermitian/self-adjoint.")
    return A


def should_check_iteration(k: Any, maxiter: int, interval: int) -> Any:
    """Return whether iteration ``k`` should refresh convergence diagnostics."""
    return (k >= maxiter) | ((k % interval) == 0)


def threshold(norm_b: Any, tol: float, atol: float) -> Any:
    """Return the absolute-plus-relative convergence threshold."""
    return max(float(atol), 0.0) + max(float(tol), 0.0) * norm_b


# ---------------------------------------------------------------------------
# Check-free primitives for solver hot loops.
#
# Iterative solvers validate their operator and right-hand side once, at entry.
# Their iterates (search directions, residuals, Krylov vectors) are then
# generated internally and are valid by construction, so the per-iteration
# membership checks the public ``apply`` / space ops run are pure overhead
# (measured at well over half of CG's NumPy runtime). These helpers route the
# hot loop through the operators' and spaces' check-free ``_*_core`` kernels.
#
# Safety: a core is used only when it is *consistent* with the public method —
# i.e. the public method is not overridden by a subclass below the class that
# defines the core. That guards against a user space that overrides ``inner``
# (custom geometry) while inheriting a built-in ``_inner_core``; there the
# override wins and the checked public method is used. Resolution is done once,
# at solver entry, so the hot loop pays neither the check nor the lookup.
# ---------------------------------------------------------------------------
def is_converged(residual_norm: Any, threshold_value: Any) -> Any:
    """Return backend-compatible convergence predicate."""
    return residual_norm <= threshold_value


def safe_inverse_nonneg(ops: Any, value: Any) -> Any:
    """
    Return ``1 / value`` where ``value > 0`` and zero otherwise.

    This helper is intended for norms and nonnegative residual magnitudes. It
    is not a general scalar inverse: for example, ``-2`` maps to ``0``, not
    ``-0.5``.
    """
    positive = value > 0
    safe_value = ops.where(positive, value, ops.ones_like(value))
    return ops.where(positive, 1.0 / safe_value, ops.zeros_like(value))


def normalize(space: Any, x: Any) -> tuple[Any, Any]:
    """Normalize a space member and return ``(unit, norm)``."""
    norm = space._norm_core(x)
    return space._scale_core(safe_inverse_nonneg(space.ops, norm), x), norm


def default_initial_vector(A: LinOp) -> Any:
    """Return a deterministic unit vector in ``A.domain`` using its geometry."""
    from ..space._capabilities import probe_element
    from .._errors import CapabilityError

    v = probe_element(A.domain, A.ops, A.dtype)
    if v is None:
        raise CapabilityError(
            "Default initialization requires space.ones() or CoordinateSpace; "
            "supply an explicit initial element (x0)."
        )
    return normalize(A.domain, v)[0]


def result_repr(name: str, fields: dict[str, Any]) -> str:
    """Return a compact result-object representation."""
    body = ", ".join(f"{key}={summarize_value(value)}" for key, value in fields.items())
    return f"{name}({body})"


class LinalgResult(NamedTuple):
    """
    Result of every :mod:`spacecore.linalg` solver.

    Fields a solver does not produce are ``None``.

    Parameters
    ----------
    x : array-like
        The computed element: the solution (:func:`cg`, :func:`lsqr`), the eigenvector
        estimate (:func:`power_iteration`, :func:`lanczos_smallest`), or the action
        ``exp(t * A) @ v`` (:func:`expm_multiply`).
    converged : bool-like
        Whether the solver's stopping test was met.
    num_iters : int-like
        Iterations executed; for the Krylov methods :func:`lanczos_smallest` and
        :func:`expm_multiply`, the Krylov dimension reached.
    residual_norm : scalar
        The solver's residual measure: ``norm(b - A x)`` for :func:`cg` and :func:`lsqr`
        (exact or estimated per ``residual_mode``), ``norm(A x - eigenvalue x)`` for
        :func:`power_iteration`, the Ritz residual estimate for :func:`lanczos_smallest`,
        and for :func:`expm_multiply` the projection indicator ``abs(beta[m] phi[m-1])``,
        which omits ``norm(v)`` and does not bound the error.
    eigenvalue : scalar or None
        Eigenvalue estimate (:func:`power_iteration`, :func:`lanczos_smallest`).
    normal_residual_norm : scalar or None
        ``norm(A.H (A x - b))`` (:func:`lsqr`).
    """

    x: Any
    converged: Any
    num_iters: Any
    residual_norm: Any
    eigenvalue: Any = None
    normal_residual_norm: Any = None

    def __repr__(self) -> str:
        """Summarize the fields this solver produced, without printing full arrays."""
        fields = {k: v for k, v in self._asdict().items() if v is not None}
        return result_repr(type(self).__name__, fields)
