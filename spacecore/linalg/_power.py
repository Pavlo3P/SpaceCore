from __future__ import annotations

from typing import Any, NamedTuple

from ..linop import LinOp
from ._utils import DEFAULT_CONVERGENCE_CHECK_INTERVAL, check_interval, check_maxiter
from ._utils import default_initial_vector, is_converged, normalize, require_linop
from ._utils import require_square, should_check_iteration


class PowerIterationResult(NamedTuple):
    """Result returned by :func:`power_iteration`."""

    eigenvalue: Any
    eigenvector: Any
    converged: Any
    num_iters: Any
    residual_norm: Any


def power_iteration(
    A: LinOp,
    *,
    x0: Any | None = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
    check_every: int = DEFAULT_CONVERGENCE_CHECK_INTERVAL,
) -> PowerIterationResult:
    """
    Estimate the dominant eigenpair of a square ``LinOp`` by power iteration.

    The method uses only ``A.apply`` and domain-space operations. It returns the
    Rayleigh quotient for the current normalized iterate, the eigenvector
    estimate, and the residual norm ``||A x - lambda x||``. The residual-based
    stopping criterion is refreshed only every ``check_every`` iterations, and
    always on the final iteration. For spectral-norm estimates of a rectangular
    operator, call this on ``A.H @ A``.
    """
    A = require_linop(A)
    require_square(A, "power_iteration")
    maxiter = check_maxiter(maxiter, A)
    check_every = check_interval(check_every)

    x = default_initial_vector(A) if x0 is None else x0
    A.domain.check_member(x)
    x, _ = normalize(A.domain, x)
    zero = A.ops.asarray(0.0, dtype=A.dtype)
    residual_norm = A.domain.norm(x) + float("inf")

    def cond_fun(carry: tuple[Any, Any, Any, int]) -> Any:
        _eigenvalue, _x, res_norm, k = carry
        return (k < maxiter) & (res_norm > tol)

    def body_fun(carry: tuple[Any, Any, Any, int]) -> tuple[Any, Any, Any, int]:
        _eigenvalue, x, _residual_norm, k = carry
        y = A.apply(x)
        x_next, _norm_y = normalize(A.domain, y)
        y_next = A.apply(x_next)
        eigenvalue_next = A.domain.inner(x_next, y_next)
        k_next = k + 1

        def refresh_residual(_: Any) -> Any:
            residual = A.codomain.axpy(-eigenvalue_next, x_next, y_next)
            return A.codomain.norm(residual)

        residual_norm_next = A.ops.cond(
            should_check_iteration(k_next, maxiter, check_every),
            refresh_residual,
            lambda _: _residual_norm,
            A.ops.asarray(0.0, dtype=A.dtype),
        )
        return eigenvalue_next, x_next, residual_norm_next, k_next

    eigenvalue, eigenvector, residual_norm, num_iters = A.ops.while_loop(
        cond_fun,
        body_fun,
        (zero, x, residual_norm, 0),
    )
    return PowerIterationResult(
        eigenvalue,
        eigenvector,
        is_converged(residual_norm, tol),
        num_iters,
        residual_norm,
    )
