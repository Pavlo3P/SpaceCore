from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from ..backend import Context
from ..functional import QuadraticForm
from ..linop import LinOp
from ..space import Space
from ._utils import DEFAULT_CONVERGENCE_CHECK_INTERVAL, check_interval, check_maxiter
from ._utils import default_initial_vector, is_converged, normalize, require_linop
from ._utils import require_square, result_repr, should_check_iteration


class PowerIterationResult(NamedTuple):
    """Result returned by :func:`power_iteration`."""

    eigenvalue: Any
    eigenvector: Any
    converged: Any
    num_iters: Any
    residual_norm: Any

    def __repr__(self) -> str:
        """Return a compact summary without printing the full eigenvector."""
        return result_repr(
            "PowerIterationResult",
            {
                "converged": self.converged,
                "num_iters": self.num_iters,
                "eigenvalue": self.eigenvalue,
                "residual_norm": self.residual_norm,
                "eigenvector": self.eigenvector,
            },
        )


class _SelfAdjointAction(NamedTuple):
    apply: Callable[[Any], Any]
    domain: Space
    ctx: Context

    @property
    def ops(self) -> Any:
        return self.ctx.ops

    @property
    def dtype(self) -> Any:
        return self.ctx.dtype


def _action_from_linop(A: LinOp) -> _SelfAdjointAction:
    A = require_linop(A)
    require_square(A, "power_iteration")
    return _SelfAdjointAction(A.apply, A.domain, A.ctx)


def _action_from_quadratic_form(q: QuadraticForm) -> _SelfAdjointAction:
    return _SelfAdjointAction(q.hess_apply, q.domain, q.ctx)


def power_iteration(
    A: LinOp | QuadraticForm,
    *,
    x0: Any | None = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
    check_every: int = DEFAULT_CONVERGENCE_CHECK_INTERVAL,
) -> PowerIterationResult:
    """
    Estimate the dominant eigenpair of a square ``LinOp`` or Hessian action.

    ``A`` may be a square ``LinOp`` or a ``QuadraticForm`` that exposes
    ``hess_apply``. Public dispatch converts either input into a fixed
    self-adjoint action before entering the numerical loop. The method returns
    the Rayleigh quotient for the current normalized iterate, the eigenvector
    estimate, and the residual norm ``||A x - lambda x||``. The residual-based
    stopping criterion is refreshed only every ``check_every`` iterations, and
    always on the final iteration. For spectral-norm estimates of a rectangular
    operator, call this on ``A.H @ A``.
    """
    if isinstance(A, QuadraticForm):
        action = _action_from_quadratic_form(A)
    elif isinstance(A, LinOp):
        action = _action_from_linop(A)
    else:
        raise TypeError(f"A must be a LinOp or QuadraticForm, got {type(A).__name__}.")

    maxiter = check_maxiter(maxiter, action)
    check_every = check_interval(check_every)

    x = default_initial_vector(action) if x0 is None else x0
    action.domain.check_member(x)
    return PowerIterationResult(*_power_iteration_core(action, x, tol, maxiter, check_every))


def _power_iteration_core(
    action: _SelfAdjointAction,
    x: Any,
    tol: float,
    maxiter: int,
    check_every: int,
) -> tuple[Any, Any, Any, Any, Any]:
    x, _ = normalize(action.domain, x)
    zero = action.ops.asarray(0.0, dtype=action.dtype)
    residual_norm = action.domain.norm(x) + float("inf")

    def cond_fun(carry: tuple[Any, Any, Any, int]) -> Any:
        _eigenvalue, _x, res_norm, k = carry
        return (k < maxiter) & (res_norm > tol)

    def body_fun(carry: tuple[Any, Any, Any, int]) -> tuple[Any, Any, Any, int]:
        _eigenvalue, x, _residual_norm, k = carry
        y = action.apply(x)
        x_next, _norm_y = normalize(action.domain, y)
        y_next = action.apply(x_next)
        eigenvalue_next = action.domain.inner(x_next, y_next)
        k_next = k + 1

        def refresh_residual(_: Any) -> Any:
            residual = action.domain.axpy(-eigenvalue_next, x_next, y_next)
            return action.domain.norm(residual)

        residual_norm_next = action.ops.cond(
            should_check_iteration(k_next, maxiter, check_every),
            refresh_residual,
            lambda _: _residual_norm,
            action.ops.asarray(0.0, dtype=action.dtype),
        )
        return eigenvalue_next, x_next, residual_norm_next, k_next

    eigenvalue, eigenvector, residual_norm, num_iters = action.ops.while_loop(
        cond_fun,
        body_fun,
        (zero, x, residual_norm, 0),
    )
    return (
        eigenvalue,
        eigenvector,
        is_converged(residual_norm, tol),
        num_iters,
        residual_norm,
    )
