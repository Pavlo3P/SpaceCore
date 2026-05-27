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
    """
    Store the result returned by :func:`power_iteration`.

    Parameters
    ----------
    eigenvalue : scalar
        Rayleigh-quotient estimate of the dominant eigenvalue.
    eigenvector : array-like
        Normalized eigenvector estimate in the operator domain.
    converged : bool-like
        Whether the residual norm satisfied ``tol``.
    num_iters : int-like
        Number of power iterations executed.
    residual_norm : scalar
        Norm of ``A x - eigenvalue * x``.
    """

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
    """Store the callable action used by power iteration."""

    apply: Callable[[Any], Any]
    domain: Space
    ctx: Context

    @property
    def ops(self) -> Any:
        """Backend operations for this action."""
        return self.ctx.ops

    @property
    def dtype(self) -> Any:
        """Default dtype for this action."""
        return self.ctx.dtype


def _action_from_linop(A: LinOp) -> _SelfAdjointAction:
    """Normalize a square linear operator into a self-adjoint action."""
    A = require_linop(A)
    require_square(A, "power_iteration")
    return _SelfAdjointAction(A.apply, A.domain, A.ctx)


def _action_from_quadratic_form(q: QuadraticForm) -> _SelfAdjointAction:
    """Normalize a quadratic form into its Hessian action."""
    return _SelfAdjointAction(q.hess_apply, q.domain, q.ctx)


def power_iteration(
    A: LinOp | QuadraticForm,
    *,
    x0: Any | None = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
    check_every: int = DEFAULT_CONVERGENCE_CHECK_INTERVAL,
) -> PowerIterationResult:
    r"""
    Estimate the dominant eigenpair of a self-adjoint action.

    Accept a square :class:`LinOp` or a :class:`QuadraticForm` exposing
    ``hess_apply``. Public dispatch converts either input into a fixed
    self-adjoint action before entering the numerical loop. "Dominant" means
    largest eigenvalue in absolute value, not necessarily the largest positive
    eigenvalue.

    Parameters
    ----------
    A : LinOp or QuadraticForm
        Square operator or quadratic form whose dominant eigenpair, largest in
        absolute value, is sought. Linear-operator inputs must satisfy
        ``A.domain == A.codomain``; this includes the underlying space type and
        inner-product geometry.
        For spectral-norm estimates of a rectangular operator, pass
        ``A.H @ A``.
    x0 : array-like or None, optional
        Initial vector in the action domain. Default is a normalized all-ones
        vector in the domain geometry.
    tol : float, optional
        Residual-norm tolerance. ``result.converged`` is ``True`` when
        ``norm(A @ x - lambda * x) < tol``. Default is 1e-6.
    maxiter : int or None, optional
        Maximum number of iterations. Default is ``prod(A.domain.shape)``.
    check_every : int, optional
        Refresh residual diagnostics every this many iterations and always on
        the final iteration. Default is
        ``DEFAULT_CONVERGENCE_CHECK_INTERVAL``.

    Returns
    -------
    PowerIterationResult
        Named tuple with fields:

        - ``eigenvalue``: Rayleigh-quotient eigenvalue estimate
        - ``eigenvector``: normalized eigenvector estimate
        - ``converged``: whether ``residual_norm < tol``
        - ``num_iters``: number of iterations executed
        - ``residual_norm``: norm of ``A x - eigenvalue * x``

    Raises
    ------
    TypeError
        If ``A`` is neither a :class:`LinOp` nor a :class:`QuadraticForm`.
    ValueError
        If a linear-operator input is not square or if iteration parameters are
        invalid.

    See Also
    --------
    lanczos_smallest : Approximate the smallest eigenpair of a Hermitian
        operator.
    cg : Solve Hermitian positive-definite systems.

    Notes
    -----
    The residual-based stopping criterion uses
    :math:`\|A x - \lambda x\|` and is refreshed only every ``check_every``
    iterations, and always on the final iteration. This function is
    JIT-compatible on the JAX backend when ``maxiter`` and ``check_every`` are
    static arguments.

    For operators with eigenvalues of mixed sign, the dominant eigenvalue is
    the one with largest absolute value, which may be negative. Convergence
    requires that this eigenvalue be separated from the rest in absolute value.
    If the dominant modulus is degenerate, for example both ``lambda`` and
    ``-lambda`` have maximum modulus, the iteration may oscillate between
    subspaces.

    Examples
    --------
    Estimate the largest eigenvalue of a diagonal operator.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.VectorSpace((3,), ctx)
    >>> A = sc.DiagonalLinOp(ctx.asarray([1.0, 3.0, 2.0]), X, ctx)
    >>> result = sc.power_iteration(A, maxiter=20, tol=1e-10)
    >>> np.allclose(result.eigenvalue, 3.0)
    True
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
    """Run the backend-loop implementation of power iteration."""
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
