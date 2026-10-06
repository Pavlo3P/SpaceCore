from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

from ..contextual import Context
from ..functional import QuadraticForm
from ..linop import LinOp
from ..space import Space
from ._utils import SpaceCoreOps
from ._utils import check_maxiter, core_normalize, default_initial_vector, is_converged
from ._utils import require_linop, require_square, resolve_apply, result_repr


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
                k: getattr(self, k)
                for k in ("converged", "num_iters", "eigenvalue", "residual_norm", "eigenvector")
            },
        )


class _SelfAdjointAction(NamedTuple):
    """Store the callable action used by power iteration."""

    apply: Callable[[Any], Any]
    domain: Space
    ctx: Context
    rayleigh: Callable[[Any, Any], Any]
    residual_norm: Callable[[Any, Any, Any], Any]

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
    # Wrap a square LinOp as a self-adjoint action with generic diagnostics.
    A = require_linop(A)
    require_square(A, "power_iteration")
    if A.is_hermitian() is False:
        raise ValueError("power_iteration requires A to be Hermitian/self-adjoint.")
    apply = resolve_apply(A)
    sops = SpaceCoreOps(A.domain)
    return _SelfAdjointAction(
        apply,
        A.domain,
        A.ctx,
        lambda x, y: A.ops.real(sops.inner(x, y)),
        lambda x, y, eigenvalue: sops.norm(sops.axpy(-eigenvalue, x, y)),
    )


def _action_from_quadratic_form(q: QuadraticForm) -> _SelfAdjointAction:
    """Normalize a quadratic form into its Hessian action."""
    sops = SpaceCoreOps(q.domain)
    # Use the form's own Rayleigh quotient and residual when it provides them.
    hess_quad = getattr(q, "hess_quad", None)
    if callable(hess_quad):

        def rayleigh(x, y):
            return q.ops.real(cast(Any, hess_quad)(x, Hx=y))
    else:

        def rayleigh(x, y):
            return q.ops.real(sops.inner(x, y))

    hess_residual_norm = getattr(q, "hess_residual_norm", None)
    if callable(hess_residual_norm):
        residual_norm = hess_residual_norm
    else:

        def residual_norm(x, y, eigenvalue):
            return sops.norm(sops.axpy(-eigenvalue, x, y))

    return _SelfAdjointAction(q.hess_apply, q.domain, q.ctx, rayleigh, residual_norm)


def power_iteration(
    A: LinOp | QuadraticForm,
    *,
    x0: Any | None = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
) -> PowerIterationResult:
    r"""
    Estimate the dominant eigenpair (largest in absolute value) of a self-adjoint action.

    Parameters
    ----------
    A : LinOp or QuadraticForm
        Square operator, or quadratic form with ``hess_apply``. A form may also supply
        ``hess_quad(x, Hx=None)`` and ``hess_residual_norm(x, Hx, eigenvalue)`` for the
        diagnostics. For a spectral norm of a rectangular ``B``, pass ``B.H @ B``.
    x0 : array-like or None, optional
        Initial vector; normalized ones by default.
    tol : float, optional
        Converged when ``norm(A @ x - eigenvalue * x) < tol``.
    maxiter : int or None, optional
        Iteration cap; ``prod(A.domain.shape)`` by default.

    Returns
    -------
    PowerIterationResult
        ``(eigenvalue, eigenvector, converged, num_iters, residual_norm)``.

    Raises
    ------
    TypeError
        If ``A`` is neither a :class:`LinOp` nor a :class:`QuadraticForm`.
    ValueError
        If a linear operator is not square or known non-Hermitian, or an iteration
        parameter is invalid.

    See Also
    --------
    lanczos_smallest : The smallest eigenpair.

    Notes
    -----
    Convergence needs a separated dominant modulus and nonzero initial overlap with
    its eigenspace. Trefethen & Bau, *Numerical Linear Algebra*, Thm 27.1; expansion
    in a metric-orthonormal eigenbasis gives the same eigenvalue-ratio argument.
    With both ``lambda`` and ``-lambda`` present at the dominant modulus the iterate
    can oscillate. JIT-compatible on JAX with ``maxiter`` static.

    Examples
    --------
    Estimate the largest eigenvalue of a diagonal operator.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((3,), ctx)
    >>> A = sc.DiagonalLinOp(ctx.asarray([1.0, 3.0, 2.0]), X, ctx)
    >>> result = sc.power_iteration(A, maxiter=20, tol=1e-10)
    >>> np.allclose(result.eigenvalue, 3.0)
    True
    """
    # Turn either input into one self-adjoint action.
    if isinstance(A, QuadraticForm):
        action = _action_from_quadratic_form(A)
    elif isinstance(A, LinOp):
        action = _action_from_linop(A)
    else:
        raise TypeError(f"A must be a LinOp or QuadraticForm, got {type(A).__name__}.")

    maxiter = check_maxiter(maxiter, cast(Any, action))

    # Start from normalized ones unless x0 is given.
    x = default_initial_vector(cast(Any, action)) if x0 is None else x0
    action.domain.check_member(x)
    return PowerIterationResult(*_power_iteration_core(action, x, tol, maxiter))


def _power_iteration_core(
    action: _SelfAdjointAction,
    x: Any,
    tol: float,
    maxiter: int,
) -> tuple[Any, Any, Any, Any, Any]:
    """Run the backend-loop implementation of power iteration."""
    sops = SpaceCoreOps(action.domain)
    # Normalize x0 and precompute y = A x.
    x, _ = core_normalize(sops, x)
    y = action.apply(x)
    zero = action.ops.asarray(0.0, dtype=action.dtype)
    residual_norm = sops.norm(x) + float("inf")

    # Carry: current eigenvalue estimate, normalized vector x, product y=A x,
    # residual norm, and iteration counter.
    def cond_fun(carry: tuple[Any, Any, Any, Any, int]) -> Any:
        _eigenvalue, _x, _y, res_norm, k = carry
        return (k < maxiter) & (res_norm > tol)

    def body_fun(carry: tuple[Any, Any, Any, Any, int]) -> tuple[Any, Any, Any, Any, int]:
        _eigenvalue, _x, y, _residual_norm, k = carry
        # x <- A x / ||A x||; eigenvalue by the Rayleigh quotient; residual ||A x - lambda x||.
        x_next, _norm_y = core_normalize(sops, y)
        y_next = action.apply(x_next)
        eigenvalue_next = action.rayleigh(x_next, y_next)
        k_next = k + 1

        residual_norm_next = action.residual_norm(x_next, y_next, eigenvalue_next)
        return eigenvalue_next, x_next, y_next, residual_norm_next, k_next

    eigenvalue, eigenvector, _y, residual_norm, num_iters = action.ops.while_loop(
        cond_fun,
        body_fun,
        (zero, x, y, residual_norm, 0),
    )
    return (
        eigenvalue,
        eigenvector,
        is_converged(residual_norm, tol),
        num_iters,
        residual_norm,
    )
