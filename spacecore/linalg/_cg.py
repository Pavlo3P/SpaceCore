from __future__ import annotations

from typing import Any, NamedTuple

from ..linop import LinOp
from ._utils import DEFAULT_CONVERGENCE_CHECK_INTERVAL, check_interval, check_maxiter
from ._utils import is_converged, real_inner, require_linop, require_square
from ._utils import (
    require_strict_cg_preconditions,
    result_repr,
    safe_inverse_nonneg,
    should_check_iteration,
    threshold,
)


class CGResult(NamedTuple):
    """
    Store the result returned by :func:`cg`.

    Parameters
    ----------
    x : array-like
        Approximate solution in ``A.domain``.
    converged : bool-like
        Whether the final residual norm satisfied the requested tolerance.
    num_iters : int-like
        Number of conjugate-gradient iterations executed.
    residual_norm : scalar
        Norm of the final residual in ``A.codomain``.
    """

    x: Any
    converged: Any
    num_iters: Any
    residual_norm: Any

    def __repr__(self) -> str:
        """Return a compact summary without printing the full solution array."""
        return result_repr(
            "CGResult",
            {
                "converged": self.converged,
                "num_iters": self.num_iters,
                "residual_norm": self.residual_norm,
                "x": self.x,
            },
        )


def cg(
    A: LinOp,
    b: Any,
    *,
    x0: Any | None = None,
    tol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int | None = None,
    check_every: int = DEFAULT_CONVERGENCE_CHECK_INTERVAL,
) -> CGResult:
    r"""
    Solve :math:`A x = b` by conjugate gradients.

    Require ``A`` to be square in the SpaceCore sense
    (``A.domain == A.codomain``), Hermitian, and positive-definite with respect
    to ``A.domain.inner``. The implementation uses only :meth:`LinOp.apply` and
    the domain-space inner product; it never materializes a dense matrix.

    Parameters
    ----------
    A : LinOp
        Linear operator that must be Hermitian positive-definite with respect
        to ``A.domain.inner``. ``A.domain`` must equal ``A.codomain``,
        including the underlying space type and inner-product geometry.
        Hermiticity and positive-definiteness are not validated by ``cg``;
        indefinite or non-Hermitian operators can diverge or produce NaN
        outputs without an explicit error.
    b : array-like
        Right-hand side in ``A.codomain``.
    x0 : array-like or None, optional
        Initial guess in ``A.domain``. Default is the zero vector.
    tol : float, optional
        Relative tolerance on the linear-system residual. ``result.converged``
        is ``True`` when the residual norm is below
        ``atol + tol * norm(b)``. Default is 1e-6.
    atol : float, optional
        Absolute residual tolerance. Default is 0.0.
    maxiter : int or None, optional
        Maximum number of iterations. Default is ``prod(A.domain.shape)``.
    check_every : int, optional
        Refresh convergence diagnostics every this many iterations and always
        on the final iteration. Default is
        ``DEFAULT_CONVERGENCE_CHECK_INTERVAL``.

    Returns
    -------
    CGResult
        Named tuple with fields:

        - ``x``: approximate solution in ``A.domain``
        - ``converged``: whether the requested tolerance was met
        - ``num_iters``: number of iterations executed
        - ``residual_norm``: final residual norm

    Raises
    ------
    TypeError
        If ``A`` is not a :class:`LinOp`.
    ValueError
        If ``A`` is not square or if iteration parameters are invalid.

    See Also
    --------
    lsqr : Solve least-squares systems for rectangular operators.
    lanczos_smallest : Approximate the smallest eigenpair of a Hermitian
        operator.

    Notes
    -----
    The residual norm is compared with
    :math:`\text{atol} + \text{tol} \| b \|` only every ``check_every``
    iterations, and always on the final iteration. This keeps convergence
    checks out of the hot loop while remaining compatible with JAX JIT control
    flow. ``maxiter`` and ``check_every`` should be treated as static JAX
    arguments.

    Iteration also stops when no numerically useful CG update remains: either
    the squared residual is at machine-precision scale or the curvature
    ``inner(p, A p)`` is nonpositive/tiny relative to the residual scale. The
    residual is refreshed before this early exit, so ``converged`` still
    reflects the returned iterate.

    For complex operators, residual norms and step sizes are computed from the
    real part of ``A.domain.inner(x, y)``. SpaceCore's complex inner-product
    convention conjugates the first argument; custom :class:`Space` subclasses
    must follow that convention for CG to converge correctly.

    Inner products and norms use ``A.domain.inner`` and ``A.domain.norm``.
    The method is correct on non-Euclidean geometries when the space supplies
    Riesz maps and ``A`` is Hermitian positive-definite in that geometry.

    References
    ----------
    Hestenes, M. R. and Stiefel, E., "Methods of Conjugate Gradients
    for Solving Linear Systems," J. Res. Natl. Bur. Stand., 49 (1952),
    409-436.

    Examples
    --------
    Solve a small positive-definite system.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((3,), ctx)
    >>> M = ctx.asarray([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    >>> A = sc.DenseLinOp(M, X, X, ctx)
    >>> b = ctx.asarray([1.0, 2.0, 3.0])
    >>> result = sc.cg(A, b, tol=1e-10)
    >>> bool(result.converged)
    True
    >>> np.allclose(A.apply(result.x), b)
    True
    """
    A = require_linop(A)
    require_square(A, "cg")
    require_strict_cg_preconditions(A)
    A.codomain.check_member(b)
    maxiter = check_maxiter(maxiter, A)
    check_every = check_interval(check_every)

    x = A.domain.zeros() if x0 is None else x0
    A.domain.check_member(x)
    r = A.codomain.add(b, A.codomain.scale(-1.0, A.apply(x)))
    p = r
    rs = real_inner(A.domain, r, r)
    residual_norm = A.domain.norm(r)
    threshold_value = threshold(A.codomain.norm(b), tol, atol)
    eps = A.ops.asarray(A.ops.eps(A.dtype), dtype=A.dtype)
    eps2 = eps * eps

    def cond_fun(carry: tuple[Any, Any, Any, Any, Any, int, Any]) -> Any:
        _x, _r, _p, _rs, res_norm, k, active = carry
        return (k < maxiter) & (res_norm > threshold_value) & active

    def body_fun(
        carry: tuple[Any, Any, Any, Any, Any, int, Any],
    ) -> tuple[Any, Any, Any, Any, Any, int, Any]:
        x, r, p, rs, _residual_norm, k, _active = carry
        Ap = A.apply(p)
        pAp = real_inner(A.domain, p, Ap)
        active = (rs > eps2) & (pAp > eps * rs)
        alpha = A.ops.where(active, rs * safe_inverse_nonneg(A.ops, pAp), A.ops.zeros_like(rs))
        x_next = A.domain.axpy(alpha, p, x)
        r_next = A.codomain.axpy(-alpha, Ap, r)
        rs_next = real_inner(A.domain, r_next, r_next)
        beta = A.ops.where(
            active, rs_next * safe_inverse_nonneg(A.ops, rs), A.ops.zeros_like(rs_next)
        )
        p_next = A.domain.axpy(beta, p, r_next)
        k_next = k + 1
        should_refresh_residual = should_check_iteration(k_next, maxiter, check_every) | (~active)
        residual_norm_next = A.ops.cond(
            should_refresh_residual,
            lambda _: A.ops.sqrt(rs_next),
            lambda _: _residual_norm,
            A.ops.asarray(0.0, dtype=A.dtype),
        )
        return x_next, r_next, p_next, rs_next, residual_norm_next, k_next, active

    x, _r, _p, _rs, residual_norm, num_iters, _active = A.ops.while_loop(
        cond_fun,
        body_fun,
        (x, r, p, rs, residual_norm, 0, A.ops.asarray(True)),
    )
    return CGResult(x, is_converged(residual_norm, threshold_value), num_iters, residual_norm)
