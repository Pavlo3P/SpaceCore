from __future__ import annotations

from typing import Any

from ..linop import LinOp
from ._utils import DEFAULT_CONVERGENCE_CHECK_INTERVAL, check_interval, check_maxiter
from ._utils import LinalgResult
from ._utils import require_hermitian_square
from ._utils import is_converged
from ._utils import (
    require_strict_cg_preconditions,
    safe_inverse_nonneg,
    should_check_iteration,
    threshold,
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
    preconditioner: LinOp | None = None,
) -> LinalgResult:
    r"""
    Solve :math:`A x = b` by conjugate gradients.

    ``A`` must be square (``A.domain == A.codomain``), Hermitian and positive-definite
    in the domain's inner product. Only ``A.apply`` and that inner product are used.

    Parameters
    ----------
    A : LinOp
        Hermitian positive-definite operator. One that is provably not Hermitian is
        rejected; definiteness is not checked.
    b : array-like
        Right-hand side in ``A.codomain``.
    x0 : array-like or None, optional
        Initial guess; zero by default.
    tol : float, optional
        Converged when the residual norm is below ``atol + tol * norm(b)``.
    atol : float, optional
        Absolute residual tolerance.
    maxiter : int or None, optional
        Iteration cap; ``prod(A.domain.shape)`` by default.
    check_every : int, optional
        Test convergence every this many iterations, and on the last one.
    preconditioner : LinOp or None, optional
        Operator ``K`` on ``A.domain`` approximating the inverse of ``A``, self-adjoint
        and positive definite in ``A.domain``; each residual ``r`` is preconditioned to
        ``K r``. The residual is still measured in ``A.domain``'s norm.

    Returns
    -------
    LinalgResult
        ``(x, converged, num_iters, residual_norm)``.

    Raises
    ------
    TypeError
        If ``A`` is not a :class:`LinOp`.
    ValueError
        If ``A`` is not square or an iteration parameter is invalid.

    See Also
    --------
    lsqr : Least squares for rectangular operators.

    Notes
    -----
    JIT-compatible on JAX with ``maxiter`` and ``check_every`` static. Stops early when
    no useful update remains (residual at machine precision, or non-positive
    curvature). Trefethen & Bau, *Numerical Linear Algebra*, Algorithm 38.1.
    In an orthonormal basis the metric recurrence is ordinary CG; complex spaces
    use conjugate inner products, and a nonzero ``x0`` starts from ``b - A x0``.

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
    # Validate once, up front: square, not provably non-Hermitian, valid b and parameters.
    A = require_hermitian_square(
        A,
        "cg",
        "cg requires A to be Hermitian/self-adjoint with respect to "
        "A.domain.inner, but A was determined to be provably "
        "non-self-adjoint in this geometry. For example, a symmetric "
        "matrix wrapped as a DenseLinOp on a weighted (non-Euclidean) "
        "inner-product space is not self-adjoint under the weighted inner "
        "product. Build a valid operator instead, e.g. a normal operator "
        "A.H @ A + lam * Identity (which is self-adjoint in any geometry), "
        "or use a Euclidean space where the symmetric matrix is "
        "self-adjoint."
    )
    require_strict_cg_preconditions(A)
    A.codomain.check_member(b)
    maxiter = check_maxiter(maxiter, A)
    check_every = check_interval(check_every)

    # Resolve check-free cores once; the hot loop then skips per-iteration
    # validation while honoring any custom geometry the spaces define.
    apply = A._apply_core
    dom = A.domain
    cod = A.codomain

    # Preconditioning z = K r; without K, z = r and this is plain CG.
    if preconditioner is None:
        def precond(r: Any) -> Any:
            return r
    else:
        if preconditioner.domain != A.domain or preconditioner.codomain != A.domain:
            raise ValueError("cg: the preconditioner must map A.domain to itself.")
        precond = preconditioner._apply_core

    # Initial residual r = b - A x; the first search direction is z = K r.
    x = A.domain.zeros() if x0 is None else x0
    A.domain.check_member(x)
    r = cod._add_core(b, cod._scale_core(-1.0, apply(x)))
    p = precond(r)
    rs = dom._real_inner_core(r, p)
    residual_norm = dom._norm_core(r)
    # Stop when ||r|| <= atol + tol ||b||; eps guards the divisions below.
    threshold_value = threshold(cod._norm_core(b), tol, atol)
    # Scalar diagnostics are real even for a complex operator (ADR-015).
    real_dtype = A.ops.real_dtype(A.dtype)
    eps = A.ops.asarray(A.ops.eps(A.dtype), dtype=real_dtype)
    eps2 = eps * eps

    def cond_fun(carry: tuple[Any, Any, Any, Any, Any, int, Any]) -> Any:
        _x, _r, _p, _rs, res_norm, k, active = carry
        # Continue while under the cap, not converged, and still making progress.
        return (k < maxiter) & (res_norm > threshold_value) & active

    def body_fun(
        carry: tuple[Any, Any, Any, Any, Any, int, Any],
    ) -> tuple[Any, Any, Any, Any, Any, int, Any]:
        x, r, p, rs, _residual_norm, k, _active = carry
        # Step length alpha = <r, z> / <p, A p>; freeze once curvature or residual vanishes.
        Ap = apply(p)
        pAp = dom._real_inner_core(p, Ap)
        active = (rs > eps2) & (pAp > eps * rs)
        alpha = A.ops.where(active, rs * safe_inverse_nonneg(A.ops, pAp), A.ops.zeros_like(rs))
        # Move x along p and update the residual without another apply.
        x_next = dom._axpy_core(alpha, p, x)
        r_next = cod._axpy_core(-alpha, Ap, r)
        z_next = precond(r_next)
        rs_next = dom._real_inner_core(r_next, z_next)
        # New direction p = z + beta p, with beta = <r_new, z_new> / <r, z>.
        beta = A.ops.where(
            active, rs_next * safe_inverse_nonneg(A.ops, rs), A.ops.zeros_like(rs_next)
        )
        p_next = dom._axpy_core(beta, p, z_next)
        k_next = k + 1
        # Refresh the reported residual norm only on check iterations.
        should_refresh_residual = should_check_iteration(k_next, maxiter, check_every) | (~active)
        residual_norm_next = A.ops.cond(
            should_refresh_residual,
            (lambda _: A.ops.sqrt(rs_next)) if preconditioner is None else (lambda _: dom._norm_core(r_next)),
            lambda _: _residual_norm,
            A.ops.asarray(0.0, dtype=real_dtype),
        )
        return x_next, r_next, p_next, rs_next, residual_norm_next, k_next, active

    # One backend while-loop, so the whole solve can be jitted.
    x, _r, _p, _rs, residual_norm, num_iters, _active = A.ops.while_loop(
        cond_fun,
        body_fun,
        (x, r, p, rs, residual_norm, 0, A.ops.asarray(True)),
    )
    return LinalgResult(x, is_converged(residual_norm, threshold_value), num_iters, residual_norm)
