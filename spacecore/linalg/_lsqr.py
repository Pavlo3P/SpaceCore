from __future__ import annotations

from typing import Any, NamedTuple

from ..linop import LinOp
from ._utils import DEFAULT_CONVERGENCE_CHECK_INTERVAL, check_interval, check_maxiter
from ._utils import is_converged, require_linop, safe_inverse_nonneg, should_check_iteration
from ._utils import result_repr, threshold


class LSQRResult(NamedTuple):
    """
    Store the result returned by :func:`lsqr`.

    Parameters
    ----------
    x : array-like
        Approximate least-squares solution in ``A.domain``.
    converged : bool-like
        Whether the normal-equation residual satisfied the requested tolerance.
    num_iters : int-like
        Number of LSQR iterations executed.
    residual_norm : scalar
        Norm of ``A x - b`` in ``A.codomain``.
    normal_residual_norm : scalar
        Norm of ``A.H @ (A x - b)`` in ``A.domain``.
    """

    x: Any
    converged: Any
    num_iters: Any
    residual_norm: Any
    normal_residual_norm: Any

    def __repr__(self) -> str:
        """Return a compact summary without printing the full solution array."""
        return result_repr(
            "LSQRResult",
            {
                "converged": self.converged,
                "num_iters": self.num_iters,
                "residual_norm": self.residual_norm,
                "normal_residual_norm": self.normal_residual_norm,
                "x": self.x,
            },
        )


def lsqr(
    A: LinOp,
    b: Any,
    *,
    x0: Any | None = None,
    tol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int | None = None,
    check_every: int = DEFAULT_CONVERGENCE_CHECK_INTERVAL,
) -> LSQRResult:
    r"""
    Solve :math:`\min_x \|A x - b\|` by LSQR.

    Allow ``A`` to be rectangular or square. The method uses
    :meth:`LinOp.apply` for forward products and ``A.H.apply`` for adjoint
    products, so the normal equations are represented implicitly and no dense
    matrix is formed.

    Parameters
    ----------
    A : LinOp
        Linear operator defining the least-squares problem.
    b : array-like
        Right-hand side in ``A.codomain``.
    x0 : array-like or None, optional
        Initial guess in ``A.domain``. Default is the zero vector.
    tol : float, optional
        Relative tolerance for the normal-equation residual. Default is 1e-6.
    atol : float, optional
        Absolute tolerance for the normal-equation residual. Default is 0.0.
    maxiter : int or None, optional
        Maximum number of iterations. Default is ``prod(A.domain.shape)``.
    check_every : int, optional
        Refresh residual diagnostics every this many iterations and always on
        the final iteration. Default is
        ``DEFAULT_CONVERGENCE_CHECK_INTERVAL``.

    Returns
    -------
    LSQRResult
        Named tuple with fields:

        - ``x``: approximate least-squares solution in ``A.domain``
        - ``converged``: whether the requested tolerance was met
        - ``num_iters``: number of iterations executed
        - ``residual_norm``: final residual norm
        - ``normal_residual_norm``: final normal-equation residual norm

    Raises
    ------
    TypeError
        If ``A`` is not a :class:`LinOp`.
    ValueError
        If iteration parameters are invalid.

    See Also
    --------
    cg : Solve square Hermitian positive-definite systems.
    power_iteration : Estimate a dominant eigenpair.

    Notes
    -----
    Convergence is tested using
    :math:`\|A^*(A x - b)\| < \text{atol} + \text{tol}\|b\|`.
    The normal-equation residual is refreshed only every ``check_every``
    iterations, and always on the final iteration. This function is
    JIT-compatible on the JAX backend when ``maxiter`` and ``check_every`` are
    static arguments.

    Works on real and complex operators. For complex operators, ``A.H`` uses
    the conjugate adjoint.

    References
    ----------
    Paige, C. C. and Saunders, M. A., "LSQR: An Algorithm for Sparse
    Linear Equations and Sparse Least Squares," ACM Trans. Math. Soft.,
    8 (1982), 43-71.

    Examples
    --------
    Solve a small overdetermined least-squares problem.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.VectorSpace((2,), ctx)
    >>> Y = sc.VectorSpace((3,), ctx)
    >>> M = ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> A = sc.DenseLinOp(M, X, Y, ctx)
    >>> b = ctx.asarray([1.0, 2.0, 3.0])
    >>> result = sc.lsqr(A, b, tol=1e-10)
    >>> np.allclose(result.x, [1.0, 2.0])
    True
    """
    A = require_linop(A)
    A.codomain.check_member(b)
    maxiter = check_maxiter(maxiter, A)
    check_every = check_interval(check_every)

    x = A.domain.zeros() if x0 is None else x0
    A.domain.check_member(x)
    residual = A.codomain.add(b, A.codomain.scale(-1.0, A.apply(x)))
    beta = A.codomain.norm(residual)
    normal_residual_norm = A.domain.norm(A.H.apply(residual))
    u = residual
    u = A.codomain.scale(safe_inverse_nonneg(A.ops, beta), u)
    v = A.H.apply(u)
    alpha = A.domain.norm(v)
    v = A.domain.scale(safe_inverse_nonneg(A.ops, alpha), v)
    w = v
    phi_bar = beta
    rho_bar = alpha
    residual_norm = beta
    threshold_value = threshold(A.codomain.norm(b), tol, atol)

    def cond_fun(carry: tuple[Any, ...]) -> Any:
        _x, _u, _v, _w, _alpha, _beta, _rho_bar, _phi_bar, _res_norm, norm_res, k = carry
        return (k < maxiter) & (norm_res > threshold_value)

    def body_fun(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        x, u, v, w, alpha, _beta, rho_bar, phi_bar, _residual_norm, _normal_residual, k = carry
        u_next = A.codomain.axpy(-alpha, u, A.apply(v))
        beta_next = A.codomain.norm(u_next)
        u_next = A.codomain.scale(safe_inverse_nonneg(A.ops, beta_next), u_next)

        v_next = A.domain.axpy(-beta_next, v, A.H.apply(u_next))
        alpha_next = A.domain.norm(v_next)
        v_next = A.domain.scale(safe_inverse_nonneg(A.ops, alpha_next), v_next)

        rho = A.ops.sqrt(rho_bar * rho_bar + beta_next * beta_next)
        inv_rho = safe_inverse_nonneg(A.ops, rho)
        c = rho_bar * inv_rho
        s = beta_next * inv_rho
        theta = s * alpha_next
        rho_bar_next = -c * alpha_next
        phi = c * phi_bar
        phi_bar_next = s * phi_bar

        x_next = A.domain.axpy(phi * inv_rho, w, x)
        w_next = A.domain.axpy(-(theta * inv_rho), w, v_next)
        k_next = k + 1

        def refresh_residuals(payload: tuple[Any, Any, Any]) -> tuple[Any, Any]:
            x_candidate, _old_residual_norm, _old_normal_residual = payload
            residual_next = A.codomain.add(A.apply(x_candidate), A.codomain.scale(-1.0, b))
            return A.codomain.norm(residual_next), A.domain.norm(A.H.apply(residual_next))

        def keep_residuals(payload: tuple[Any, Any, Any]) -> tuple[Any, Any]:
            _x_candidate, old_residual_norm, old_normal_residual = payload
            return old_residual_norm, old_normal_residual

        residual_norm_next, normal_residual_norm_next = A.ops.cond(
            should_check_iteration(k_next, maxiter, check_every),
            refresh_residuals,
            keep_residuals,
            (x_next, _residual_norm, _normal_residual),
        )
        return (
            x_next,
            u_next,
            v_next,
            w_next,
            alpha_next,
            beta_next,
            rho_bar_next,
            phi_bar_next,
            residual_norm_next,
            normal_residual_norm_next,
            k_next,
        )

    x, *_rest, residual_norm, normal_residual_norm, num_iters = A.ops.while_loop(
        cond_fun,
        body_fun,
        (x, u, v, w, alpha, beta, rho_bar, phi_bar, residual_norm, normal_residual_norm, 0),
    )
    return LSQRResult(
        x,
        is_converged(normal_residual_norm, threshold_value),
        num_iters,
        residual_norm,
        normal_residual_norm,
    )
