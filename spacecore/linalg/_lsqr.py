from __future__ import annotations

from typing import Any, NamedTuple

from ..linop import LinOp
from ._utils import DEFAULT_CONVERGENCE_CHECK_INTERVAL, SpaceCoreOps, check_interval
from ._utils import check_maxiter, is_converged, require_linop, resolve_apply, resolve_rapply
from ._utils import safe_inverse_nonneg, should_check_iteration, result_repr, threshold


class LSQRResult(NamedTuple):
    """
    Result of :func:`lsqr`.

    Parameters
    ----------
    x : array-like
        Least-squares solution in ``A.domain``.
    converged : bool-like
        Whether the tolerance was met.
    num_iters : int-like
        Iterations executed.
    residual_norm : scalar
        ``norm(A x - b)``, exact or LSQR's estimate per ``residual_mode``.
    normal_residual_norm : scalar
        ``norm(A.H (A x - b))``, exact or estimated likewise.
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
    residual_mode: str = "exact",
) -> LSQRResult:
    r"""
    Solve :math:`\min_x \|A x - b\|` by LSQR.

    ``A`` may map between different spaces. Only ``A.apply`` and ``A.H.apply`` are
    used; the normal equations are never formed.

    Parameters
    ----------
    A : LinOp
        Operator from ``A.domain`` to ``A.codomain``.
    b : array-like
        Right-hand side in ``A.codomain``.
    x0 : array-like or None, optional
        Initial guess; zero by default.
    tol : float, optional
        Converged when ``norm(A.H @ (A @ x - b))`` is below ``atol + tol * norm(b)``.
    atol : float, optional
        Absolute tolerance on that normal-equation residual.
    maxiter : int or None, optional
        Iteration cap; ``prod(A.domain.shape)`` by default.
    check_every : int, optional
        Refresh residual diagnostics every this many iterations, and on the last one.
    residual_mode : {"exact", "recurrence"}, optional
        ``"exact"`` recomputes both residuals at each check, costing one extra forward
        and adjoint apply; ``"recurrence"`` uses LSQR's scalar estimates for free.

    Returns
    -------
    LSQRResult
        ``(x, converged, num_iters, residual_norm, normal_residual_norm)``.

    Raises
    ------
    TypeError
        If ``A`` is not a :class:`LinOp`.
    ValueError
        If an iteration parameter or ``residual_mode`` is invalid.

    See Also
    --------
    cg : Square Hermitian positive-definite systems.

    Notes
    -----
    JIT-compatible on JAX with ``maxiter``, ``check_every`` and ``residual_mode``
    static. A small normal-equation residual does not imply a small error when ``A`` is
    ill-conditioned. Paige and Saunders, ACM Trans. Math. Soft. 8 (1982).

    Examples
    --------
    Solve a small overdetermined least-squares problem.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> Y = sc.DenseCoordinateSpace((3,), ctx)
    >>> M = ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> A = sc.DenseLinOp(M, X, Y, ctx)
    >>> b = ctx.asarray([1.0, 2.0, 3.0])
    >>> result = sc.lsqr(A, b, tol=1e-10)
    >>> np.allclose(result.x, [1.0, 2.0])
    True
    """
    # Validate once, up front.
    A = require_linop(A)
    A.codomain.check_member(b)
    maxiter = check_maxiter(maxiter, A)
    check_every = check_interval(check_every)
    if residual_mode not in {"exact", "recurrence"}:
        raise ValueError("residual_mode must be 'exact' or 'recurrence'.")

    # Resolve check-free cores once; the hot loop then skips per-iteration
    # validation while honoring any custom geometry the spaces define.
    apply = resolve_apply(A)
    radj = resolve_rapply(A)
    dom = SpaceCoreOps(A.domain)
    cod = SpaceCoreOps(A.codomain)

    # Golub-Kahan start: beta u = b - A x and alpha v = A.H u.
    x = A.domain.zeros() if x0 is None else x0
    A.domain.check_member(x)
    residual = cod.add(b, cod.scale(-1.0, apply(x)))
    beta = cod.norm(residual)
    u = residual
    u = cod.scale(safe_inverse_nonneg(A.ops, beta), u)
    v = radj(u)
    alpha = dom.norm(v)
    # Initial normal-equation residual ||A.H r||, exact or from the recurrence.
    if residual_mode == "exact":
        normal_residual_norm = dom.norm(radj(residual))
    else:
        normal_residual_norm = beta * alpha
    v = dom.scale(safe_inverse_nonneg(A.ops, alpha), v)
    # LSQR state: search direction w and the scalar recurrences phi_bar, rho_bar.
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
        # Bidiagonalization step: next unit vectors u and v.
        u_next = cod.axpy(-alpha, u, apply(v))
        beta_next = cod.norm(u_next)
        u_next = cod.scale(safe_inverse_nonneg(A.ops, beta_next), u_next)

        v_next = dom.axpy(-beta_next, v, radj(u_next))
        alpha_next = dom.norm(v_next)
        v_next = dom.scale(safe_inverse_nonneg(A.ops, alpha_next), v_next)

        # Givens rotation (c, s) that eliminates the new subdiagonal entry.
        rho = A.ops.sqrt(rho_bar * rho_bar + beta_next * beta_next)
        inv_rho = safe_inverse_nonneg(A.ops, rho)
        c = rho_bar * inv_rho
        s = beta_next * inv_rho
        theta = s * alpha_next
        rho_bar_next = -c * alpha_next
        phi = c * phi_bar
        phi_bar_next = s * phi_bar
        # Free residual estimates from the recurrence.
        recurrence_residual_norm = A.ops.abs(phi_bar_next)
        recurrence_normal_residual_norm = alpha_next * A.ops.abs(s * phi)

        # Update the solution and the search direction.
        x_next = dom.axpy(phi * inv_rho, w, x)
        w_next = dom.axpy(-(theta * inv_rho), w, v_next)
        k_next = k + 1

        # Exact mode recomputes the residuals (one forward and one adjoint apply).
        if residual_mode == "exact":

            def refresh_residuals(payload: tuple[Any, Any, Any]) -> tuple[Any, Any]:
                x_candidate, _old_residual_norm, _old_normal_residual = payload
                residual_next = cod.add(apply(x_candidate), cod.scale(-1.0, b))
                return cod.norm(residual_next), dom.norm(radj(residual_next))
        else:

            def refresh_residuals(payload: tuple[Any, Any, Any]) -> tuple[Any, Any]:
                _x_candidate, _old_residual_norm, _old_normal_residual = payload
                return recurrence_residual_norm, recurrence_normal_residual_norm

        def keep_residuals(payload: tuple[Any, Any, Any]) -> tuple[Any, Any]:
            _x_candidate, old_residual_norm, old_normal_residual = payload
            return old_residual_norm, old_normal_residual

        # Refresh the reported residuals only on check iterations.
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
