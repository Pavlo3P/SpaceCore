from __future__ import annotations

from typing import Any, NamedTuple

from ..linop import LinOp
from ._utils import DEFAULT_CONVERGENCE_CHECK_INTERVAL, check_interval, check_maxiter
from ._utils import is_converged, require_linop, safe_inverse, should_check_iteration
from ._utils import threshold


class LSQRResult(NamedTuple):
    """Result returned by :func:`lsqr`."""

    x: Any
    converged: Any
    num_iters: Any
    residual_norm: Any
    normal_residual_norm: Any


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
    """
    Solve ``min_x ||A x - b||_2`` by the LSQR Krylov iteration.

    The operator may be rectangular or square. The method uses ``A.apply`` for
    forward products and ``A.H.apply`` for adjoint products, so the normal
    equations are represented implicitly and no dense matrix is formed.
    Convergence is tested against ``atol + tol * ||b||`` using
    ``||A.H @ (A x - b)||``. That normal-equation residual is refreshed only
    every ``check_every`` iterations, and always on the final iteration, so the
    expensive stopping diagnostic is not evaluated on every Krylov step.
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
    u = A.codomain.scale(safe_inverse(A.ops, beta), u)
    v = A.H.apply(u)
    alpha = A.domain.norm(v)
    v = A.domain.scale(safe_inverse(A.ops, alpha), v)
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
        u_next = A.codomain.scale(safe_inverse(A.ops, beta_next), u_next)

        v_next = A.domain.axpy(-beta_next, v, A.H.apply(u_next))
        alpha_next = A.domain.norm(v_next)
        v_next = A.domain.scale(safe_inverse(A.ops, alpha_next), v_next)

        rho = A.ops.sqrt(rho_bar * rho_bar + beta_next * beta_next)
        inv_rho = safe_inverse(A.ops, rho)
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
