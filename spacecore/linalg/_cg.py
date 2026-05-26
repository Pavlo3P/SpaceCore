from __future__ import annotations

from typing import Any, NamedTuple

from ..linop import LinOp
from ._utils import DEFAULT_CONVERGENCE_CHECK_INTERVAL, check_interval, check_maxiter
from ._utils import is_converged, real_inner, require_linop, require_square
from ._utils import result_repr, safe_inverse_nonneg, should_check_iteration, threshold


class CGResult(NamedTuple):
    """Result returned by :func:`cg`."""

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
    """
    Solve ``A x = b`` by conjugate gradients.

    ``A`` must be a square symmetric/Hermitian positive-definite ``LinOp``.
    The implementation uses only ``A.apply`` and the domain-space inner product;
    it never materializes a dense matrix. The residual norm is compared with
    ``atol + tol * ||b||`` only every ``check_every`` iterations, and always on
    the final iteration. This avoids checking the stopping criterion on every
    step while remaining compatible with JAX JIT control flow.
    """
    A = require_linop(A)
    require_square(A, "cg")
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

    def cond_fun(carry: tuple[Any, Any, Any, Any, Any, int]) -> Any:
        _x, _r, _p, _rs, res_norm, k = carry
        return (k < maxiter) & (res_norm > threshold_value)

    def body_fun(carry: tuple[Any, Any, Any, Any, Any, int]) -> tuple[Any, Any, Any, Any, Any, int]:
        x, r, p, rs, _residual_norm, k = carry
        Ap = A.apply(p)
        pAp = real_inner(A.domain, p, Ap)
        active = (rs > eps) & (pAp > eps)
        alpha = A.ops.where(active, rs * safe_inverse_nonneg(A.ops, pAp), A.ops.zeros_like(rs))
        x_next = A.domain.axpy(alpha, p, x)
        r_next = A.codomain.axpy(-alpha, Ap, r)
        rs_next = real_inner(A.domain, r_next, r_next)
        beta = A.ops.where(active, rs_next * safe_inverse_nonneg(A.ops, rs), A.ops.zeros_like(rs_next))
        p_next = A.domain.axpy(beta, p, r_next)
        k_next = k + 1
        current_residual_norm = A.domain.norm(r_next)
        residual_norm_next = A.ops.cond(
            should_check_iteration(k_next, maxiter, check_every),
            lambda _: current_residual_norm,
            lambda _: _residual_norm,
            A.ops.asarray(0.0, dtype=A.dtype),
        )
        return x_next, r_next, p_next, rs_next, residual_norm_next, k_next

    x, _r, _p, _rs, residual_norm, num_iters = A.ops.while_loop(
        cond_fun,
        body_fun,
        (x, r, p, rs, residual_norm, 0),
    )
    return CGResult(x, is_converged(residual_norm, threshold_value), num_iters, residual_norm)
