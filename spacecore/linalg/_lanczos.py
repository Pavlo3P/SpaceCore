from __future__ import annotations

from typing import Any, NamedTuple
from warnings import warn


from ..linop import LinOp
from ..types import DenseArray
from ._utils import DEFAULT_CONVERGENCE_CHECK_INTERVAL, check_interval
from ._utils import require_linop, require_square, safe_inverse_nonneg, should_check_iteration
from ._utils import result_repr


class LanczosResult(NamedTuple):
    """Result returned by :func:`lanczos_smallest`."""

    eigenvalue: Any
    eigenvector: Any
    residual_norm: Any
    krylov_dim: Any
    converged: Any

    def __repr__(self) -> str:
        """Return a compact summary without printing the full eigenvector."""
        return result_repr(
            "LanczosResult",
            {
                "eigenvalue": self.eigenvalue,
                "eigenvector": self.eigenvector,
                "residual_norm": self.residual_norm,
                "krylov_dim": self.krylov_dim,
                "converged": self.converged,
            },
        )


StochasticLanczosResult = LanczosResult


def _check_lanczos_max_iter(max_iter: int) -> int:
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    return max_iter


def lanczos_smallest(
    A: LinOp,
    initial_vector: Any,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    check_every: int = DEFAULT_CONVERGENCE_CHECK_INTERVAL,
) -> LanczosResult:
    r"""Approximate the smallest eigenpair of a Hermitian operator.

    The operator is supplied as a square ``LinOp`` and ``initial_vector`` is an
    element of ``A.domain``. The implementation keeps fixed-size coordinate
    arrays for JAX compatibility, safely handles zero initial vectors, and
    refines the returned eigenvalue with the Rayleigh quotient of the
    reconstructed Ritz vector in the original space.

    Mathematically, Lanczos builds an orthonormal Krylov basis ``V`` for
    ``span{v, T v, T^2 v, ...}`` and a tridiagonal projection
    :math:`T_k = V^\dagger T V`. The returned vector is the Ritz vector
    reconstructed in the original coordinates, and the returned scalar is the
    Rayleigh quotient
    :math:`(x^\dagger T x) / (x^\dagger x)`.

    Args:
        A: Square Hermitian linear operator.
        initial_vector: Starting vector in ``A.domain``.
        max_iter: Maximum number of Lanczos steps.
        tol: Breakdown tolerance for the off-diagonal Lanczos coefficient.
        check_every: Refresh the breakdown-based stopping decision only every
            this many iterations, and always on the final iteration.

    Returns:
        ``LanczosResult`` containing the smallest approximated eigenpair, the
        standard Ritz residual estimate ``beta[m] * abs(y[m - 1])``, the
        Krylov dimension reached, and a convergence flag. The residual estimate
        is computed from the tridiagonal recurrence; callers that need the true
        residual can evaluate ``A.apply(eigenvector) - eigenvalue * eigenvector``
        once more in the original space.
    """
    A = require_linop(A)
    require_square(A, "lanczos_smallest")
    max_iter = _check_lanczos_max_iter(max_iter)
    check_every = check_interval(check_every)
    A.domain.check_member(initial_vector)
    ops = A.ops
    ctx = A.ctx
    real_dtype = ops.real_dtype(ctx.dtype)

    v0 = A.domain.flatten(initial_vector)
    v0 = ctx.assert_dense(v0)
    n = v0.shape[0]

    V = ops.zeros((max_iter + 1, n), dtype=ctx.dtype)
    alphas = ops.zeros((max_iter,), dtype=real_dtype)
    betas = ops.zeros((max_iter + 1,), dtype=real_dtype)

    tol_s = ops.asarray(tol, dtype=real_dtype)
    eps_s = ops.asarray(1e-12, dtype=real_dtype)

    v0_norm = A.domain.norm(initial_vector)

    e0 = ops.zeros((n,), dtype=ctx.dtype)
    e0 = ops.index_set(e0, (0,), ctx.asarray(1.0), copy=True)
    e0_member = A.domain.unflatten(e0)
    e0_norm = A.domain.norm(e0_member)
    e0_unit = A.domain.flatten(A.domain.scale(safe_inverse_nonneg(ops, e0_norm), e0_member))

    v0_unit = ops.cond(
        v0_norm > eps_s,
        lambda _: A.domain.flatten(
            A.domain.scale(safe_inverse_nonneg(ops, v0_norm), initial_vector)
        ),
        lambda _: e0_unit,
        ops.asarray(0.0, dtype=real_dtype),
    )
    V = ops.index_set(V, (0, slice(None)), v0_unit, copy=True)

    beta0 = ops.asarray(1.0, dtype=real_dtype)
    i0 = 0
    keep_going0 = ops.asarray(True)

    full_indices = ops.arange(max_iter + 1)
    idx = ops.arange(max_iter)
    coeffs_zero = ops.zeros((max_iter + 1,), dtype=ctx.dtype)

    def cond_fun(state: tuple[Any, Any, Any, Any, Any, Any]) -> Any:
        i, _V, _alphas, _betas, _beta, keep_going = state
        return (i < max_iter) & keep_going

    def body_fun(state: tuple[Any, Any, Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any, Any, Any]:
        i, V_, alphas_, betas_, beta, keep_going = state

        v_i = V_[i]
        v_i_member = A.domain.unflatten(v_i)
        w_member = A.apply(v_i_member)
        w = A.codomain.flatten(w_member)
        w = ctx.assert_dense(w)

        alpha = ops.real(A.domain.inner(v_i_member, w_member))
        alphas_ = ops.index_set(alphas_, (i,), alpha, copy=True)

        w = ops.cond(
            i == 0,
            lambda w_in: w_in - alpha * v_i,
            lambda w_in: w_in - alpha * v_i - betas_[i] * V_[i - 1],
            w,
        )

        w_member = A.domain.unflatten(w)
        valid = full_indices < (i + 1)
        mask = ops.where(
            valid,
            ops.asarray(1.0, dtype=real_dtype),
            ops.asarray(0.0, dtype=real_dtype),
        )
        mask = ops.astype(mask, ctx.dtype)

        coeffs_full = coeffs_zero

        def fill_coeff(j: int, coeffs_in: DenseArray) -> DenseArray:
            v_j_member = A.domain.unflatten(V_[j])
            coeff = A.domain.inner(v_j_member, w_member)
            return ops.index_set(coeffs_in, (j,), coeff, copy=True)

        coeffs_full = ops.fori_loop(0, max_iter + 1, fill_coeff, coeffs_full)
        coeffs_valid = coeffs_full * mask
        proj = ops.sum(coeffs_valid[:, None] * V_, axis=0)
        w = w - proj

        w_member = A.domain.unflatten(w)
        beta_new = A.domain.norm(w_member)
        betas_ = ops.index_set(betas_, (i + 1,), beta_new, copy=True)

        def set_next(V_in: DenseArray) -> DenseArray:
            w_unit = A.domain.flatten(A.domain.scale(safe_inverse_nonneg(ops, beta_new), w_member))
            return ops.index_set(V_in, (i + 1, slice(None)), w_unit, copy=True)

        V_ = ops.cond(beta_new >= tol_s, set_next, lambda V_in: V_in, V_)
        i_next = i + 1
        keep_going_next = ops.cond(
            should_check_iteration(i_next, max_iter, check_every),
            lambda _: beta_new >= tol_s,
            lambda _: keep_going,
            ops.asarray(0.0, dtype=real_dtype),
        )

        return i_next, V_, alphas_, betas_, beta_new, keep_going_next

    i_final, V, alphas, betas, _beta_final, _keep_going = ops.while_loop(
        cond_fun, body_fun, (i0, V, alphas, betas, beta0, keep_going0)
    )
    m = i_final

    mask_alpha = idx < m
    inactive_sentinel = (
        ops.max(ops.abs(alphas))
        + 2.0 * ops.max(ops.abs(betas))
        + ops.asarray(1.0, dtype=real_dtype)
    )
    alphas_full = ops.where(mask_alpha, alphas, inactive_sentinel)
    betas_full = ops.where(full_indices == m, ops.asarray(0.0, dtype=real_dtype), betas)

    T = ops.zeros((max_iter, max_iter), dtype=real_dtype)

    def fill_diag(ii: int, T_in: DenseArray) -> DenseArray:
        return ops.index_set(T_in, (ii, ii), alphas_full[ii], copy=True)

    T = ops.fori_loop(0, max_iter, fill_diag, T)

    def fill_off(ii: int, T_in: DenseArray) -> DenseArray:
        b = betas_full[ii + 1]
        T_in = ops.index_set(T_in, (ii, ii + 1), b, copy=True)
        T_in = ops.index_set(T_in, (ii + 1, ii), b, copy=True)
        return T_in

    T = ops.fori_loop(0, max_iter - 1, fill_off, T)

    _eigvals, eigvecs = ops.eigh(T)
    y_full = eigvecs[:, 0]
    residual_norm = betas[m] * ops.abs(y_full[m - 1])
    converged = residual_norm < tol_s

    mask_y = ops.where(
        idx < m,
        ops.asarray(1.0, dtype=real_dtype),
        ops.asarray(0.0, dtype=real_dtype),
    )
    mask_y = ops.astype(mask_y, y_full.dtype)
    y_valid = y_full * mask_y

    V_reduced = V[:max_iter, :]
    x_flat = ops.einsum("j,jn->n", y_valid, V_reduced)

    x_member = A.domain.unflatten(x_flat)
    x_norm = A.domain.norm(x_member)
    x_flat = ops.cond(
        x_norm > eps_s,
        lambda _: A.domain.flatten(A.domain.scale(safe_inverse_nonneg(ops, x_norm), x_member)),
        lambda _: e0_unit,
        ops.asarray(0.0, dtype=real_dtype),
    )

    x = A.domain.unflatten(x_flat)
    Ax = A.apply(x)

    num = ops.real(A.domain.inner(x, Ax))
    den = ops.real(A.domain.inner(x, x))
    lam = num / den

    return LanczosResult(lam, x, residual_norm, m, converged)


def stochastic_lanczos(
    A: LinOp,
    initial_vector: Any,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    check_every: int = DEFAULT_CONVERGENCE_CHECK_INTERVAL,
) -> LanczosResult:
    """
    Deprecated alias for :func:`lanczos_smallest`.

    Returns
    -------
    LanczosResult
        Result from :func:`lanczos_smallest`.
    """
    warn(
        "stochastic_lanczos is deprecated; use lanczos_smallest instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return lanczos_smallest(
        A,
        initial_vector,
        max_iter=max_iter,
        tol=tol,
        check_every=check_every,
    )
