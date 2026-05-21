from __future__ import annotations

from typing import Any

from ..linop import LinOp
from ..types import DenseArray
from ._utils import DEFAULT_CONVERGENCE_CHECK_INTERVAL, check_interval
from ._utils import require_linop, require_square, should_check_iteration


def _check_lanczos_max_iter(max_iter: int) -> int:
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    return max_iter


def stochastic_lanczos(
    A: LinOp,
    initial_vector: Any,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    check_every: int = DEFAULT_CONVERGENCE_CHECK_INTERVAL,
) -> tuple[DenseArray, Any]:
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
        A pair ``(eigenvalue, eigenvector)`` for the smallest approximated
        eigenpair.
    """
    A = require_linop(A)
    require_square(A, "stochastic_lanczos")
    max_iter = _check_lanczos_max_iter(max_iter)
    check_every = check_interval(check_every)
    A.domain.check_member(initial_vector)
    ops = A.ops
    ctx = A.ctx

    v0 = A.domain.flatten(initial_vector)
    v0 = ctx.assert_dense(v0)
    n = v0.shape[0]

    V = ops.zeros((max_iter + 1, n), dtype=ctx.dtype)
    alphas = ops.zeros((max_iter,), dtype=ctx.dtype)
    betas = ops.zeros((max_iter + 1,), dtype=ctx.dtype)

    tol_s = ctx.asarray(tol)
    eps_s = ctx.asarray(1e-12)

    v0_norm = ops.sqrt(ops.real(ops.vdot(v0, v0)))

    e0 = ops.zeros((n,), dtype=ctx.dtype)
    e0 = ops.index_set(e0, (0,), ctx.asarray(1.0), copy=True)

    v0_unit = ops.cond(
        v0_norm > eps_s,
        lambda _: v0 / v0_norm,
        lambda _: e0,
        ctx.asarray(0.0),
    )
    V = ops.index_set(V, (0, slice(None)), v0_unit, copy=True)

    beta0 = ctx.asarray(1.0)
    i0 = 0
    keep_going0 = ops.asarray(True)

    full_indices = ops.arange(max_iter + 1)
    idx = ops.arange(max_iter)

    def cond_fun(state: tuple[Any, Any, Any, Any, Any, Any]) -> Any:
        i, _V, _alphas, _betas, _beta, keep_going = state
        return (i < max_iter) & keep_going

    def body_fun(state: tuple[Any, Any, Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any, Any, Any]:
        i, V_, alphas_, betas_, beta, keep_going = state

        v_i = V_[i]
        w = A.codomain.flatten(A.apply(A.domain.unflatten(v_i)))
        w = ctx.assert_dense(w)

        alpha = ops.real(ops.vdot(v_i, w))
        alphas_ = ops.index_set(alphas_, (i,), alpha, copy=True)

        w = ops.cond(
            i == 0,
            lambda w_in: w_in - alpha * v_i,
            lambda w_in: w_in - alpha * v_i - betas_[i] * V_[i - 1],
            w,
        )

        valid = full_indices < (i + 1)
        mask = ops.where(valid, ctx.asarray(1.0), ctx.asarray(0.0))
        mask = ops.astype(mask, w.dtype)

        coeffs_full = ops.einsum("jn,n->j", ops.conj(V_), w)
        coeffs_valid = coeffs_full * mask
        proj = ops.sum(coeffs_valid[:, None] * V_, axis=0)
        w = w - proj

        beta_new = ops.sqrt(ops.real(ops.vdot(w, w)))
        betas_ = ops.index_set(betas_, (i + 1,), beta_new, copy=True)

        def set_next(V_in: DenseArray) -> DenseArray:
            return ops.index_set(V_in, (i + 1, slice(None)), w / beta_new, copy=True)

        V_ = ops.cond(beta_new >= tol_s, set_next, lambda V_in: V_in, V_)
        i_next = i + 1
        keep_going_next = ops.cond(
            should_check_iteration(i_next, max_iter, check_every),
            lambda _: beta_new >= tol_s,
            lambda _: keep_going,
            ctx.asarray(0.0),
        )

        return i_next, V_, alphas_, betas_, beta_new, keep_going_next

    i_final, V, alphas, betas, _beta_final, _keep_going = ops.while_loop(
        cond_fun, body_fun, (i0, V, alphas, betas, beta0, keep_going0)
    )
    m = i_final

    mask_alpha = idx < m
    alphas_full = ops.where(mask_alpha, alphas, ctx.asarray(1e10))
    betas_full = ops.where(full_indices == m, ctx.asarray(0.0), betas)

    T = ops.zeros((max_iter, max_iter), dtype=ctx.dtype)

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

    mask_y = ops.where(idx < m, ctx.asarray(1.0), ctx.asarray(0.0))
    mask_y = ops.astype(mask_y, y_full.dtype)
    y_valid = y_full * mask_y

    V_reduced = V[:max_iter, :]
    x_flat = ops.einsum("j,jn->n", y_valid, V_reduced)

    x_norm = ops.sqrt(ops.real(ops.vdot(x_flat, x_flat)))
    x_flat = ops.cond(
        x_norm > eps_s,
        lambda _: x_flat / x_norm,
        lambda _: e0,
        ctx.asarray(0.0),
    )

    x = A.domain.unflatten(x_flat)
    Ax = A.apply(x)

    num = ops.real(A.domain.inner(x, Ax))
    den = ops.real(A.domain.inner(x, x))
    lam = num / den

    return lam, x
