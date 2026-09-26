from __future__ import annotations

from typing import Any, NamedTuple


from ..linop import LinOp
from ..space import DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace
from ..types import DenseArray
from ._utils import DEFAULT_CONVERGENCE_CHECK_INTERVAL, SpaceCoreOps, check_interval
from ._utils import require_linop, require_square, resolve_apply, safe_inverse_nonneg
from ._utils import result_repr, should_check_iteration


class LanczosResult(NamedTuple):
    """
    Store the result returned by :func:`lanczos_smallest`.

    Parameters
    ----------
    eigenvalue : scalar
        Ritz approximation to the smallest eigenvalue.
    eigenvector : array-like
        Ritz vector in ``A.domain``.
    residual_norm : scalar
        Standard Ritz residual estimate.
    krylov_dim : int-like
        Krylov dimension reached before breakdown or ``max_iter``.
    converged : bool-like
        Whether ``residual_norm < tol``.
    """

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


class _LanczosBasisResult(NamedTuple):
    """Store fixed-size Lanczos basis data and tridiagonal projection."""

    V: DenseArray
    T: DenseArray
    alphas: DenseArray
    betas: DenseArray
    krylov_dim: Any
    initial_norm: Any
    tol: Any
    e0_unit: DenseArray


def _check_lanczos_max_iter(max_iter: int) -> int:
    """Validate and normalize the maximum Lanczos iteration count."""
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    return max_iter


def _build_tridiagonal(
    ops: Any,
    alphas: DenseArray,
    betas: DenseArray,
    max_iter: int,
    m: Any,
    real_dtype: Any,
) -> DenseArray:
    """Build the fixed-size tridiagonal Lanczos projection."""
    # Fixed-size T for jit: rows past the Krylov dimension m get a large sentinel
    # diagonal so their eigenvalues sort above the real ones, and a zero coupling.
    idx = ops.arange(max_iter)
    full_indices = ops.arange(max_iter + 1)
    mask_alpha = idx < m
    inactive_sentinel = (
        ops.max(ops.abs(alphas))
        + 2.0 * ops.max(ops.abs(betas))
        + ops.asarray(1.0, dtype=real_dtype)
    )
    alphas_full = ops.where(mask_alpha, alphas, inactive_sentinel)
    betas_full = ops.where(full_indices == m, ops.asarray(0.0, dtype=real_dtype), betas)

    # Fill the diagonal, then the symmetric off-diagonals.
    T = ops.zeros((max_iter, max_iter), dtype=real_dtype)

    def fill_diag(ii: int, T_in: DenseArray) -> DenseArray:
        return ops.index_set(T_in, (ii, ii), alphas_full[ii], copy=True)

    T = ops.fori_loop(0, max_iter, fill_diag, T)

    def fill_off(ii: int, T_in: DenseArray) -> DenseArray:
        b = betas_full[ii + 1]
        T_in = ops.index_set(T_in, (ii, ii + 1), b, copy=True)
        T_in = ops.index_set(T_in, (ii + 1, ii), b, copy=True)
        return T_in

    return ops.fori_loop(0, max_iter - 1, fill_off, T)


def _lanczos_basis_and_tridiag(
    A: LinOp,
    initial_vector: Any,
    max_iter: int,
    tol: float,
    real_dtype: Any,
    check_every: int,
) -> _LanczosBasisResult:
    """Build a Lanczos basis and tridiagonal projection."""
    ops = A.ops
    ctx = A.ctx
    apply = resolve_apply(A)
    dom = SpaceCoreOps(A.domain)
    # Plain Euclidean coordinate spaces reorthogonalize with one einsum.
    use_euclidean_reorth = (
        type(A.domain) in (DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace)
        and A.domain.is_euclidean
    )

    # Preallocate the basis V and coefficients at full size, for jit.
    v0 = A.domain.flatten(initial_vector)
    v0 = ctx.assert_dense(v0)
    n = v0.shape[0]

    V = ops.zeros((max_iter + 1, n), dtype=ctx.dtype)
    alphas = ops.zeros((max_iter,), dtype=real_dtype)
    betas = ops.zeros((max_iter + 1,), dtype=real_dtype)

    tol_s = ops.asarray(tol, dtype=real_dtype)
    eps_s = ops.asarray(1e-12, dtype=real_dtype)

    # First basis vector: v0 normalized, or e0 if v0 is (numerically) zero.
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
    m_true0 = ops.asarray(max_iter)
    keep_going0 = ops.asarray(True)

    full_indices = ops.arange(max_iter + 1)
    coeffs_zero = ops.zeros((max_iter + 1,), dtype=ctx.dtype)

    def cond_fun(state: tuple[Any, Any, Any, Any, Any, Any, Any]) -> Any:
        i, _V, _alphas, _betas, _beta, m_true, keep_going = state
        return (i < max_iter) & keep_going

    def body_fun(
        state: tuple[Any, Any, Any, Any, Any, Any, Any],
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        i, V_, alphas_, betas_, beta, m_true, keep_going = state

        # Three-term recurrence: w = A v_i - alpha_i v_i - beta_i v_{i-1}.
        v_i = V_[i]
        v_i_member = A.domain.unflatten(v_i)
        w_member = apply(v_i_member)
        w = A.codomain.flatten(w_member)
        w = ctx.assert_dense(w)

        alpha = dom.real_inner(v_i_member, w_member)
        alphas_ = ops.index_set(alphas_, (i,), alpha, copy=True)

        w = ops.cond(
            i == 0,
            lambda w_in: w_in - alpha * v_i,
            lambda w_in: w_in - alpha * v_i - betas_[i] * V_[i - 1],
            w,
        )

        # Full reorthogonalization against the basis built so far (masked).
        w_member = A.domain.unflatten(w)
        valid = full_indices < (i + 1)
        mask = ops.where(
            valid,
            ops.asarray(1.0, dtype=real_dtype),
            ops.asarray(0.0, dtype=real_dtype),
        )
        mask = ops.astype(mask, ctx.dtype)

        if use_euclidean_reorth:
            coeffs_full = ops.einsum("jn,n->j", ops.conj(V_), w)
        else:
            coeffs_full = coeffs_zero

            def fill_coeff(j: int, coeffs_in: DenseArray) -> DenseArray:
                v_j_member = A.domain.unflatten(V_[j])
                coeff = dom.inner(v_j_member, w_member)
                return ops.index_set(coeffs_in, (j,), coeff, copy=True)

            coeffs_full = ops.fori_loop(0, max_iter + 1, fill_coeff, coeffs_full)
        coeffs_valid = coeffs_full * mask
        proj = ops.sum(coeffs_valid[:, None] * V_, axis=0)
        w = w - proj

        # beta_{i+1} = ||w||; below tol the Krylov space is exhausted (breakdown).
        w_member = A.domain.unflatten(w)
        beta_new = dom.norm(w_member)
        betas_ = ops.index_set(betas_, (i + 1,), beta_new, copy=True)
        breakdown = beta_new < tol_s
        m_true_next = ops.where(breakdown & (m_true == max_iter), i + 1, m_true)

        # Store the next unit vector unless breakdown happened.
        def set_next(V_in: DenseArray) -> DenseArray:
            w_unit = A.domain.flatten(dom.scale(safe_inverse_nonneg(ops, beta_new), w_member))
            return ops.index_set(V_in, (i + 1, slice(None)), w_unit, copy=True)

        V_ = ops.cond(beta_new >= tol_s, set_next, lambda V_in: V_in, V_)
        i_next = i + 1
        # Stop only on check iterations, to keep the loop cheap.
        keep_going_next = ops.cond(
            should_check_iteration(i_next, max_iter, check_every),
            lambda _: beta_new >= tol_s,
            lambda _: keep_going,
            ops.asarray(0.0, dtype=real_dtype),
        )

        return i_next, V_, alphas_, betas_, beta_new, m_true_next, keep_going_next

    i_final, V, alphas, betas, _beta_final, m_true, _keep_going = ops.while_loop(
        cond_fun, body_fun, (i0, V, alphas, betas, beta0, m_true0, keep_going0)
    )
    # Krylov dimension actually reached, and the projected tridiagonal T.
    m = ops.minimum(m_true, ops.asarray(i_final))
    T = _build_tridiagonal(ops, alphas, betas, max_iter, m, real_dtype)
    return _LanczosBasisResult(V, T, alphas, betas, m, v0_norm, tol_s, e0_unit)


def lanczos_smallest(
    A: LinOp,
    initial_vector: Any,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    check_every: int = DEFAULT_CONVERGENCE_CHECK_INTERVAL,
) -> LanczosResult:
    r"""
    Approximate the smallest eigenpair of a Hermitian operator by Lanczos.

    The eigenvalue returned is the Rayleigh quotient of the Ritz vector, reconstructed
    in ``A.domain``.

    Parameters
    ----------
    A : LinOp
        Square operator, Hermitian in the domain's inner product. Known non-Hermitian
        operators are rejected; unknown ones are trusted.
    initial_vector : array-like
        Starting vector in ``A.domain``; a zero vector falls back to a coordinate vector.
    max_iter : int, optional
        Maximum Krylov dimension; a static argument under ``jax.jit``.
    tol : float, optional
        Breakdown threshold, and the residual below which ``converged`` is set.
    check_every : int, optional
        Test for breakdown every this many iterations, and on the last one.

    Returns
    -------
    LanczosResult
        ``(eigenvalue, eigenvector, residual_norm, krylov_dim, converged)``.

    Raises
    ------
    TypeError
        If ``A`` is not a :class:`LinOp`.
    ValueError
        If ``A`` is not square, is known non-Hermitian, or ``max_iter`` is invalid.

    See Also
    --------
    power_iteration : The dominant eigenpair.

    Notes
    -----
    ``residual_norm`` is the recurrence estimate :math:`\beta_m |y_{m-1}|`. The smallest
    Ritz value can recover the smallest eigenvalue only if ``initial_vector`` overlaps
    its eigenspace. Trefethen & Bau, *Numerical Linear Algebra*, Algorithm 36.1 and
    Thm 36.1; multiplying the Lanczos relation by a unit Ritz vector gives the residual.
    Orthonormal coordinates give the metric version, with conjugate inner products
    for complex spaces. JIT-compatible on JAX with ``max_iter`` and ``check_every`` static.

    Examples
    --------
    Approximate the smallest eigenpair of a diagonal operator.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((3,), ctx)
    >>> A = sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 4.0]), X, ctx)
    >>> result = sc.lanczos_smallest(A, ctx.asarray([1.0, 1.0, 1.0]), max_iter=3)
    >>> np.allclose(result.eigenvalue, 1.0)
    True
    """
    # Validate once: square, not provably non-Hermitian, valid parameters.
    A = require_linop(A)
    require_square(A, "lanczos_smallest")
    if A.is_hermitian() is False:
        raise ValueError("lanczos_smallest requires A to be Hermitian/self-adjoint.")
    max_iter = _check_lanczos_max_iter(max_iter)
    check_every = check_interval(check_every)
    A.domain.check_member(initial_vector)
    ops = A.ops
    ctx = A.ctx
    real_dtype = ops.real_dtype(ctx.dtype)
    idx = ops.arange(max_iter)
    basis = _lanczos_basis_and_tridiag(A, initial_vector, max_iter, tol, real_dtype, check_every)

    # Smallest eigenpair of T; residual estimate beta_m |y_{m-1}|.
    m = basis.krylov_dim
    _eigvals, eigvecs = ops.eigh(basis.T)
    y_full = eigvecs[:, 0]
    residual_norm = basis.betas[m] * ops.abs(y_full[m - 1])
    converged = residual_norm < basis.tol

    # Ritz vector x = V y, using only the m active basis vectors.
    mask_y = ops.where(
        idx < m,
        ops.asarray(1.0, dtype=real_dtype),
        ops.asarray(0.0, dtype=real_dtype),
    )
    mask_y = ops.astype(mask_y, y_full.dtype)
    y_valid = y_full * mask_y

    V_reduced = basis.V[:max_iter, :]
    x_flat = ops.einsum("j,jn->n", y_valid, V_reduced)

    # Normalize x, falling back to e0 if it vanished.
    x_member = A.domain.unflatten(x_flat)
    x_norm = A.domain.norm(x_member)
    x_flat = ops.cond(
        x_norm > ops.asarray(1e-12, dtype=real_dtype),
        lambda _: A.domain.flatten(A.domain.scale(safe_inverse_nonneg(ops, x_norm), x_member)),
        lambda _: basis.e0_unit,
        ops.asarray(0.0, dtype=real_dtype),
    )

    # Report the Rayleigh quotient <x, A x> / <x, x> in the original space.
    x = A.domain.unflatten(x_flat)
    Ax = A.apply(x)

    num = ops.real(A.domain.inner(x, Ax))
    den = ops.real(A.domain.inner(x, x))
    lam = num / den

    return LanczosResult(lam, x, residual_norm, m, converged)
