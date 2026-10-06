from __future__ import annotations

from typing import Any, NamedTuple


from ..linop import LinOp
from ..linop._metric import _VECTOR_SPACES
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
        return result_repr("LanczosResult", self._asdict())


class _LanczosBasisResult(NamedTuple):
    """Store fixed-size Lanczos basis data and tridiagonal projection."""

    V: Any
    T: DenseArray
    betas: DenseArray
    krylov_dim: Any
    initial_norm: Any
    tol: Any
    initial_unit: Any


class _KrylovBasis:
    """Private storage adapter; vector arithmetic stays in the space algebra.

    Eager custom elements are held unchanged in a list. JAX custom elements
    must be registered pytrees with fixed leaf shapes and dtypes; their leaves
    are stacked separately without converting elements to coordinates.
    """

    def __init__(self, space: Any, example: Any, capacity: int) -> None:
        self.space = space
        self.ops = space.ops
        self.dom = SpaceCoreOps(space)
        self.capacity = capacity
        self.dense = type(space) in _VECTOR_SPACES
        self.euclidean = self.dense and space.is_euclidean
        self.tree = None
        if self.dense:
            self.shape = example.shape
        elif self.ops.family == "jax":
            self.tree = self.ops.jax.tree_util
            _, self.structure = self.tree.tree_flatten(example)

    def allocate(self) -> Any:
        if self.dense:
            return self.ops.zeros((self.capacity, *self.shape), dtype=self.space.dtype)
        zero = self.space.zeros()
        if self.tree is not None:
            leaves, structure = self.tree.tree_flatten(zero)
            if structure != self.structure:
                raise TypeError("Krylov elements and space.zeros() must have the same pytree structure.")
            return tuple(self.ops.stack([leaf] * self.capacity) for leaf in leaves)
        # ponytail: native lists cover eager custom objects; specialize only if profiling warrants it.
        return [zero] * self.capacity

    def get(self, storage: Any, i: Any) -> Any:
        if self.tree is not None:
            return self.tree.tree_unflatten(self.structure, [leaf[i] for leaf in storage])
        return storage[i]

    def set(self, storage: Any, i: Any, x: Any) -> Any:
        if self.dense:
            return self.ops.index_set(storage, i, x, copy=True)
        if self.tree is not None:
            leaves, structure = self.tree.tree_flatten(x)
            if structure != self.structure:
                raise TypeError("Krylov operations must preserve the element pytree structure.")
            return tuple(self.ops.index_set(a, i, b, copy=True) for a, b in zip(storage, leaves))
        updated = storage.copy()
        updated[i] = x
        return updated

    def inner(self, storage: Any, x: Any, count: Any) -> Any:
        if self.euclidean:
            vectors = self.ops.reshape(storage, (self.capacity, -1))
            coeffs = self.ops.einsum("jn,n->j", self.ops.conj(vectors), self.ops.reshape(x, (-1,)))
            return self.ops.where(self.ops.arange(self.capacity) < count, coeffs, self.ops.zeros_like(coeffs))
        coeffs = self.ops.zeros((self.capacity,), dtype=self.space.dtype)

        def body(j: Any, values: Any) -> Any:
            return self.ops.index_set(values, j, self.dom.inner(self.get(storage, j), x), copy=True)

        return self.ops.fori_loop(0, count, body, coeffs)

    def lincomb(self, storage: Any, coeffs: Any, count: Any) -> Any:
        if self.dense:
            vectors = self.ops.reshape(storage, (self.capacity, -1))[:coeffs.shape[0]]
            return self.ops.reshape(self.ops.einsum("j,jn->n", coeffs, vectors), self.shape)

        def body(j: Any, x: Any) -> Any:
            return self.dom.axpy(coeffs[j], self.get(storage, j), x)

        return self.ops.fori_loop(0, count, body, self.space.zeros())


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

    # The diagonal, then the symmetric off-diagonals.
    off = betas_full[1:max_iter]
    return ops.diag(alphas_full) + ops.diag(off, 1) + ops.diag(off, -1)


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
    apply = resolve_apply(A)
    dom = SpaceCoreOps(A.domain)
    basis = _KrylovBasis(A.domain, initial_vector, max_iter + 1)
    V = basis.allocate()
    alphas = ops.zeros((max_iter,), dtype=real_dtype)
    betas = ops.zeros((max_iter + 1,), dtype=real_dtype)
    tol_s = ops.asarray(tol, dtype=real_dtype)

    v0_norm = dom.norm(initial_vector)
    v0_unit = dom.scale(safe_inverse_nonneg(ops, v0_norm), initial_vector)
    V = basis.set(V, 0, v0_unit)

    beta0 = ops.asarray(1.0, dtype=real_dtype)
    i0 = 0
    m_true0 = ops.asarray(max_iter)
    keep_going0 = v0_norm > 0

    def cond_fun(state: tuple[Any, Any, Any, Any, Any, Any, Any]) -> Any:
        i, _V, _alphas, _betas, _beta, m_true, keep_going = state
        return (i < max_iter) & keep_going

    def body_fun(
        state: tuple[Any, Any, Any, Any, Any, Any, Any],
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        i, V_, alphas_, betas_, beta, m_true, keep_going = state

        # Three-term recurrence: w = A v_i - alpha_i v_i - beta_i v_{i-1}.
        v_i = basis.get(V_, i)
        w = apply(v_i)
        alpha = dom.real_inner(v_i, w)
        alphas_ = ops.index_set(alphas_, i, alpha, copy=True)
        w = dom.axpy(-alpha, v_i, w)
        w = ops.cond(
            i == 0,
            lambda w_in: w_in,
            lambda w_in: dom.axpy(-betas_[i], basis.get(V_, i - 1), w_in),
            w,
        )

        # Full reorthogonalization using the declared inner product.
        coeffs = basis.inner(V_, w, i + 1)
        w_member = dom.axpy(-1, basis.lincomb(V_, coeffs, i + 1), w)

        # beta_{i+1} = ||w||; below tol the Krylov space is exhausted.
        beta_new = dom.norm(w_member)
        betas_ = ops.index_set(betas_, (i + 1,), beta_new, copy=True)
        breakdown = beta_new < tol_s
        m_true_next = ops.where(breakdown & (m_true == max_iter), i + 1, m_true)

        # Store the next unit vector unless breakdown happened.
        def set_next(V_in: Any) -> Any:
            w_unit = dom.scale(safe_inverse_nonneg(ops, beta_new), w_member)
            return basis.set(V_in, i + 1, w_unit)

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
    return _LanczosBasisResult(V, T, betas, m, v0_norm, tol_s, v0_unit)


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
        Nonzero starting element in ``A.domain``.
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
        If ``A`` is not square, is known non-Hermitian, ``max_iter`` is invalid,
        or the starting element is zero (eager execution).

    See Also
    --------
    power_iteration : The dominant eigenpair.

    Notes
    -----
    ``residual_norm`` is the recurrence estimate :math:`\beta_m |y_{m-1}|`. The smallest
    Ritz value can recover the smallest eigenvalue only if ``initial_vector`` overlaps
    its eigenspace. Trefethen & Bau, *Numerical Linear Algebra*, Algorithm 36.1 and
    Thm 36.1; multiplying the Lanczos relation by a unit Ritz vector gives the residual.
    Uses the space algebra and native elements. Custom JAX elements must be
    registered pytrees whose structure, leaf shapes, and dtypes stay fixed.
    ``max_iter`` and ``check_every`` must be static under ``jax.jit``.
    A zero start raises eagerly; under JAX tracing it returns a NaN eigenvalue,
    a zero eigenvector, and ``converged=False``.

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
    initial_norm = A.domain.norm(initial_vector)
    try:
        zero_start = bool(initial_norm == 0)
    except TypeError:  # JAX traced predicate; report invalid input in the result below.
        zero_start = False
    if zero_start:
        raise ValueError("lanczos_smallest requires a nonzero initial_vector.")
    basis = _lanczos_basis_and_tridiag(A, initial_vector, max_iter, tol, real_dtype, check_every)

    # Smallest eigenpair of T; residual estimate beta_m |y_{m-1}|.
    m = basis.krylov_dim
    _eigvals, eigvecs = ops.eigh(basis.T)
    y_full = eigvecs[:, 0]
    residual_norm = basis.betas[m] * ops.abs(y_full[m - 1])
    converged = (residual_norm < basis.tol) & (basis.initial_norm > 0)

    # Ritz vector x = V y, using only the m active basis vectors.
    mask_y = ops.where(
        idx < m,
        ops.asarray(1.0, dtype=real_dtype),
        ops.asarray(0.0, dtype=real_dtype),
    )
    mask_y = ops.astype(mask_y, y_full.dtype)
    y_valid = y_full * mask_y

    vectors = _KrylovBasis(A.domain, initial_vector, max_iter + 1)
    x = vectors.lincomb(basis.V, y_valid, m)
    x_norm = A.domain.norm(x)
    x = ops.cond(
        x_norm > 0,
        lambda _: A.domain.scale(safe_inverse_nonneg(ops, x_norm), x),
        lambda _: basis.initial_unit,
        ops.asarray(0.0, dtype=real_dtype),
    )

    # Report the Rayleigh quotient in the original space.
    Ax = A.apply(x)

    num = ops.real(A.domain.inner(x, Ax))
    den = ops.real(A.domain.inner(x, x))
    lam = ops.where(den > 0, num * safe_inverse_nonneg(ops, den), ops.asarray(float("nan"), dtype=real_dtype))

    return LanczosResult(lam, x, residual_norm, m, converged)
