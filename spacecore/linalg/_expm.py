from __future__ import annotations

from typing import Any, NamedTuple

from ..linop import LinOp
from ._lanczos import _check_lanczos_max_iter, _lanczos_basis_and_tridiag
from ._utils import require_linop, require_square, result_repr


class ExpmMultiplyResult(NamedTuple):
    """
    Store the result returned by :func:`expm_multiply`.

    Parameters
    ----------
    result : array-like
        Vector in the domain of the input operator approximating
        ``exp(t * A) @ v``.
    krylov_dim : int-like
        Actual Krylov dimension reached before breakdown or ``max_iter``.
    residual_estimate : scalar
        Projected exponential residual estimate
        ``abs(beta[m] * phi[m - 1])``.
    converged : bool-like
        Boolean indicating whether ``residual_estimate < tol``.
    """

    result: Any
    krylov_dim: Any
    residual_estimate: Any
    converged: Any

    def __repr__(self) -> str:
        """Return a compact summary without printing the full vector."""
        return result_repr(
            "ExpmMultiplyResult",
            {
                "converged": self.converged,
                "krylov_dim": self.krylov_dim,
                "residual_estimate": self.residual_estimate,
                "result": self.result,
            },
        )


def expm_multiply(
    A: LinOp,
    v: Any,
    t: float | complex = 1.0,
    *,
    max_iter: int = 30,
    tol: float = 1e-10,
) -> ExpmMultiplyResult:
    r"""
    Compute :math:`\exp(t A) v` by Lanczos projection, without forming ``A``.

    Parameters
    ----------
    A : LinOp
        Square operator, Hermitian in the domain's inner product. Known non-Hermitian
        operators are rejected; unknown ones are trusted.
    v : array-like
        Vector in ``A.domain``.
    t : float or complex, optional
        Scalar multiplying ``A``; a complex ``t`` needs a complex ``ctx.dtype``.
    max_iter : int, optional
        Maximum Krylov dimension; 20 to 50 usually suffices. Static under ``jax.jit``.
    tol : float, optional
        Breakdown threshold, and the residual below which ``converged`` is set.

    Returns
    -------
    ExpmMultiplyResult
        The result vector, the Krylov dimension used, the residual estimate
        :math:`|\beta_m \phi_{m-1}|`, and the convergence flag.

    Raises
    ------
    TypeError
        If ``A`` is not a :class:`LinOp`.
    ValueError
        If ``A`` is not square, is known non-Hermitian, or ``max_iter`` is invalid.

    See Also
    --------
    lanczos_smallest : The same Krylov projection, for eigenpairs.

    Examples
    --------
    Apply the exponential of a diagonal operator.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> A = sc.DiagonalLinOp(ctx.asarray([0.0, 1.0]), X, ctx)
    >>> v = ctx.asarray([2.0, 3.0])
    >>> result = sc.expm_multiply(A, v, t=0.5, max_iter=5)
    >>> np.allclose(result.result, [2.0, 3.0 * np.exp(0.5)], atol=1e-10)
    True
    """
    # Validate once: square, not provably non-Hermitian, valid parameters.
    A = require_linop(A)
    require_square(A, "expm_multiply")
    if A.is_hermitian() is False:
        raise ValueError("expm_multiply requires A to be Hermitian/self-adjoint.")
    max_iter = _check_lanczos_max_iter(max_iter)
    A.domain.check_member(v)

    ops = A.ops
    ctx = A.ctx
    real_dtype = ops.real_dtype(ctx.dtype)
    # Lanczos basis V and tridiagonal T = V* A V.
    basis = _lanczos_basis_and_tridiag(A, v, max_iter, tol, real_dtype, check_every=1)

    # Zero the inactive block of T, so the fixed-size eigh sees only the m-dimensional part.
    m = basis.krylov_dim
    idx = ops.arange(max_iter)
    active_mask = idx < m
    active_matrix_mask = active_mask[:, None] & active_mask[None, :]
    T_safe = ops.where(active_matrix_mask, basis.T, ops.zeros_like(basis.T))

    # exp(t T) e1 from the eigendecomposition of the small T.
    eigvals, eigvecs = ops.eigh(T_safe)
    exp_eigs = ops.exp(t * eigvals)
    expT_e1 = eigvecs @ (exp_eigs * eigvecs[0, :])
    expT_e1 = ops.where(active_mask, expT_e1, ops.zeros_like(expT_e1))

    # exp(t A) v is approximately ||v|| V exp(t T) e1.
    V_reduced = basis.V[:max_iter, :]
    result_flat = basis.initial_norm * ops.einsum("j,jn->n", expT_e1, V_reduced)
    result = A.domain.unflatten(result_flat)

    # Residual estimate beta_m |phi_{m-1}|.
    last_coeff = ops.abs(expT_e1[m - 1])
    residual_estimate = basis.betas[m] * last_coeff
    converged = residual_estimate < basis.tol

    return ExpmMultiplyResult(result, m, residual_estimate, converged)
