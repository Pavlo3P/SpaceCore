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
    Compute :math:`\exp(t A) v` by Krylov projection.

    Require ``A`` to be Hermitian or structurally unknown. The method builds a
    Lanczos basis and applies the exponential of the small tridiagonal
    projection, avoiding dense materialization of ``A``.

    Parameters
    ----------
    A : LinOp
        Square Hermitian linear operator.
    v : array-like
        Initial vector in ``A.domain``.
    t : float or complex, optional
        Scalar time/scale multiplying ``A``. Complex values are supported for
        complex-valued contexts, for example Schrodinger evolution. Default is
        1.0.
    max_iter : int, optional
        Maximum Krylov dimension. Values around 20-50 are usually sufficient
        when :math:`|t|\|A\|` is moderate. Default is 30.
    tol : float, optional
        Breakdown tolerance for Lanczos and threshold for the projected
        exponential residual estimate. Default is 1e-10.

    Returns
    -------
    ExpmMultiplyResult
        Result vector in ``A.domain``, the Krylov dimension used, the standard
        estimate ``abs(beta[m] * phi[m - 1])``, and a convergence flag.

    Raises
    ------
    TypeError
        If ``A`` is not a :class:`LinOp`.
    ValueError
        If ``A`` is not square, is known to be non-Hermitian, or if
        ``max_iter`` is invalid.

    See Also
    --------
    lanczos_smallest : Build the related Hermitian Krylov projection.
    power_iteration : Estimate a dominant eigenpair.

    Notes
    -----
    The projected exponential is computed as
    :math:`\exp(t T) e_0` using an eigendecomposition of the small real
    symmetric tridiagonal matrix ``T``. This is JIT-compatible on the JAX
    backend when ``max_iter`` is static.

    The returned residual estimate is
    :math:`|\beta_m \phi_{m-1}|`, where ``phi`` is the projected exponential
    vector. Callers that need the true residual can perform one additional
    operator application.

    Examples
    --------
    Apply the exponential of a diagonal operator.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.VectorSpace((2,), ctx)
    >>> A = sc.DiagonalLinOp(ctx.asarray([0.0, 1.0]), X, ctx)
    >>> v = ctx.asarray([2.0, 3.0])
    >>> result = sc.expm_multiply(A, v, t=0.5, max_iter=2)
    >>> np.allclose(result.result, [2.0, 3.0 * np.exp(0.5)])
    True
    """
    A = require_linop(A)
    require_square(A, "expm_multiply")
    if A.is_hermitian() is False:
        raise ValueError("expm_multiply requires A to be Hermitian/self-adjoint.")
    max_iter = _check_lanczos_max_iter(max_iter)
    A.domain.check_member(v)

    ops = A.ops
    ctx = A.ctx
    real_dtype = ops.real_dtype(ctx.dtype)
    basis = _lanczos_basis_and_tridiag(A, v, max_iter, tol, real_dtype, check_every=1)

    eigvals, eigvecs = ops.eigh(basis.T)
    exp_eigs = ops.exp(t * eigvals)
    expT_e1 = eigvecs @ (exp_eigs * eigvecs[0, :])

    V_reduced = basis.V[:max_iter, :]
    result_flat = basis.initial_norm * ops.einsum("j,jn->n", expT_e1, V_reduced)
    result = A.domain.unflatten(result_flat)

    last_coeff = ops.abs(expT_e1[basis.krylov_dim - 1])
    residual_estimate = basis.betas[basis.krylov_dim] * last_coeff
    converged = residual_estimate < basis.tol

    return ExpmMultiplyResult(result, basis.krylov_dim, residual_estimate, converged)
