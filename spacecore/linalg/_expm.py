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

    Require ``A`` to be square in the SpaceCore sense
    (``A.domain == A.codomain``) and Hermitian with respect to
    ``A.domain.inner``. The method builds a Lanczos basis and applies the
    exponential of the small tridiagonal projection, avoiding dense
    materialization of ``A``.

    Parameters
    ----------
    A : LinOp
        Linear operator that must be Hermitian/self-adjoint with respect to
        ``A.domain.inner``. ``A.domain`` must equal ``A.codomain``, including
        the underlying space type and inner-product geometry. Operators with
        structurally unknown Hermiticity (``A.is_hermitian()`` returns
        ``None``) are accepted on trust; the caller is responsible for ensuring
        Hermiticity. Non-Hermitian inputs produce undefined results.
    v : array-like
        Initial vector in ``A.domain``.
    t : float or complex, optional
        Scalar multiplier on ``A``. Complex values require a complex-valued
        ``ctx.dtype`` such as ``complex64`` or ``complex128``. Using a complex
        ``t`` with a real-valued context produces backend-dependent results.
        Default is 1.0.
    max_iter : int, optional
        Maximum Krylov dimension. Values around 20-50 are usually sufficient
        when :math:`|t|\|A\|` is moderate. Must be a Python ``int`` rather
        than a traced JAX scalar; under ``jax.jit`` it is treated as a static
        argument and changing it triggers retracing. Default is 30.
    tol : float, optional
        Tolerance used both for Lanczos breakdown and for the convergence flag:
        ``result.converged`` is ``True`` when the projected exponential
        residual estimate is below ``tol``. Default is 1e-10.

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

    Hermiticity is enforced only when it can be structurally verified: known
    non-Hermitian operators raise ``ValueError``. Operators with unknown
    structure, such as many matrix-free operators and operators on custom
    spaces, are trusted.

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
    >>> result = sc.expm_multiply(A, v, t=0.5, max_iter=5)
    >>> np.allclose(result.result, [2.0, 3.0 * np.exp(0.5)], atol=1e-10)
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

    m = basis.krylov_dim
    idx = ops.arange(max_iter)
    active_mask = idx < m
    active_matrix_mask = active_mask[:, None] & active_mask[None, :]
    T_safe = ops.where(active_matrix_mask, basis.T, ops.zeros_like(basis.T))

    eigvals, eigvecs = ops.eigh(T_safe)
    exp_eigs = ops.exp(t * eigvals)
    expT_e1 = eigvecs @ (exp_eigs * eigvecs[0, :])
    expT_e1 = ops.where(active_mask, expT_e1, ops.zeros_like(expT_e1))

    V_reduced = basis.V[:max_iter, :]
    result_flat = basis.initial_norm * ops.einsum("j,jn->n", expT_e1, V_reduced)
    result = A.domain.unflatten(result_flat)

    last_coeff = ops.abs(expT_e1[m - 1])
    residual_estimate = basis.betas[m] * last_coeff
    converged = residual_estimate < basis.tol

    return ExpmMultiplyResult(result, m, residual_estimate, converged)
