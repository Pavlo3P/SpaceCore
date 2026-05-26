from __future__ import annotations

from typing import Any, NamedTuple

from ..linop import LinOp
from ._lanczos import _check_lanczos_max_iter, _lanczos_basis_and_tridiag
from ._utils import require_linop, require_square, result_repr


class ExpmMultiplyResult(NamedTuple):
    """Result returned by :func:`expm_multiply`.

    Attributes
    ----------
    result:
        Vector in the domain of the input operator approximating
        ``exp(t * A) @ v``.
    krylov_dim:
        Actual Krylov dimension reached before breakdown or ``max_iter``.
    residual_estimate:
        Projected exponential residual estimate
        ``abs(beta[m] * phi[m - 1])``.
    converged:
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
    """
    Compute ``exp(t * A) @ v`` for a Hermitian operator via Krylov projection.

    Parameters
    ----------
    A:
        Square Hermitian linear operator.
    v:
        Initial vector in ``A.domain``.
    t:
        Scalar time/scale multiplying ``A``. Complex values are supported for
        complex-valued contexts, for example Schrödinger evolution.
    max_iter:
        Maximum Krylov dimension. Values around 20-50 are usually sufficient
        when ``abs(t) * ||A||`` is moderate.
    tol:
        Breakdown tolerance for Lanczos and threshold for the projected
        exponential residual estimate.

    Returns
    -------
    ExpmMultiplyResult
        Result vector in ``A.domain``, the Krylov dimension used, the standard
        estimate ``abs(beta[m] * phi[m - 1])``, and a convergence flag.
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
