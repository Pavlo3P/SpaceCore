from __future__ import annotations

from typing import Any, Tuple

from ._vector import VectorSpace
from ..types import DenseArray
from ..backend import Context


class HermitianSpace(VectorSpace):
    """
    Space of dense n×n Hermitian matrices.

    - Elements are backend-native dense arrays, shape (n, n).
    - Membership enforces Hermitian structure up to tolerances.
    - Inner product is Frobenius / Hilbert–Schmidt:
        ⟨X, Y⟩ = vdot(vec(X), vec(Y)),
      where vdot conjugates the first argument (backend-defined).
    """

    def __init__(self,
                 n: int,
                 atol: float = 0.,
                 rtol: float = 0.,
                 enforce_herm: bool = True,
                 ctx: Context | str | None = None,
                 ):
        if n <= 0:
            raise ValueError("n must be positive.")

        shape = (n, n)
        super(HermitianSpace, self).__init__(shape, ctx)

        self.atol = atol
        self.rtol = rtol
        self.enforce_herm = enforce_herm

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, HermitianSpace):
           return (super(HermitianSpace, self).__eq__(other)
                   and self.atol == other.atol
                   and self.rtol == other.rtol
                   and self.enforce_herm == other.enforce_herm)
        return False

    @property
    def n(self) -> int:
        return self.shape[0]

    def _check_member(self, x: Any) -> None:
        super()._check_member(x)

        if not self.is_hermitian(x):
            raise TypeError("Matrix is not Hermitian (within the specified tolerances).")

    def is_hermitian(self, X: DenseArray) -> bool:
        ops = self.ctx.ops
        Xh = ops.conj(X).T
        diff = X - Xh

        # Validation is typically done outside jit, so it is OK to reduce via
        # backend's .max when available (NumPy/JAX arrays have it).
        adiff = ops.abs(diff)
        aX = ops.abs(X)

        # max(abs(diff)) <= atol + rtol*max(abs(X))
        max_adiff = adiff.max()
        max_aX = aX.max()

        # Convert 0-d arrays to Python scalars if needed
        def _as_float(v):
            try:
                return float(v)
            except TypeError:
                return float(v.item())

        max_adiff_f = _as_float(max_adiff)
        max_aX_f = _as_float(max_aX)

        thresh = float(self.atol) + float(self.rtol) * max_aX_f
        return max_adiff_f <= thresh

    def symmetrize(self, X: DenseArray) -> DenseArray:
        """Project onto the Hermitian cone: (X + X^H)/2."""
        return (X + X.T.conj()) * 0.5

    def eigh(self, X: DenseArray, k: int = None) -> Tuple[DenseArray, DenseArray]:
        self.check_member(X)
        return self.ops.eigh(X)

    def unflatten(self, v: DenseArray) -> DenseArray:
        vv = self.ctx.assert_dense(v)
        X = self.ops.reshape(vv, self.shape)
        return self.symmetrize(X)

    def psd_proj(self, X: DenseArray) -> DenseArray:
        self.check_member(X)
        evals, evecs = self.ops.eigh(X)
        evals = self.ops.maximum(evals, 0.)
        return self.eig_to_dense(evals, evecs)

    def eig_to_dense(self, evals: DenseArray, evecs: DenseArray) -> DenseArray:
        self.ctx.assert_dense(evals)
        self.ctx.assert_dense(evecs)
        X = (evecs * evals) @ evecs.T.conj()
        self.check_member(X)
        return X

    def _convert(self, new_ctx: Context) -> HermitianSpace:
        return HermitianSpace(self.n, self.atol, self.rtol, self.enforce_herm, new_ctx)
