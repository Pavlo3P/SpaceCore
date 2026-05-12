from __future__ import annotations

from typing import Any, Tuple, Callable

from ._checks import HermitianCheck, SquareMatrixCheck
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

    def _local_checks(self):
        return (
            SquareMatrixCheck(),
            HermitianCheck(
                atol=self.atol,
                rtol=self.rtol,
                enforce=self.enforce_herm,
            ),
        )

    def is_hermitian(self, x: DenseArray) -> bool:
        return HermitianCheck(
            atol=self.atol,
            rtol=self.rtol,
            enforce=self.enforce_herm,
        ).is_valid(self, x)

    def symmetrize(self, x: DenseArray) -> DenseArray:
        """Project onto the Hermitian cone: (X + X^H)/2."""
        return (x + x.T.conj()) * 0.5

    def eigh(self, x: DenseArray, k: int = None) -> Tuple[DenseArray, DenseArray]:
        self.check_member(x)
        return self.ops.eigh(x)

    def unflatten(self, v: DenseArray) -> DenseArray:
        vv = self.ctx.assert_dense(v) if self._enable_checks else v
        X = vv.reshape(self.shape)
        return self.symmetrize(X)

    def psd_proj(self, x: DenseArray) -> DenseArray:
        self.check_member(x)
        evals, evecs = self.ops.eigh(x)
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

    def apply(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        """
        Apply a scalar function to a Hermitian matrix via spectral calculus.

        For a Hermitian matrix
        $$
        X \in \mathbb{H}^n,
        $$
        with eigendecomposition
        $$
        X = U \operatorname{diag}(\lambda) U^*,
        $$
        this method returns
        $$
        f(X) = U \operatorname{diag}(f(\lambda)) U^*,
        $$
        where ``f`` is applied entrywise to the eigenvalue vector
        $$
        \lambda \in \mathbb{R}^n.
        $$

        Parameters
        ----------
        x:
            Hermitian matrix in this space. Must have shape ``(n, n)`` and
            satisfy the Hermitian membership conditions of the space.
        f:
            Callable applied to the eigenvalues of ``x``. It should accept a
            dense backend array of eigenvalues and return an array of the same
            shape.

        Returns
        -------
        DenseArray
            The Hermitian matrix obtained by spectral application of ``f`` to
            ``x``.

        Raises
        ------
        TypeError
            If ``x`` is not a valid Hermitian element of this space.

        Notes
        -----
        This is not an entrywise matrix transformation. The function is applied
        to the spectrum of ``x``, not to its matrix entries.

        In particular, if
        $$
        X = U \operatorname{diag}(\lambda) U^*,
        $$
        then the eigenvectors are preserved and only the eigenvalues are
        transformed.
        """
        self.check_member(x)
        evals, evecs = self.ops.eigh(x)
        fevals = self._apply_entrywise(evals, f)

        return self.eig_to_dense(fevals, evecs)
