from __future__ import annotations

from typing import Any, Tuple, Callable, cast

from ..checks import HermitianCheck, SquareMatrixCheck
from ..base import EuclideanJordanAlgebraSpace, StarSpace
from ._dense_coordinate import DenseCoordinateSpace
from ..._checks import CheckLevel
from ..._checks import checked_method
from ...types import DenseArray
from ...contextual import Context


class HermitianSpace(DenseCoordinateSpace, StarSpace, EuclideanJordanAlgebraSpace):
    r"""
    Dense ``n x n`` Hermitian matrices with the Frobenius inner product.

    A real vector space: scaling by a non-real number is rejected. Membership checks
    Hermitian structure up to ``atol``/``rtol``.
    Horn & Johnson, *Matrix Analysis*, 2nd ed., Thm 4.1.5 (spectral decomposition).

    Parameters
    ----------
    n : int
        Matrix dimension.
    atol : float, optional
        Absolute tolerance of the Hermitian check.
    rtol : float, optional
        Relative tolerance of the Hermitian check.
    enforce_herm : bool, optional
        Whether membership checks Hermitian structure at all.
    ctx : Context, str, or None, optional
        Backend context.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    # Hermitian matrices with complex entries form a *real* vector space of
    # dimension n^2 -- their complex span is all of Mat(n, C). The dtype says
    # "complex" (correctly, about the entries); the scalars are real either way.
    _fields = ("n", "atol", "rtol", "enforce_herm", "ctx")

    declared_scalar_field = "real"

    def __init__(
        self,
        n: int,
        atol: float = 0.0,
        rtol: float = 0.0,
        enforce_herm: bool = True,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ):
        # An n x n coordinate space plus Hermitian-check tolerances.
        if n <= 0:
            raise ValueError("n must be positive.")

        shape = (n, n)
        super(HermitianSpace, self).__init__(shape, ctx, check_level=check_level)

        self.atol = atol
        self.rtol = rtol
        self.enforce_herm = enforce_herm

    # Equality is inherited from DenseCoordinateSpace (backend gate + field +
    # shape (= n) + fixed Frobenius geometry). The membership tolerances
    # ``atol``/``rtol``/``enforce_herm`` are deliberately NOT part of identity:
    # they are validation policy (like ``check_level``), not the mathematical
    # space of n x n Hermitian matrices they describe.

    def _space_descriptor(self) -> str:
        """Return ``Herm(n)``; the real/complex field shows in the dtype tag."""
        return f"Herm({self.n})"

    @property
    def n(self) -> int:
        """Matrix dimension of this Hermitian space."""
        return self.shape[0]

    def _local_checks(self):
        """Return membership checks local to Hermitian spaces."""
        # Membership: square, and Hermitian within the tolerances.
        return (
            SquareMatrixCheck(),
            HermitianCheck(
                atol=self.atol,
                rtol=self.rtol,
                enforce=self.enforce_herm,
            ),
        )

    def is_hermitian(self, x: DenseArray) -> bool:
        """Return whether ``x`` satisfies this space's Hermitian check."""
        return self._local_checks()[1].is_valid(self, x)

    def symmetrize(self, x: DenseArray) -> DenseArray:
        r"""Project ``x`` onto the Hermitian subspace as :math:`(X + X^*) / 2`."""
        x_adj = self.ops.conj(self.ops.swapaxes(x, -1, -2))
        return (x + x_adj) * 0.5

    @checked_method(in_space="self")
    def star(self, x: DenseArray) -> DenseArray:
        """Return the canonical star operation for Hermitian elements: identity."""
        return x

    @checked_method(in_space="self", arg_positions=(0, 1))
    def jordan(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """Return the Hermitian Jordan product ``(xy + yx) / 2``."""
        xy = self.ops.matmul(x, y)
        yx = self.ops.matmul(y, x)
        return self.symmetrize((xy + yx) * 0.5)

    def spectrum(self, x: DenseArray) -> DenseArray:
        """Return the Hermitian eigenvalue spectrum of ``x``."""
        self._check_unbatched_member(x)
        return self.ops.eigh(x)[0]

    def spectral_decompose(self, x: DenseArray) -> Tuple[DenseArray, DenseArray]:
        """Return the Hermitian eigendecomposition ``(evals, evecs)``."""
        self._check_unbatched_member(x)
        return self.ops.eigh(x)

    def from_spectrum(self, eigvals: DenseArray, frame: DenseArray) -> DenseArray:
        """Reconstruct a Hermitian element from eigenvalues and eigenvectors."""
        return self.eig_to_dense(eigvals, frame)

    def trace(self, x: DenseArray) -> DenseArray:
        """Return the (real) trace as the diagonal sum, avoiding an eigendecomposition.

        ``'...ii->...'`` sums the trailing-two-axes diagonal and preserves leading
        batch axes; ``ops.trace``/``ops.diagonal`` default to the *first* two axes
        and are not batch-safe for ``(..., n, n)`` input.
        """
        self._check_unbatched_member(x)
        return self.ops.real(self.ops.einsum("...ii->...", x))

    def unit(self) -> DenseArray:
        """Return the Jordan identity: the ``n x n`` identity in the space dtype."""
        return self.ops.eye(self.n, dtype=self.dtype)

    def _unflatten_core(self, v: DenseArray) -> DenseArray:
        """Reshape dense coordinates and symmetrize the result."""
        vv = self._coerce_dense(v)
        X = vv.reshape(self.shape)
        return self.symmetrize(X)

    @checked_method(in_space="self")
    def psd_proj(self, x: DenseArray) -> DenseArray:
        """Project onto the PSD cone in Frobenius norm by clipping negative eigenvalues.

        Horn & Johnson, *Matrix Analysis*, 2nd ed., Thm 4.1.5 and Eq. (5.2.7):
        in an eigenbasis, off-diagonal entries add squared error and the optimal
        nonnegative diagonal entries are ``max(lambda_i, 0)``.
        """
        evals, evecs = self.spectral_decompose(x)
        evals = self.ops.maximum(evals, cast(Any, 0.0))
        return self.eig_to_dense(evals, evecs)

    def eig_to_dense(self, evals: DenseArray, evecs: DenseArray) -> DenseArray:
        """Reconstruct a Hermitian matrix from eigenvalues and eigenvectors.

        The ``U diag(evals) U^*`` reconstruction is mathematically Hermitian but
        accumulates floating-point skew at the level of a few ULP, which a
        zero-tolerance Hermitian space would otherwise reject. Symmetrizing
        projects onto the Hermitian part (a no-op up to that skew) so the
        reconstruction is a valid member of this space.
        """
        # X = U diag(evals) U*, symmetrized to remove rounding skew.
        self.ctx.assert_dense(evals)
        self.ctx.assert_dense(evecs)
        X = self.ops.einsum("...ij,...j,...kj->...ik", evecs, evals, self.ops.conj(evecs))
        X = self.symmetrize(X)
        self._check_unbatched_member(X)
        return X

    @checked_method(in_space="self")
    def spectral_apply(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        r"""
        Return :math:`f(X) = U \operatorname{diag}(f(\lambda)) U^*`, with ``f`` applied to the eigenvalues.

        Eigenvectors are kept: Higham, *Functions of Matrices*, Sec. 4.5.

        Parameters
        ----------
        x : DenseArray
            Hermitian matrix in this space.
        f : callable
            Real-valued scalar function applied elementwise to the spectrum,
            returning an array of the same shape; equal eigenvalues need equal images.

        Returns
        -------
        DenseArray
            The Hermitian matrix :math:`f(X)`.
        """
        # Transform the eigenvalues, keep the eigenvectors.
        evals, evecs = self.spectral_decompose(x)
        fevals = self._apply_entrywise(evals, f)

        return self.eig_to_dense(fevals, evecs)
