from __future__ import annotations

from typing import Any, Tuple, Callable, cast

from ..checks import HermitianCheck, SquareMatrixCheck
from ..base import EuclideanJordanAlgebraSpace, StarSpace
from ._dense_coordinate import DenseCoordinateSpace
from ..._checks import checked_method
from ...types import DenseArray
from ...backend import Context


class HermitianSpace(DenseCoordinateSpace, StarSpace, EuclideanJordanAlgebraSpace):
    r"""
    Represent dense Hermitian matrices with Frobenius geometry.

    Elements are backend-native dense arrays with shape ``(n, n)``.
    Membership enforces Hermitian structure up to tolerances.

    The inner product is Frobenius / Hilbert-Schmidt:
    ``<X, Y> = vdot(vec(X), vec(Y))``, where ``vdot`` conjugates the
    first argument according to backend rules.

    ``HermitianSpace`` currently uses Euclidean/Frobenius geometry in flattened
    coordinates and does not expose custom geometry injection. Metric-aware
    Hermitian geometries should be introduced as a separate class or explicit
    extension.

    Parameters
    ----------
    n : int
        Matrix dimension.
    atol : float, optional
        Absolute tolerance for Hermitian membership checks.
    rtol : float, optional
        Relative tolerance for Hermitian membership checks.
    enforce_herm : bool, optional
        Whether membership checks enforce Hermitian structure.
    ctx : Context, str, or None, optional
        Backend context specification.

    Attributes
    ----------
    n : int
        Matrix dimension.
    """

    def __init__(
        self,
        n: int,
        atol: float = 0.0,
        rtol: float = 0.0,
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
        return HermitianCheck(
            atol=self.atol,
            rtol=self.rtol,
            enforce=self.enforce_herm,
        ).is_valid(self, x)

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

    def unflatten(self, v: DenseArray) -> DenseArray:
        """Reshape dense coordinates and symmetrize the result."""
        vv = self._coerce_dense(v)
        X = vv.reshape(self.shape)
        return self.symmetrize(X)

    @checked_method(in_space="self")
    def psd_proj(self, x: DenseArray) -> DenseArray:
        """Project a Hermitian element onto the positive semidefinite cone."""
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
        self.ctx.assert_dense(evals)
        self.ctx.assert_dense(evecs)
        X = self.ops.einsum("...ij,...j,...kj->...ik", evecs, evals, self.ops.conj(evecs))
        X = self.symmetrize(X)
        self._check_unbatched_member(X)
        return X

    def _convert(self, new_ctx: Context) -> HermitianSpace:
        """Convert this Hermitian space to ``new_ctx``."""
        return HermitianSpace(self.n, self.atol, self.rtol, self.enforce_herm, new_ctx)

    def _apply_entrywise(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        """Apply ``f`` entrywise and verify that shape is preserved."""
        try:
            y = f(x)
        except Exception:
            y = self.ops.vectorize(f)(x)
        if self._checks_at_least("cheap") and y.shape != x.shape:
            raise ValueError("Function application changed shape.")
        return y

    @checked_method(in_space="self")
    def spectral_apply(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        r"""
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
        evals, evecs = self.spectral_decompose(x)
        fevals = self._apply_entrywise(evals, f)

        return self.eig_to_dense(fevals, evecs)
