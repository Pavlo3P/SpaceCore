from __future__ import annotations

from typing import Any, Callable, Tuple

from ..base import (
    EuclideanInnerProduct,
    EuclideanJordanAlgebraSpace,
    InnerProduct,
    JordanAlgebraSpace,
    StarSpace,
)
from ..._checks import checked_method
from ..._contextual import normalize_context
from ...backend import Context
from ...types import DenseArray
from ._dense_coordinate import DenseCoordinateSpace


def _is_real_euclidean(ctx: Context, geometry: InnerProduct | None) -> bool:
    """Return whether elementwise coordinates have Euclidean-Jordan geometry."""
    geometry = EuclideanInnerProduct() if geometry is None else geometry
    return not ctx.ops.is_complex_dtype(ctx.dtype) and type(geometry) is EuclideanInnerProduct


class DenseVectorSpace(DenseCoordinateSpace, StarSpace):
    """
    Plain one-dimensional dense vectors with no Jordan capability by default.

    Parameters
    ----------
    shape : tuple of int
        One-dimensional dense vector shape.
    ctx : Context, str, or None, optional
        Context specification used for dense arrays.
    geometry : InnerProduct or None, optional
        Inner-product geometry. If omitted, Euclidean coordinate geometry is
        used.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
    ) -> None:
        shape = tuple(shape)
        if len(shape) != 1:
            raise ValueError(f"DenseVectorSpace requires one-dimensional shape, got {shape}.")
        super().__init__(shape, ctx, geometry=geometry)

    @checked_method(in_space="self")
    def star(self, x: DenseArray) -> DenseArray:
        """Return elementwise conjugation."""
        return self.ops.conj(x)

    def _convert(self, new_ctx: Context) -> DenseVectorSpace:
        return DenseVectorSpace(self.shape, new_ctx, geometry=self.geometry.convert(new_ctx))


class ElementwiseJordanSpace(JordanAlgebraSpace, DenseCoordinateSpace, StarSpace):
    """
    Elementwise Jordan algebra for real or complex dense coordinates.

    Parameters
    ----------
    shape : tuple of int
        Canonical dense array shape for one element.
    ctx : Context, str, or None, optional
        Context specification used for dense arrays.
    geometry : InnerProduct or None, optional
        Inner-product geometry. If omitted, Euclidean coordinate geometry is
        used.
    """

    def __new__(
        cls,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
    ):
        if cls is ElementwiseJordanSpace:
            resolved_ctx = normalize_context(ctx)
            if _is_real_euclidean(resolved_ctx, geometry):
                cls = EuclideanElementwiseJordanSpace
        return super(ElementwiseJordanSpace, cls).__new__(cls)

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
    ) -> None:
        DenseCoordinateSpace.__init__(self, shape, ctx, geometry=geometry)

    @checked_method(in_space="self")
    def star(self, x: DenseArray) -> DenseArray:
        """Return elementwise conjugation."""
        return self.ops.conj(x)

    @checked_method(in_space="self", arg_positions=(0, 1))
    def jordan(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """Return the elementwise Jordan product."""
        return x * y

    def _check_unbatched_member(self, x: DenseArray) -> None:
        """Run member checks for one element, while allowing leading batches."""
        if self._enable_checks and tuple(getattr(x, "shape", ())) == self.shape:
            self._check_member(x)

    def spectrum(self, x: DenseArray) -> DenseArray:
        """Return ``x`` as its Jordan spectrum under elementwise product."""
        self._check_unbatched_member(x)
        return x

    def spectral_decompose(self, x: DenseArray) -> tuple[DenseArray, None]:
        """Return the trivial spectral decomposition ``(x, None)``."""
        self._check_unbatched_member(x)
        return x, None

    def from_spectrum(self, eigvals: DenseArray, frame: Any = None) -> DenseArray:
        """Reconstruct an elementwise Jordan element from its spectrum."""
        if frame is not None:
            raise ValueError(f"{type(self).__name__}.from_spectrum expects frame=None.")
        self._check_unbatched_member(eigvals)
        return eigvals

    def _apply_entrywise(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        """Apply ``f`` entrywise and verify that shape is preserved."""
        try:
            y = f(x)
        except Exception:
            y = self.ops.vectorize(f)(x)
        if self._enable_checks and y.shape != x.shape:
            raise ValueError("Function application changed shape.")
        return y

    @checked_method(in_space="self", out_space="self")
    def spectral_apply(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        """Apply a scalar function coordinatewise."""
        return self._apply_entrywise(x, f)

    def _convert(self, new_ctx: Context) -> ElementwiseJordanSpace:
        return ElementwiseJordanSpace(self.shape, new_ctx, geometry=self.geometry.convert(new_ctx))


class EuclideanElementwiseJordanSpace(ElementwiseJordanSpace, EuclideanJordanAlgebraSpace):
    """
    Real elementwise Euclidean Jordan algebra.

    Parameters
    ----------
    shape : tuple of int
        Canonical dense array shape for one element.
    ctx : Context, str, or None, optional
        Context specification used for dense arrays.
    geometry : InnerProduct or None, optional
        Inner-product geometry. This class is selected only for real contexts
        with Euclidean coordinate geometry.
    """
