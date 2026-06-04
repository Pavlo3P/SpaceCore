from __future__ import annotations

from typing import Any, Callable, Tuple

from ..base import EuclideanInnerProduct, EuclideanJordanAlgebraSpace, InnerProduct, StarSpace
from ..._checks import checked_method
from ...backend import Context
from ...types import DenseArray
from ._dense_coordinate import DenseCoordinateSpace


def _require_elementwise_jordan_geometry(geometry: InnerProduct | None) -> InnerProduct | None:
    if geometry is None:
        return None
    if type(geometry) is EuclideanInnerProduct:
        return geometry
    raise TypeError(
        "ElementwiseJordanSpace requires EuclideanInnerProduct; "
        f"got {type(geometry).__name__}. Use DenseCoordinateSpace for generic or weighted coordinates."
    )


class ElementwiseJordanSpace(DenseCoordinateSpace, StarSpace, EuclideanJordanAlgebraSpace):
    """Dense coordinate space with elementwise star and Jordan product."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
    ) -> None:
        super().__init__(shape, ctx, geometry=_require_elementwise_jordan_geometry(geometry))

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

    def from_spectrum(self, eigvals: DenseArray, frame: Any) -> DenseArray:
        """Reconstruct an elementwise Jordan element from its spectrum."""
        if frame is not None:
            raise ValueError(f"{type(self).__name__}.from_spectrum expects frame=None.")
        self._check_unbatched_member(eigvals)
        return eigvals

    @checked_method(in_space="self", out_space="self")
    def spectral_apply(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        """Apply a scalar function coordinatewise."""
        return self._apply_entrywise(x, f)

    @checked_method(in_space="self", out_space="self")
    def apply(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        """Backward-compatible alias for elementwise spectral application."""
        return self.spectral_apply(x, f)

    def _convert(self, new_ctx: Context) -> ElementwiseJordanSpace:
        return type(self)(self.shape, new_ctx, geometry=self.geometry.convert(new_ctx))


class DenseVectorSpace(ElementwiseJordanSpace):
    r"""Concrete one-dimensional dense vectors with Euclidean elementwise Jordan structure."""

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

    def _convert(self, new_ctx: Context) -> DenseVectorSpace:
        return DenseVectorSpace(self.shape, new_ctx, geometry=self.geometry.convert(new_ctx))
