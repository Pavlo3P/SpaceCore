from __future__ import annotations

from math import prod
from typing import Any, Callable, Tuple

from ._base import (
    CoordinateSpace,
    EuclideanJordanAlgebraSpace,
    StarSpace,
)
from ._checks import BackendCheck, DTypeCheck, ShapeCheck
from ._inner import EuclideanInnerProduct, InnerProduct
from .._checks import checked_method
from ..types import DenseArray
from ..backend import Context


class DenseCoordinateSpace(CoordinateSpace, StarSpace, EuclideanJordanAlgebraSpace):
    r"""Concrete dense backend arrays with arbitrary finite coordinate shape."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
    ) -> None:
        super().__init__(tuple(shape), ctx)
        self.geometry: InnerProduct = geometry if geometry is not None else EuclideanInnerProduct()
        self._size = prod(self.shape)
        self._is_flat_shape = self.shape == (self._size,)

    def __eq__(self, other: Any) -> bool:
        if type(self) is type(other):
            return (
                super().__eq__(other)
                and type(self.geometry) is type(other.geometry)
                and self.geometry == other.geometry
            )
        return False

    def _local_checks(self):
        """Return membership checks local to dense coordinate spaces."""
        return BackendCheck(), ShapeCheck(), DTypeCheck()

    def zeros(self) -> DenseArray:
        """Return the zero vector in this space."""
        return self.ops.zeros(self.shape, dtype=self.dtype)

    @checked_method(in_space="self", arg_positions=(0, 1))
    def add(self, x: Any, y: Any) -> DenseArray:
        """Return the vector-space sum ``x + y``."""
        return x + y

    def add_batch(self, x: Any, y: Any) -> DenseArray:
        """Return the leading-axis batch sum ``x + y``."""
        return x + y

    @checked_method(in_space="self", arg_positions=(1,))
    def scale(self, a: Any, x: Any) -> DenseArray:
        """Return the scalar product ``a * x``."""
        return a * x

    def scale_batch(self, a: Any, x: Any) -> DenseArray:
        """Return the leading-axis batch scalar product ``a * x``."""
        return a * x

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: Any, y: Any) -> Any:
        r"""Return :math:`\langle x, y\rangle_X` using this space's geometry."""
        return self.geometry.inner(self.ops, x, y)

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
        """Reconstruct a dense-coordinate element from its spectrum."""
        if frame is not None:
            raise ValueError(f"{type(self).__name__}.from_spectrum expects frame=None.")
        self._check_unbatched_member(eigvals)
        return eigvals

    @checked_method(in_space="self")
    def flatten(self, X: DenseArray) -> DenseArray:
        """Return ``X`` as a dense one-dimensional coordinate vector."""
        return X if self._is_flat_shape else X.reshape((-1,))

    def unflatten(self, v: DenseArray) -> DenseArray:
        """Reshape a flat coordinate vector into this space's canonical shape."""
        V = self.ctx.assert_dense(v) if self._enable_checks else v
        return V if self._is_flat_shape else V.reshape(self.shape)

    def flatten_batch(self, xs: DenseArray) -> DenseArray:
        """Flatten a leading-axis batch of dense elements to ``(N, size)``."""
        xs = self.ctx.assert_dense(xs) if self._enable_checks else xs
        return xs if self._is_flat_shape else xs.reshape((xs.shape[0], -1))

    def unflatten_batch(self, vs: DenseArray) -> DenseArray:
        """Unflatten rows of shape ``(N, size)`` into dense space elements."""
        vs = self.ctx.assert_dense(vs) if self._enable_checks else vs
        return vs if self._is_flat_shape else vs.reshape((vs.shape[0],) + self.shape)

    def _convert(self, new_ctx: Context) -> DenseCoordinateSpace:
        """Convert this dense coordinate space to ``new_ctx`` without changing shape."""
        return type(self)(self.shape, new_ctx, geometry=self.geometry.convert(new_ctx))

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

    @checked_method(in_space="self", out_space="self")
    def apply(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        """Backward-compatible alias for coordinatewise spectral application."""
        return self.spectral_apply(x, f)


class DenseVectorSpace(DenseCoordinateSpace):
    r"""Concrete one-dimensional dense vectors with configurable geometry."""

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
        """Convert this dense vector space to ``new_ctx`` without changing shape."""
        return DenseVectorSpace(self.shape, new_ctx, geometry=self.geometry.convert(new_ctx))
