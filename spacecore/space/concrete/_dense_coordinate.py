from __future__ import annotations

from math import prod
from typing import Any, Tuple

from ..base import CoordinateSpace, EuclideanInnerProduct, InnerProduct, InnerProductSpace
from ..checks import BackendCheck, DTypeCheck, FieldCheck, ShapeCheck, SpaceCheck
from ..._checks import checked_method
from ...backend import Context
from ...types import DenseArray


class DenseCoordinateSpace(CoordinateSpace, InnerProductSpace):
    r"""
    Concrete dense backend arrays with arbitrary finite coordinate shape.

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

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
    ) -> None:
        super().__init__(tuple(shape), ctx)
        self.geometry: InnerProduct = geometry if geometry is not None else EuclideanInnerProduct()
        self.geometry.validate_for(self)
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

    def _space_descriptor(self) -> str:
        """Append a ``[weighted]`` marker for non-Euclidean geometry.

        Folding the marker into the descriptor (rather than the repr body) keeps
        the geometry visible even when the space is nested inside a stacked or
        tree space's descriptor.
        """
        base = super()._space_descriptor()
        return base if self.is_euclidean else f"{base}[weighted]"

    def _local_checks(self) -> tuple[SpaceCheck, ...]:
        """Return membership checks local to dense coordinate spaces."""
        return BackendCheck(), ShapeCheck(), FieldCheck(), DTypeCheck()

    def zeros(self) -> DenseArray:
        """Return the zero vector in this space."""
        return self.ops.zeros(self.shape, dtype=self.dtype)

    @checked_method(in_space="self", arg_positions=(0, 1))
    def add(self, x: Any, y: Any) -> DenseArray:
        """Return the vector-space sum ``x + y``."""
        return self._add_core(x, y)

    def _add_core(self, x: Any, y: Any) -> DenseArray:
        """Return ``x + y`` without membership checks."""
        result = x + y
        return self.ctx.asarray(result) if self.shape == () else result

    def add_batch(self, x: Any, y: Any) -> DenseArray:
        """Return the leading-axis batch sum ``x + y``."""
        return x + y

    @checked_method(in_space="self", arg_positions=(1,))
    def scale(self, a: Any, x: Any) -> DenseArray:
        """Return the scalar product ``a * x``."""
        return self._scale_core(a, x)

    def _scale_core(self, a: Any, x: Any) -> DenseArray:
        """Return ``a * x`` without membership checks."""
        result = a * x
        return self.ctx.asarray(result) if self.shape == () else result

    def scale_batch(self, a: Any, x: Any) -> DenseArray:
        """Return the leading-axis batch scalar product ``a * x``."""
        return a * x

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: Any, y: Any) -> Any:
        r"""Return :math:`\langle x, y\rangle_X` using this space's geometry."""
        return self._inner_core(x, y)

    def _inner_core(self, x: Any, y: Any) -> Any:
        """Return the configured inner product without membership checks."""
        return self.geometry.inner(self.ops, x, y)

    @checked_method(in_space="self")
    def flatten(self, X: DenseArray) -> DenseArray:
        """Return ``X`` as a dense one-dimensional coordinate vector."""
        return X if self._is_flat_shape else X.reshape((-1,))

    def unflatten(self, v: DenseArray) -> DenseArray:
        """Reshape a flat coordinate vector into this space's canonical shape."""
        V = self._coerce_dense(v)
        return V if self._is_flat_shape else V.reshape(self.shape)

    def flatten_batch(self, xs: DenseArray) -> DenseArray:
        """Flatten a leading-axis batch of dense elements to ``(N, size)``."""
        xs = self._coerce_dense(xs)
        return xs if self._is_flat_shape else xs.reshape((xs.shape[0], -1))

    def unflatten_batch(self, vs: DenseArray) -> DenseArray:
        """Unflatten rows of shape ``(N, size)`` into dense space elements."""
        vs = self._coerce_dense(vs)
        return vs if self._is_flat_shape else vs.reshape((vs.shape[0],) + self.shape)

    def _check_unbatched_member(self, x: DenseArray) -> None:
        """Run member checks for a single element while allowing batched input.

        Validates ``x`` only when its shape matches a single element, so callers
        that also accept leading batches (e.g. spectral operations) skip the
        per-element check for batched input rather than failing it.
        """
        if self.check_level != "none" and tuple(getattr(x, "shape", ())) == self.shape:
            self._check_member(x)

    def _convert(self, new_ctx: Context) -> DenseCoordinateSpace:
        """Convert this dense coordinate space to ``new_ctx`` without changing shape."""
        return DenseCoordinateSpace(self.shape, new_ctx, geometry=self.geometry.convert(new_ctx))
