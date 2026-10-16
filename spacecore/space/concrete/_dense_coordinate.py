from __future__ import annotations

from math import prod
from typing import Any, Callable, Tuple

from ..base import CoordinateSpace, EuclideanInnerProduct, InnerProduct, InnerProductSpace
from ..checks import BackendCheck, DTypeCheck, FieldCheck, ShapeCheck, SpaceCheck
from ..._checks import CheckLevel
from ...contextual import Context
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
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    _fields = ("shape", "ctx", "geometry")

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        super().__init__(tuple(shape), ctx, check_level=check_level)
        # Euclidean geometry unless one is given; validate it against this space.
        self.geometry: InnerProduct = geometry if geometry is not None else EuclideanInnerProduct()
        self.geometry.validate_for(self)
        # Cache the flat size and whether the shape is already flat.
        self._size = prod(self.shape)
        self._is_flat_shape = self.shape == (self._size,)

    def _eq_algebra(self, other: Any) -> bool:
        # Tier 2: geometry kind (structural, also shields the geometry's own
        # allclose). Tier 3: geometry value comparison (weights for weighted).
        # Equal spaces also need equal geometry.
        return (
            super()._eq_algebra(other)
            and type(self.geometry) is type(other.geometry)
            and self.geometry == other.geometry
        )

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

    def _add_core(self, x: Any, y: Any) -> DenseArray:
        """Return ``x + y`` without membership checks."""
        result = x + y
        return self.ctx.asarray(result) if self.shape == () else result

    def _add_batch_core(self, x: Any, y: Any) -> DenseArray:
        """Return the leading-axis batch sum ``x + y``."""
        return x + y

    def _scale_core(self, a: Any, x: Any) -> DenseArray:
        """Return ``a * x`` without membership checks."""
        result = a * x
        return self.ctx.asarray(result) if self.shape == () else result

    def _scale_batch_core(self, a: Any, x: Any) -> DenseArray:
        """Return the leading-axis batch scalar product ``a * x``."""
        return a * x

    def _flatten_core(self, X: DenseArray) -> DenseArray:
        """Return ``X`` as a dense one-dimensional coordinate vector."""
        return X if self._is_flat_shape else X.reshape((-1,))

    def _unflatten_core(self, v: DenseArray) -> DenseArray:
        """Reshape a flat coordinate vector into this space's canonical shape."""
        V = self._coerce_dense(v)
        return V if self._is_flat_shape else V.reshape(self.shape)

    def _flatten_batch_core(self, xs: DenseArray) -> DenseArray:
        """Flatten a leading-axis batch of dense elements to ``(N, size)``."""
        xs = self._coerce_dense(xs)
        return xs if self._is_flat_shape else xs.reshape((xs.shape[0], -1))

    def _unflatten_batch_core(self, vs: DenseArray) -> DenseArray:
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

    def _apply_entrywise(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        """Apply ``f`` entrywise and verify that shape is preserved."""
        # Apply f to the whole array, else elementwise; the shape must not change.
        try:
            y = f(x)
        except Exception:
            y = self.ops.vectorize(f)(x)
        if self._checks_at_least("cheap") and y.shape != x.shape:
            raise ValueError("Function application changed shape.")
        return y
