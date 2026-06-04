from __future__ import annotations

from typing import Any

from ..base import (
    CoordinateSpace,
    EuclideanJordanAlgebraSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
    Space,
    StarSpace,
)
from ._product import ProductSpace
from ...backend import Context, jax_pytree_class
from ..checks import BackendCheck, DTypeCheck, ShapeCheck
from ..._checks import checked_method
from ..._contextual import resolve_context_priority
from ...types import DenseArray

_STACKED_FALLBACK_ERRORS = (TypeError, ValueError, AttributeError, IndexError)


@jax_pytree_class
class StackedSpace(CoordinateSpace):
    """N independent copies of a base space, stacked on a leading axis.

    Elements have shape ``(count,) + base.shape``. Operations act independently
    per copy where appropriate, and the inner product sums over the stack axis,
    so the stack is itself one Hilbert-space element.

    ``StackedSpace`` wraps leaf spaces only. For products, use
    ``ProductSpace(...).stacked(count)``, which returns a product of stacked
    component spaces.
    """


    def __new__(cls, base: Space, count: int, ctx: Context | str | None = None):
        if cls is StackedSpace:
            if isinstance(base, EuclideanJordanAlgebraSpace) and isinstance(base, StarSpace):
                cls = StackedEuclideanJordanAlgebraSpace
            elif isinstance(base, JordanAlgebraSpace):
                cls = StackedJordanAlgebraSpace
            elif isinstance(base, StarSpace):
                cls = StackedStarSpace
            elif isinstance(base, InnerProductSpace):
                cls = StackedInnerProductSpace
        return super(StackedSpace, cls).__new__(cls)

    def __init__(self, base: Space, count: int, ctx: Context | str | None = None) -> None:
        if isinstance(base, ProductSpace):
            raise TypeError(
                "StackedSpace cannot wrap ProductSpace directly; use "
                "ProductSpace(...).stacked(count), which stacks each component."
            )
        if count <= 0:
            raise ValueError("StackedSpace count must be positive.")
        ctx = resolve_context_priority(ctx, base)
        self.base = base.convert(ctx)
        self.count = int(count)
        super().__init__((self.count,) + tuple(self.base.shape), ctx)
        if hasattr(self.base, "geometry"):
            self.geometry = self.base.geometry

    def __eq__(self, other: Any) -> bool:
        """Return whether another stacked space has the same base and count."""
        if type(other) is type(self):
            return self.ctx == other.ctx and self.count == other.count and self.base == other.base
        return False


    def _local_checks(self):
        """Return membership checks local to stacked dense coordinate spaces."""
        return BackendCheck(), ShapeCheck(), DTypeCheck()

    def zeros(self) -> DenseArray:
        """Return the stacked zero element."""
        return self.ops.zeros(self.shape, dtype=self.dtype)

    def ones(self) -> DenseArray:
        """Return the stacked all-ones element."""
        return self.ops.ones(self.shape, dtype=self.dtype)

    @checked_method(in_space="self", arg_positions=(0, 1))
    def add(self, x: Any, y: Any) -> DenseArray:
        """Return the stacked sum ``x + y``."""
        return x + y

    def add_batch(self, x: Any, y: Any) -> DenseArray:
        """Return the leading-axis batch sum of stacked elements."""
        return x + y

    @checked_method(in_space="self", arg_positions=(1,))
    def scale(self, a: Any, x: Any) -> DenseArray:
        """Return the stacked scalar product ``a * x``."""
        return a * x

    def scale_batch(self, a: Any, x: Any) -> DenseArray:
        """Return the leading-axis batch scalar product of stacked elements."""
        return a * x

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: Any, y: Any) -> Any:
        """Return ``sum_i base.inner(x[i], y[i])`` as a scalar."""
        if self.base.is_euclidean:
            return self.ops.vdot(x, y)
        try:
            y_dual = self.base.riesz(y)
            return self.ops.vdot(x, y_dual)
        except _STACKED_FALLBACK_ERRORS:
            values = self.ops.vmap(self.base.inner, in_axes=(0, 0), out_axes=0)(x, y)
            return self.ops.sum(values)

    def riesz(self, x: Any) -> Any:
        """Apply the base Riesz map to every stacked copy."""
        if self.base.is_euclidean:
            return x
        try:
            return self.base.riesz(x)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(self.base.riesz, in_axes=0, out_axes=0)(x)

    def riesz_inverse(self, x: Any) -> Any:
        """Apply the base inverse Riesz map to every stacked copy."""
        if self.base.is_euclidean:
            return x
        try:
            return self.base.riesz_inverse(x)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(self.base.riesz_inverse, in_axes=0, out_axes=0)(x)

    @property
    def is_euclidean(self) -> bool:
        """Return whether the base geometry is Euclidean."""
        return self.base.is_euclidean


    def star(self, x: Any) -> Any:
        """Return the base-space star operation for each stacked copy."""
        try:
            return self.base.star(x)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(self.base.star, in_axes=0, out_axes=0)(x)

    @checked_method(in_space="self", arg_positions=(0, 1))
    def jordan(self, x: Any, y: Any) -> Any:
        """Return the base-space Jordan product for each stacked copy."""
        try:
            return self.base.jordan(x, y)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(self.base.jordan, in_axes=(0, 0), out_axes=0)(x, y)

    def spectrum(self, x: Any) -> Any:
        """Return spectra for each leading-axis copy of the base space."""
        try:
            return self.base.spectrum(x)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(self.base.spectrum, in_axes=0, out_axes=0)(x)

    def spectral_decompose(self, x: Any) -> Any:
        """Return spectral decompositions for each leading-axis copy."""
        try:
            return self.base.spectral_decompose(x)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(self.base.spectral_decompose, in_axes=0, out_axes=0)(x)

    def from_spectrum(self, eigvals: Any, frame: Any) -> Any:
        """Reconstruct stacked elements from base spectral data."""
        try:
            return self.base.from_spectrum(eigvals, frame)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(self.base.from_spectrum, in_axes=(0, 0), out_axes=0)(eigvals, frame)


    def spectral_apply(self, x: Any, f: Any) -> Any:
        """Apply the base-space spectral calculus to each stacked copy."""
        try:
            return self.base.spectral_apply(x, f)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(lambda xi: self.base.spectral_apply(xi, f), in_axes=0, out_axes=0)(x)

    def apply(self, x: Any, f: Any) -> Any:
        """Backward-compatible alias for spectral application."""
        return self.spectral_apply(x, f)

    @checked_method(in_space="self")
    def flatten(self, x: Any) -> DenseArray:
        """Flatten the whole stacked element to one coordinate vector."""
        return x.reshape((-1,))

    def unflatten(self, v: DenseArray) -> DenseArray:
        """Unflatten one coordinate vector to the stacked element shape."""
        v = self.ctx.assert_dense(v) if self._enable_checks else v
        return v.reshape(self.shape)

    def flatten_batch(self, xs: DenseArray) -> DenseArray:
        """Flatten a batch of stacked elements to ``(N, count * base.size)``."""
        xs = self.ctx.assert_dense(xs) if self._enable_checks else xs
        return xs.reshape((xs.shape[0], -1))

    def unflatten_batch(self, vs: DenseArray) -> DenseArray:
        """Unflatten rows to a batch of stacked elements."""
        vs = self.ctx.assert_dense(vs) if self._enable_checks else vs
        return vs.reshape((vs.shape[0],) + self.shape)

    def _convert(self, new_ctx: Context) -> StackedSpace:
        """Convert the base space and rebuild the stacked space."""
        return type(self)(self.base.convert(new_ctx), self.count, new_ctx)

    def stacked(self, count: int) -> StackedSpace:
        """Return a flattened stack of this stack: ``base.stacked(count * k)``."""
        return StackedSpace(self.base, self.count * int(count), self.ctx)

    def tree_flatten(self):
        """Flatten this space for JAX pytree registration."""
        return (), (self.base, self.count, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this space from pytree aux data."""
        base, count, ctx = aux
        return cls(base, count, ctx)


@jax_pytree_class
class StackedInnerProductSpace(StackedSpace, InnerProductSpace):
    """Stacked space whose base supports an inner product."""

    def __init__(self, base, count, ctx=None):
        super().__init__(base, count, ctx)
        if not isinstance(self.base, InnerProductSpace):
            raise TypeError("StackedInnerProductSpace requires an InnerProductSpace base.")


@jax_pytree_class
class StackedStarSpace(StackedSpace, StarSpace):
    """Stacked space whose base supports a star operation."""

    def __init__(self, base, count, ctx=None):
        super().__init__(base, count, ctx)
        if not isinstance(self.base, StarSpace):
            raise TypeError("StackedStarSpace requires a StarSpace base.")


@jax_pytree_class
class StackedJordanAlgebraSpace(StackedSpace, JordanAlgebraSpace):
    """Stacked space whose base supports Jordan algebra operations."""

    def __init__(self, base, count, ctx=None):
        super().__init__(base, count, ctx)
        if not isinstance(self.base, JordanAlgebraSpace):
            raise TypeError("StackedJordanAlgebraSpace requires a JordanAlgebraSpace base.")


@jax_pytree_class
class StackedEuclideanJordanAlgebraSpace(StackedSpace, StarSpace, EuclideanJordanAlgebraSpace):
    """Stacked space whose base supports Euclidean Jordan algebra operations."""

    def __init__(self, base, count, ctx=None):
        super().__init__(base, count, ctx)
        if not isinstance(self.base, EuclideanJordanAlgebraSpace):
            raise TypeError("StackedEuclideanJordanAlgebraSpace requires a EuclideanJordanAlgebraSpace base.")
        if not isinstance(self.base, StarSpace):
            raise TypeError("StackedEuclideanJordanAlgebraSpace requires a StarSpace base.")
