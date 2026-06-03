from __future__ import annotations

from typing import Any

from ._base import Space
from ._product import ProductSpace
from ._vector import VectorSpace
from ..backend import Context, jax_pytree_class
from .._checks import checked_method
from .._contextual import resolve_context_priority
from ..types import DenseArray

_STACKED_FALLBACK_ERRORS = (TypeError, ValueError, AttributeError, IndexError)


@jax_pytree_class
class StackedSpace(VectorSpace):
    """N independent copies of a base space, stacked on a leading axis.

    Elements have shape ``(count,) + base.shape``. Operations act independently
    per copy where appropriate, and the inner product sums over the stack axis,
    so the stack is itself one Hilbert-space element.

    ``StackedSpace`` wraps leaf spaces only. For products, use
    ``ProductSpace(...).stacked(count)``, which returns a product of stacked
    component spaces.
    """

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
        super().__init__((self.count,) + tuple(self.base.shape), ctx, geometry=self.base.geometry)

    def __eq__(self, other: Any) -> bool:
        """Return whether another stacked space has the same base and count."""
        if type(other) is type(self):
            return self.ctx == other.ctx and self.count == other.count and self.base == other.base
        return False

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
        return StackedSpace(self.base.convert(new_ctx), self.count, new_ctx)

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
