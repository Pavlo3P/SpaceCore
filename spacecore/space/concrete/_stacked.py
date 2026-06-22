from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ..base import (
    CoordinateSpace,
    EuclideanJordanAlgebraSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
    Space,
    StarSpace,
)
from ._tree_space import TreeSpace, _space_capabilities
from ..._checks import checked_method
from ..._contextual import resolve_context_priority
from ...backend import BackendOps, Context, jax_pytree_class
from ...types import DenseArray
from ..checks import BackendCheck, DTypeCheck, FieldCheck, ShapeCheck

_STACKED_FALLBACK_ERRORS = (TypeError, ValueError, AttributeError, IndexError)

_CAP_INNER = InnerProductSpace
_CAP_STAR = StarSpace
_CAP_JORDAN = JordanAlgebraSpace
_CAP_EUCLIDEAN_JORDAN = EuclideanJordanAlgebraSpace

_STACKED_REGISTRY: dict[frozenset[type], type[StackedSpace]] = {}


def _validate_stacked_base(base: Any, owner: str = "StackedSpace") -> CoordinateSpace:
    """Validate a stacked-space base before capability-specific access."""
    if not isinstance(base, CoordinateSpace):
        raise TypeError(
            f"{owner} requires base to be a CoordinateSpace; base is {type(base).__name__}."
        )
    if isinstance(base, TreeSpace):
        raise TypeError(
            "StackedSpace cannot wrap TreeSpace directly; use "
            "tree_space.stacked(count), which stacks each leaf."
        )
    return base


def _validate_count(count: int, owner: str = "StackedSpace") -> int:
    """Validate and normalize stacked copy count."""
    count = int(count)
    if count < 0:
        raise ValueError(f"{owner} count must be nonnegative.")
    return count


def _stacked_capabilities(base: Space) -> frozenset[type]:
    """Return capabilities copied from the stacked base space."""
    return _space_capabilities(base)


def _stacked_class_for(capabilities: frozenset[type]) -> type[StackedSpace]:
    """Return the deterministic concrete class for a stacked capability set."""
    return _STACKED_REGISTRY.get(capabilities, StackedSpace)


def _require_base(base: Space, capability: type, owner: str) -> None:
    """Raise if ``base`` lacks ``capability``."""
    if not isinstance(base, capability):
        raise TypeError(
            f"{owner} requires base to be a {capability.__name__}; base is {type(base).__name__}."
        )


@jax_pytree_class
class StackedSpace(CoordinateSpace):
    """
    Leading-axis copies of a coordinate leaf space.

    Baseline ``StackedSpace`` exposes only coordinate-space operations. Direct
    construction dispatches to a more specific class that preserves the base
    space's inner-product, star, Jordan, and Euclidean-Jordan capabilities.

    Parameters
    ----------
    base : Space
        Coordinate leaf space to repeat along a leading axis.
    count : int
        Number of leading-axis copies.
    ctx : Context, str, or None, optional
        Context specification. If omitted, the context is resolved from
        ``base``.
    """

    def __new__(cls, base: Space, count: int, ctx: Context | str | None = None):
        if cls is StackedSpace:
            base = _validate_stacked_base(base)
            _validate_count(count)
            resolved_ctx = resolve_context_priority(ctx, base)
            cls = _stacked_class_for(_stacked_capabilities(base.convert(resolved_ctx)))
        return super(StackedSpace, cls).__new__(cls)

    def __init__(self, base: Space, count: int, ctx: Context | str | None = None) -> None:
        base = _validate_stacked_base(base, type(self).__name__)
        count = _validate_count(count, type(self).__name__)
        ctx = resolve_context_priority(ctx, base)
        self.base = base.convert(ctx)
        self.count = count
        super().__init__((self.count,) + tuple(self.base.shape), ctx)

    def __eq__(self, other: Any) -> bool:
        """Return whether another stacked space has the same base and count."""
        if isinstance(other, StackedSpace):
            return self.ctx == other.ctx and self.count == other.count and self.base == other.base
        return False

    def _local_checks(self):
        """Return membership checks local to stacked dense coordinate spaces."""
        return BackendCheck(), ShapeCheck(), FieldCheck(), DTypeCheck()

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

    @checked_method(in_space="self")
    def flatten(self, x: Any) -> DenseArray:
        """Flatten the whole stacked element to one coordinate vector."""
        return x.reshape((-1,))

    def unflatten(self, v: DenseArray) -> DenseArray:
        """Unflatten one coordinate vector to the stacked element shape."""
        v = self._coerce_dense(v)
        return v.reshape(self.shape)

    def flatten_batch(self, xs: DenseArray) -> DenseArray:
        """Flatten a batch of stacked elements to ``(N, count * base.size)``."""
        xs = self._coerce_dense(xs)
        return xs.reshape((xs.shape[0], -1))

    def unflatten_batch(self, vs: DenseArray) -> DenseArray:
        """Unflatten rows to a batch of stacked elements."""
        vs = self._coerce_dense(vs)
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


class _StackedInnerProductMixin:
    """Inner-product operations for stacks whose base supports them."""

    if TYPE_CHECKING:
        # Provided by the StackedSpace host this mixin is combined with; narrowed
        # to the relevant capability per method (see ``cast`` calls below).
        @property
        def base(self) -> Space: ...
        @property
        def ops(self) -> BackendOps: ...

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: Any, y: Any) -> Any:
        """Return ``sum_i base.inner(x[i], y[i])`` as a scalar."""
        base = cast(InnerProductSpace, self.base)
        if base.is_euclidean:
            return self.ops.vdot(x, y)
        try:
            y_dual = base.riesz(y)
            return self.ops.vdot(x, y_dual)
        except _STACKED_FALLBACK_ERRORS:
            values = self.ops.vmap(base.inner, in_axes=(0, 0), out_axes=0)(x, y)
            return self.ops.sum(values)

    def riesz(self, x: Any) -> Any:
        """Apply the base Riesz map to every stacked copy."""
        base = cast(InnerProductSpace, self.base)
        if base.is_euclidean:
            return x
        try:
            return base.riesz(x)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(base.riesz, in_axes=0, out_axes=0)(x)

    def riesz_inverse(self, x: Any) -> Any:
        """Apply the base inverse Riesz map to every stacked copy."""
        base = cast(InnerProductSpace, self.base)
        if base.is_euclidean:
            return x
        try:
            return base.riesz_inverse(x)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(base.riesz_inverse, in_axes=0, out_axes=0)(x)

    @property
    def is_euclidean(self) -> bool:
        """Return whether the base geometry is Euclidean."""
        return cast(InnerProductSpace, self.base).is_euclidean


class _StackedStarMixin:
    """Star operation for stacks whose base supports it."""

    if TYPE_CHECKING:
        # Provided by the StackedSpace host this mixin is combined with.
        @property
        def base(self) -> Space: ...
        @property
        def ops(self) -> BackendOps: ...

    def star(self, x: Any) -> Any:
        """Return the base-space star operation for each stacked copy."""
        base = cast(StarSpace, self.base)
        try:
            return base.star(x)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(base.star, in_axes=0, out_axes=0)(x)


class _StackedJordanMixin:
    """Jordan operations for stacks whose base supports them."""

    if TYPE_CHECKING:
        # Provided by the StackedSpace host this mixin is combined with.
        @property
        def base(self) -> Space: ...
        @property
        def ops(self) -> BackendOps: ...

    @checked_method(in_space="self", arg_positions=(0, 1))
    def jordan(self, x: Any, y: Any) -> Any:
        """Return the base-space Jordan product for each stacked copy."""
        base = cast(JordanAlgebraSpace, self.base)
        try:
            return base.jordan(x, y)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(base.jordan, in_axes=(0, 0), out_axes=0)(x, y)

    def spectrum(self, x: Any) -> Any:
        """Return spectra for each leading-axis copy of the base space."""
        base = cast(JordanAlgebraSpace, self.base)
        try:
            return base.spectrum(x)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(base.spectrum, in_axes=0, out_axes=0)(x)

    def spectral_decompose(self, x: Any) -> Any:
        """Return spectral decompositions for each leading-axis copy."""
        base = cast(JordanAlgebraSpace, self.base)
        try:
            return base.spectral_decompose(x)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(base.spectral_decompose, in_axes=0, out_axes=0)(x)

    def from_spectrum(self, eigvals: Any, frame: Any) -> Any:
        """Reconstruct stacked elements from base spectral data."""
        base = cast(JordanAlgebraSpace, self.base)
        try:
            return base.from_spectrum(eigvals, frame)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(base.from_spectrum, in_axes=(0, 0), out_axes=0)(
                eigvals, frame
            )

    def spectral_apply(self, x: Any, f: Any) -> Any:
        """Apply the base-space spectral calculus to each stacked copy."""
        base = cast(JordanAlgebraSpace, self.base)
        try:
            return base.spectral_apply(x, f)
        except _STACKED_FALLBACK_ERRORS:
            return self.ops.vmap(lambda xi: base.spectral_apply(xi, f), in_axes=0, out_axes=0)(
                x
            )


@jax_pytree_class
class _StackedInnerProductSpace(_StackedInnerProductMixin, StackedSpace, InnerProductSpace):
    """Stacked space whose base supports an inner product."""

    def __init__(self, base, count, ctx=None):
        base = _validate_stacked_base(base, type(self).__name__)
        _require_base(base, InnerProductSpace, type(self).__name__)
        super().__init__(base, count, ctx)


@jax_pytree_class
class _StackedStarSpace(_StackedStarMixin, StackedSpace, StarSpace):
    """Stacked space whose base supports a star operation."""

    def __init__(self, base, count, ctx=None):
        base = _validate_stacked_base(base, type(self).__name__)
        _require_base(base, StarSpace, type(self).__name__)
        super().__init__(base, count, ctx)


@jax_pytree_class
class _StackedJordanAlgebraSpace(_StackedJordanMixin, StackedSpace, JordanAlgebraSpace):
    """Stacked space whose base supports Jordan algebra operations."""

    def __init__(self, base, count, ctx=None):
        base = _validate_stacked_base(base, type(self).__name__)
        _require_base(base, JordanAlgebraSpace, type(self).__name__)
        super().__init__(base, count, ctx)


@jax_pytree_class
class _StackedEuclideanJordanAlgebraSpace(
    _StackedInnerProductMixin,
    _StackedJordanMixin,
    StackedSpace,
    EuclideanJordanAlgebraSpace,
):
    """Stacked space whose base supports Euclidean Jordan algebra operations."""

    def __init__(self, base, count, ctx=None):
        base = _validate_stacked_base(base, type(self).__name__)
        _require_base(base, EuclideanJordanAlgebraSpace, type(self).__name__)
        super().__init__(base, count, ctx)
        _require_base(self.base, EuclideanJordanAlgebraSpace, type(self).__name__)


@jax_pytree_class
class _StackedInnerProductStarSpace(
    _StackedInnerProductMixin,
    _StackedStarMixin,
    StackedSpace,
    InnerProductSpace,
    StarSpace,
):
    """Stacked implementation for inner-product plus star capability."""


@jax_pytree_class
class _StackedInnerProductJordanSpace(
    _StackedInnerProductMixin,
    _StackedJordanMixin,
    StackedSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
):
    """Stacked implementation for inner-product plus Jordan capability."""


@jax_pytree_class
class _StackedStarJordanSpace(
    _StackedStarMixin,
    _StackedJordanMixin,
    StackedSpace,
    StarSpace,
    JordanAlgebraSpace,
):
    """Stacked implementation for star plus Jordan capability."""


@jax_pytree_class
class _StackedInnerProductStarJordanSpace(
    _StackedInnerProductMixin,
    _StackedStarMixin,
    _StackedJordanMixin,
    StackedSpace,
    InnerProductSpace,
    StarSpace,
    JordanAlgebraSpace,
):
    """Stacked implementation for inner-product, star, and Jordan capability."""


@jax_pytree_class
class _StackedEuclideanJordanStarSpace(
    _StackedStarMixin,
    _StackedEuclideanJordanAlgebraSpace,
    StarSpace,
):
    """Stacked implementation for Euclidean-Jordan plus star capability."""


_STACKED_REGISTRY.update(
    {
        frozenset(): StackedSpace,
        frozenset({_CAP_INNER}): _StackedInnerProductSpace,
        frozenset({_CAP_STAR}): _StackedStarSpace,
        frozenset({_CAP_JORDAN}): _StackedJordanAlgebraSpace,
        frozenset({_CAP_INNER, _CAP_STAR}): _StackedInnerProductStarSpace,
        frozenset({_CAP_INNER, _CAP_JORDAN}): _StackedInnerProductJordanSpace,
        frozenset({_CAP_STAR, _CAP_JORDAN}): _StackedStarJordanSpace,
        frozenset({_CAP_INNER, _CAP_STAR, _CAP_JORDAN}): _StackedInnerProductStarJordanSpace,
        frozenset(
            {_CAP_INNER, _CAP_JORDAN, _CAP_EUCLIDEAN_JORDAN}
        ): _StackedEuclideanJordanAlgebraSpace,
        frozenset(
            {_CAP_INNER, _CAP_STAR, _CAP_JORDAN, _CAP_EUCLIDEAN_JORDAN}
        ): _StackedEuclideanJordanStarSpace,
    }
)


__all__ = [
    "StackedSpace",
    "_stacked_capabilities",
]
