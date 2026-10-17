from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ..base import (
    CoordinateSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
    Space,
    StarSpace,
)
from ._dispatch import build_dispatch_classes, capability_abcs
from ._tree_space import TreeSpace
from .._capabilities import (
    _CAP_INNER, _CAP_STAR, _CAP_JORDAN,
    _space_capabilities, registry_key,
)
from ..._checks import CheckLevel
from ...contextual import resolve_context_priority
from ...contextual import Context
from ...backend import BackendOps
from ...backend import PyTreeNode
from ...types import DenseArray
from ..checks import BackendCheck, DTypeCheck, FieldCheck, ShapeCheck

_STACKED_FALLBACK_ERRORS = (TypeError, ValueError, AttributeError, IndexError)

_STACKED_REGISTRY: dict[frozenset[type], type[StackedSpace]] = {}


def _validate_stacked_base(base: Any, owner: str = "StackedSpace") -> CoordinateSpace:
    """Validate a stacked-space base before capability-specific access."""
    # The base must be a coordinate space; trees stack leaf by leaf instead.
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
    """Return capabilities copied from the stacked base space (validated by its own __init__)."""
    return _space_capabilities(base)


def _require_base(base: Space, capability: type, owner: str) -> None:
    """Raise if ``base`` lacks ``capability``."""
    if not isinstance(base, capability):
        raise TypeError(
            f"{owner} requires base to be a {capability.__name__}; base is {type(base).__name__}."
        )


class StackedSpace(PyTreeNode, CoordinateSpace):
    """
    ``count`` leading-axis copies of a coordinate space; operations ``vmap`` over the copies.

    Construction picks a subclass that keeps the base space's inner-product, star and
    Jordan capabilities.

    Parameters
    ----------
    base : Space
        Coordinate space to repeat.
    count : int
        Number of copies.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``base`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    _fields = ("base", "count", "ctx")

    def __new__(
        cls,
        base: Space,
        count: int,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ):
        # Pick the subclass that carries exactly the base's capabilities.
        if cls is StackedSpace:
            base = _validate_stacked_base(base)
            _validate_count(count)
            resolved_ctx = resolve_context_priority(ctx, base)
            capabilities = _stacked_capabilities(base.convert(resolved_ctx))
            cls = _STACKED_REGISTRY.get(registry_key(capabilities), StackedSpace)
        return super(StackedSpace, cls).__new__(cls)

    def __init__(
        self,
        base: Space,
        count: int,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        # Element shape is (count,) + base.shape.
        base = _validate_stacked_base(base, type(self).__name__)
        count = _validate_count(count, type(self).__name__)
        ctx = resolve_context_priority(ctx, base)
        self.base = base.convert(ctx)
        self.count = count
        super().__init__(
            (self.count,) + tuple(self.base.shape), ctx, check_level=check_level
        )

    def _eq_algebra(self, other: Any) -> bool:
        # Tier 2: count + base. ``base == other.base`` is load-bearing — the
        # __new__ capability dispatch can map different bases onto the same
        # private subclass, so the type-identity gate alone is not sufficient.
        return super()._eq_algebra(other) and self.count == other.count and self.base == other.base

    def _repr_class_name(self) -> str:
        """Present the public ``StackedSpace`` label, not the private dispatch subclass."""
        return "StackedSpace"

    def _space_descriptor(self) -> str:
        """Return ``count×<base descriptor>`` (e.g. ``8×ℝ^3``)."""
        from ..._repr import describe_space

        return f"{self.count}×{describe_space(self.base)}"

    def _local_checks(self):
        """Return membership checks local to stacked dense coordinate spaces."""
        return BackendCheck(), ShapeCheck(), FieldCheck(), DTypeCheck()

    def zeros(self) -> DenseArray:
        """Return the stacked zero element."""
        return self.ops.zeros(self.shape, dtype=self.dtype)

    def ones(self) -> DenseArray:
        """Return the stacked all-ones element."""
        return self.ops.ones(self.shape, dtype=self.dtype)

    def _add_core(self, x: Any, y: Any) -> DenseArray:
        """Return the stacked sum ``x + y``."""
        return x + y

    def _add_batch_core(self, x: Any, y: Any) -> DenseArray:
        """Return the leading-axis batch sum of stacked elements."""
        return x + y

    def _scale_core(self, a: Any, x: Any) -> DenseArray:
        """Return the stacked scalar product ``a * x``."""
        return a * x

    def _scale_batch_core(self, a: Any, x: Any) -> DenseArray:
        """Return the leading-axis batch scalar product of stacked elements."""
        return a * x

    def _flatten_core(self, x: Any) -> DenseArray:
        """Flatten the whole stacked element to one coordinate vector."""
        return x.reshape((-1,))

    def _unflatten_core(self, v: DenseArray) -> DenseArray:
        """Unflatten one coordinate vector to the stacked element shape."""
        v = self._coerce_dense(v)
        return v.reshape(self.shape)

    def _flatten_batch_core(self, xs: DenseArray) -> DenseArray:
        """Flatten a batch of stacked elements to ``(N, count * base.size)``."""
        xs = self._coerce_dense(xs)
        return xs.reshape((xs.shape[0], -1))

    def _unflatten_batch_core(self, vs: DenseArray) -> DenseArray:
        """Unflatten rows to a batch of stacked elements."""
        vs = self._coerce_dense(vs)
        return vs.reshape((vs.shape[0],) + self.shape)

    def stacked(self, count: int) -> StackedSpace:
        """Return a flattened stack of this stack: ``base.stacked(count * k)``."""
        return StackedSpace(self.base, self.count * int(count), self.ctx)

class _StackedHostMixin:
    """Type-only declarations of the StackedSpace host surface used by stack mixins."""

    if TYPE_CHECKING:
        # Provided by the StackedSpace host these mixins are combined with; the
        # base is narrowed to the relevant capability per method (see ``cast`` calls).
        @property
        def base(self) -> Space: ...
        @property
        def ops(self) -> BackendOps: ...

    def _per_copy(self, fn: Any, *args: Any) -> Any:
        """Call ``fn`` on the whole stack; if the base cannot take it, vmap ``fn`` over the copies."""
        try:
            return fn(*args)
        except _STACKED_FALLBACK_ERRORS:
            in_axes = 0 if len(args) == 1 else (0,) * len(args)
            return self.ops.vmap(fn, in_axes=in_axes, out_axes=0)(*args)


class _StackedInnerProductMixin(_StackedHostMixin):
    """Inner-product operations for stacks whose base supports them."""

    def _inner_core(self, x: Any, y: Any) -> Any:
        """Return ``sum_i base.inner(x[i], y[i])`` as a scalar."""
        base = cast(InnerProductSpace, self.base)
        # One vdot over all copies, against the base's batched Riesz map of y.
        return self.ops.vdot(x, y if base.is_euclidean else base._riesz_core(y))

    def _riesz_core(self, x: Any) -> Any:
        """Apply the base Riesz map to every stacked copy (the geometry broadcasts)."""
        base = cast(InnerProductSpace, self.base)
        return x if base.is_euclidean else base._riesz_core(x)

    def _riesz_inverse_core(self, x: Any) -> Any:
        """Apply the base inverse Riesz map to every stacked copy."""
        base = cast(InnerProductSpace, self.base)
        return x if base.is_euclidean else base._riesz_inverse_core(x)

    @property
    def is_euclidean(self) -> bool:
        """Return whether the base geometry is Euclidean."""
        return cast(InnerProductSpace, self.base).is_euclidean


class _StackedStarMixin(_StackedHostMixin):
    """Star operation for stacks whose base supports it."""

    def _star_core(self, x: Any) -> Any:
        """Return the base-space star operation for each stacked copy."""
        base = cast(StarSpace, self.base)
        return self._per_copy(base._star_core, x)


class _StackedJordanMixin(_StackedHostMixin):
    """Jordan operations for stacks whose base supports them."""

    def _jordan_core(self, x: Any, y: Any) -> Any:
        """Return the base-space Jordan product for each stacked copy."""
        base = cast(JordanAlgebraSpace, self.base)
        return self._per_copy(base._jordan_core, x, y)

    def _spectrum_core(self, x: Any) -> Any:
        """Return spectra for each leading-axis copy of the base space."""
        base = cast(JordanAlgebraSpace, self.base)
        return self._per_copy(base._spectrum_core, x)

    def _spectral_decompose_core(self, x: Any) -> Any:
        """Return spectral decompositions for each leading-axis copy."""
        base = cast(JordanAlgebraSpace, self.base)
        return self._per_copy(base._spectral_decompose_core, x)

    def _from_spectrum_core(self, eigvals: Any, frame: Any) -> Any:
        """Reconstruct stacked elements from base spectral data."""
        base = cast(JordanAlgebraSpace, self.base)
        return self._per_copy(base._from_spectrum_core, eigvals, frame)

    def _spectral_apply_core(self, x: Any, f: Any) -> Any:
        """Apply the base-space spectral calculus to each stacked copy."""
        base = cast(JordanAlgebraSpace, self.base)
        return self._per_copy(lambda xi: base._spectral_apply_core(xi, f), x)

    def _trace_core(self, x: Any) -> Any:
        """Return the direct-sum trace: the sum of the per-copy base traces."""
        base = cast(JordanAlgebraSpace, self.base)
        traces = self._per_copy(base._trace_core, x)
        # Reduce only the copy axis; leading batch axes are preserved.
        return self.ops.sum(traces, axis=-1)

    def _determinant_core(self, x: Any) -> Any:
        """Return the direct-sum determinant: the product of the per-copy base determinants."""
        base = cast(JordanAlgebraSpace, self.base)
        dets = self._per_copy(base._determinant_core, x)
        # Reduce only the copy axis; leading batch axes are preserved.
        return self.ops.prod(dets, axis=-1)

    def unit(self) -> Any:
        """Return the stacked identity: ``base.unit()`` replicated across every copy."""
        base = cast(JordanAlgebraSpace, self.base)
        return self.ops.broadcast_to(base.unit(), self.shape)


def _guarded_init(self, base, count, ctx=None, check_level=None):
    """Build the stack, then require the converted base to have this subclass's capabilities."""
    StackedSpace.__init__(self, base, count, ctx, check_level=check_level)
    for capability in capability_abcs(type(self)._dispatch_caps):
        _require_base(self.base, capability, type(self).__name__)


_STACKED_REGISTRY.update(build_dispatch_classes(
    StackedSpace,
    {_CAP_INNER: _StackedInnerProductMixin, _CAP_STAR: _StackedStarMixin, _CAP_JORDAN: _StackedJordanMixin},
    prefix="_Stacked",
    module_globals=globals(),
    namespace={"__init__": _guarded_init},
))


__all__ = [
    "StackedSpace",
    "_stacked_capabilities",
]
