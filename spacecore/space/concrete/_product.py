from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

from ..base import (
    CoordinateSpace,
    EuclideanJordanAlgebraSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
    Space,
    StarSpace,
)
from ..checks import ProductComponentCheck, ProductStructureCheck
from .._structure import ProductStructure, PytreeStructure, TupleStructure
from ._dense_coordinate import DenseCoordinateSpace
from ._dense_vector import (
    DenseVectorSpace,
    ElementwiseJordanSpace,
    EuclideanElementwiseJordanSpace,
    _validate_euclidean_elementwise_jordan,
)
from ..._checks import checked_method
from ..._contextual import resolve_context_priority
from ...backend import Context, jax_pytree_class
from ...types import DenseArray

ProductElement = Any
CapabilitySet = frozenset[type]


@jax_pytree_class
@dataclass(frozen=True)
class ProductSpectralDecomposition:
    """
    Component spectral data independent of product element structure.

    Parameters
    ----------
    eigvals : tuple
        Per-component eigenvalue data in product component order.
    frames : tuple
        Per-component frame data in product component order.
    """

    eigvals: tuple[Any, ...]
    frames: tuple[Any, ...]

    def tree_flatten(self):
        """Flatten spectral data for JAX pytree registration."""
        return (self.eigvals, self.frames), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild spectral data from JAX pytree children."""
        eigvals, frames = children
        return cls(tuple(eigvals), tuple(frames))


_CAP_INNER = InnerProductSpace
_CAP_STAR = StarSpace
_CAP_JORDAN = JordanAlgebraSpace
_CAP_EUCLIDEAN_JORDAN = EuclideanJordanAlgebraSpace


# Filled after concrete classes are defined. ``ProductSpace.__new__`` looks up
# this registry at construction time, after module import has completed.
_PRODUCT_REGISTRY: dict[CapabilitySet, type[ProductSpace]] = {}


def _prod_int(shape: Tuple[int, ...]) -> int:
    """Return the integer product of a shape tuple."""
    p = 1
    for d in shape:
        p *= int(d)
    return int(p)


def _space_capabilities(space: Space) -> CapabilitySet:
    """Return structural capabilities advertised by ``space``."""
    caps: set[type] = set()
    if isinstance(space, InnerProductSpace):
        caps.add(_CAP_INNER)
    if isinstance(space, StarSpace):
        caps.add(_CAP_STAR)
    if isinstance(space, JordanAlgebraSpace):
        caps.add(_CAP_JORDAN)
    if isinstance(space, EuclideanJordanAlgebraSpace):
        if isinstance(space, EuclideanElementwiseJordanSpace):
            _validate_euclidean_elementwise_jordan(space, space.geometry)
        caps.add(_CAP_EUCLIDEAN_JORDAN)
    return frozenset(caps)


def _validate_product_spaces(
    spaces: Any, owner: str = "ProductSpace"
) -> tuple[CoordinateSpace, ...]:
    """Validate product components before capability-specific access."""
    if not isinstance(spaces, Sequence):
        raise TypeError(
            f"{owner} requires a sequence of CoordinateSpace components; got {type(spaces).__name__}."
        )
    spaces = tuple(spaces)
    if len(spaces) == 0:
        raise ValueError(f"{owner} requires at least one subspace.")
    for index, component in enumerate(spaces):
        if not isinstance(component, CoordinateSpace):
            raise TypeError(
                f"{owner} requires every component to be a CoordinateSpace; "
                f"component {index} is {type(component).__name__}."
            )
    return spaces


def _product_capabilities(spaces: Sequence[Space]) -> CapabilitySet:
    """Return capabilities shared by every product component."""
    if not spaces:
        return frozenset()
    shared = set(_space_capabilities(spaces[0]))
    for component in spaces[1:]:
        shared.intersection_update(_space_capabilities(component))
    return frozenset(shared)


def _product_class_for(capabilities: CapabilitySet) -> type[ProductSpace]:
    """Return the deterministic concrete class for a product capability set."""
    return _PRODUCT_REGISTRY.get(capabilities, ProductSpace)


def _require_all_components(spaces: Sequence[Space], capability: type, owner: str) -> None:
    """Raise if any component lacks ``capability``."""
    for index, component in enumerate(spaces):
        if not isinstance(component, capability):
            raise TypeError(
                f"{owner} requires every component to be a {capability.__name__}; "
                f"component {index} is {type(component).__name__}."
            )


@jax_pytree_class
class ProductSpace(CoordinateSpace):
    r"""
    Represent a Cartesian product of coordinate spaces.

    Baseline ``ProductSpace`` exposes only coordinate-space operations. When
    constructed directly, it dispatches to a more specific concrete class if all
    components share inner-product, star, Jordan, or Euclidean-Jordan
    capabilities.

    Parameters
    ----------
    spaces : tuple of Space
        Coordinate component spaces in product order.
    ctx : Context, str, or None, optional
        Context specification. If omitted, the context is resolved from the
        component spaces.
    structure : ProductStructure or None, optional
        Element adapter used to map product elements to ordered components. If
        omitted, tuple-structured product elements are used.
    """

    def __new__(
        cls,
        spaces: Tuple[Space, ...],
        ctx: Context | str | None = None,
        structure: ProductStructure | None = None,
    ):
        if cls is ProductSpace:
            spaces_tuple = _validate_product_spaces(spaces)
            resolved_ctx = resolve_context_priority(ctx, *spaces_tuple)
            converted_spaces = tuple(space.convert(resolved_ctx) for space in spaces_tuple)
            cls = _product_class_for(_product_capabilities(converted_spaces))
        return super(ProductSpace, cls).__new__(cls)

    def __init__(
        self,
        spaces: Tuple[Space, ...],
        ctx: Context | str | None = None,
        structure: ProductStructure | None = None,
    ) -> None:
        spaces = _validate_product_spaces(spaces, type(self).__name__)
        if structure is None:
            structure = TupleStructure()
        if not isinstance(structure, ProductStructure):
            raise TypeError(
                "ProductSpace structure must be a ProductStructure, "
                f"got {type(structure).__name__}."
            )
        ctx = resolve_context_priority(ctx, *spaces)

        dims = tuple(_prod_int(s.shape) for s in spaces)
        offsets: List[int] = [0]
        for d in dims:
            offsets.append(offsets[-1] + d)

        self._dims = dims
        self._offsets = tuple(offsets)
        self._slices = tuple(slice(offsets[i], offsets[i + 1]) for i in range(len(dims)))
        shape = (offsets[-1],)

        super(ProductSpace, self).__init__(shape, ctx)
        uniform_spaces = tuple(sp.convert(self.ctx) for sp in spaces)
        self.spaces = uniform_spaces
        self._structure = structure
        self._arity = len(uniform_spaces)
        self._vector_fast_path = all(
            type(sp)
            in (
                DenseCoordinateSpace,
                DenseVectorSpace,
                ElementwiseJordanSpace,
                EuclideanElementwiseJordanSpace,
            )
            for sp in uniform_spaces
        )
        self._component_shapes = tuple(sp.shape for sp in uniform_spaces)
        self._component_is_flat = tuple(
            shape == (dim,) for shape, dim in zip(self._component_shapes, self._dims)
        )
        raw_array_ops = getattr(self.ctx.ops, "np", None)
        if raw_array_ops is not None:
            self._concatenate = raw_array_ops.concatenate
            self._concatenate_uses_dim = False
        else:
            raw_array_ops = getattr(self.ctx.ops, "jnp", None)
            if raw_array_ops is not None:
                self._concatenate = raw_array_ops.concatenate
                self._concatenate_uses_dim = False
            else:
                raw_torch = getattr(self.ctx.ops, "torch", None)
                if raw_torch is not None:
                    self._concatenate = raw_torch.cat
                    self._concatenate_uses_dim = True
                else:
                    self._concatenate = self.ctx.ops.concatenate
                    self._concatenate_uses_dim = False
        if self._arity >= 1:
            self._slice0 = self._slices[0]
            self._shape0 = self._component_shapes[0]
            self._is_flat0 = self._component_is_flat[0]
        if self._arity >= 2:
            self._slice1 = self._slices[1]
            self._shape1 = self._component_shapes[1]
            self._is_flat1 = self._component_is_flat[1]

    def __eq__(self, other: Any) -> bool:
        """Return whether another product space has the same ordered components."""
        if isinstance(other, ProductSpace):
            return (
                self.ctx == other.ctx
                and self.spaces == other.spaces
                and self._structure == other._structure
            )
        return False

    @classmethod
    def from_template(
        cls,
        spaces: Tuple[Space, ...],
        template_element: Any,
        ctx: Context | str | None = None,
    ) -> ProductSpace:
        """Build a product whose elements match a registered pytree template."""
        structure = PytreeStructure(template_element)
        product = cls(spaces, ctx=ctx, structure=structure)
        structure.to_components(template_element, arity=product.arity)
        return product

    def _convert(self, new_ctx: Context) -> ProductSpace:
        """Convert all component spaces to ``new_ctx``."""
        new_spaces = tuple(sp.convert(new_ctx) for sp in self.spaces)
        return ProductSpace(new_spaces, new_ctx, structure=self._structure)

    def _local_checks(self):
        """Return membership checks local to product spaces."""
        return ProductStructureCheck(), ProductComponentCheck()

    @property
    def arity(self) -> int:
        """Number of component spaces."""
        return len(self.spaces)

    @property
    def structure(self) -> ProductStructure:
        """Element representation adapter for this product space."""
        return self._structure

    def _components(self, x: Any) -> tuple[Any, ...]:
        """Return ordered product components using the configured structure."""
        return self._structure.to_components(x, arity=self.arity)

    def _from_components(self, parts: tuple[Any, ...]) -> Any:
        """Rebuild a product element using the configured structure."""
        return self._structure.from_components(tuple(parts), arity=self.arity)

    def _ones_for_space(self, space: Space) -> Any:
        """Return a component one element without requiring Space.ones globally."""
        ones = getattr(space, "ones", None)
        if callable(ones):
            return ones()
        return self.ops.ones(space.shape, dtype=self.dtype)

    def zeros(self) -> ProductElement:
        """Return the product-space zero element."""
        return self._from_components(tuple(s.zeros() for s in self.spaces))

    def ones(self) -> ProductElement:
        """Return the product-space all-ones element."""
        return self._from_components(tuple(self._ones_for_space(s) for s in self.spaces))

    @checked_method(in_space="self", arg_positions=(0, 1))
    def add(self, x: ProductElement, y: ProductElement) -> ProductElement:
        """Return the componentwise product-space sum."""
        x_parts = self._components(x)
        y_parts = self._components(y)
        out = tuple(s.add(xi, yi) for s, xi, yi in zip(self.spaces, x_parts, y_parts))
        return self._from_components(out)

    def add_batch(self, x: ProductElement, y: ProductElement) -> ProductElement:
        """Return the componentwise leading-axis batch sum."""
        x_parts = self._components(x)
        y_parts = self._components(y)
        out = tuple(s.add_batch(xi, yi) for s, xi, yi in zip(self.spaces, x_parts, y_parts))
        return self._from_components(out)

    @checked_method(in_space="self", arg_positions=(1,))
    def scale(self, a: Any, x: ProductElement) -> ProductElement:
        """Return the componentwise scalar product."""
        parts = self._components(x)
        out = tuple(s.scale(a, xi) for s, xi in zip(self.spaces, parts))
        return self._from_components(out)

    def scale_batch(self, a: Any, x: ProductElement) -> ProductElement:
        """Return the componentwise leading-axis batch scalar product."""
        parts = self._components(x)
        out = tuple(s.scale_batch(a, xi) for s, xi in zip(self.spaces, parts))
        return self._from_components(out)

    def stacked(self, count: int) -> ProductSpace:
        """Return a product whose components are stacked leaf spaces."""
        return ProductSpace(
            tuple(s.stacked(count) for s in self.spaces),
            self.ctx,
            structure=self._structure,
        )

    @checked_method(in_space="self")
    def flatten(self, x: ProductElement) -> DenseArray:
        """Concatenate component coordinate vectors into one dense vector."""
        x_parts = self._components(x)
        if self._vector_fast_path:
            if self._arity == 1:
                return x_parts[0] if self._component_is_flat[0] else x_parts[0].reshape((-1,))
            if self._arity == 2:
                x0 = x_parts[0] if self._is_flat0 else x_parts[0].reshape((-1,))
                x1 = x_parts[1] if self._is_flat1 else x_parts[1].reshape((-1,))
                if self._concatenate_uses_dim:
                    return self._concatenate((x0, x1), dim=0)
                return self._concatenate((x0, x1), axis=0)
            parts = tuple(
                xi if is_flat else xi.reshape((-1,))
                for xi, is_flat in zip(x_parts, self._component_is_flat)
            )
            if self._concatenate_uses_dim:
                return self._concatenate(parts, dim=0)
            return self._concatenate(parts, axis=0)

        parts = []
        for s, xi in zip(self.spaces, x_parts):
            vi = s.flatten(xi)
            if self._checks_at_least("cheap"):
                vi = self.ctx.assert_dense(vi)
            parts.append(vi)

        if len(parts) == 1:
            return parts[0]

        if self._concatenate_uses_dim:
            return self._concatenate(parts, dim=0)
        return self._concatenate(parts, axis=0)

    def unflatten(self, v: DenseArray) -> ProductElement:
        """Split dense coordinates into component-space elements."""
        if self._checks_at_least("cheap"):
            v = self.ctx.assert_dense(v)
            v1 = v if tuple(getattr(v, "shape", ())) == self.shape else v.reshape((-1,))
        else:
            v1 = v

        if self._vector_fast_path:
            if self._arity == 1:
                x0 = v1[self._slice0]
                return self._from_components((x0 if self._is_flat0 else x0.reshape(self._shape0),))
            if self._arity == 2:
                x0 = v1[self._slice0]
                x1 = v1[self._slice1]
                if not self._is_flat0:
                    x0 = x0.reshape(self._shape0)
                if not self._is_flat1:
                    x1 = x1.reshape(self._shape1)
                return self._from_components((x0, x1))
            parts = tuple(
                v1[slc] if is_flat else v1[slc].reshape(shape)
                for slc, shape, is_flat in zip(
                    self._slices, self._component_shapes, self._component_is_flat
                )
            )
            return self._from_components(parts)

        xs: List[Any] = []
        for s, slc in zip(self.spaces, self._slices):
            vi = v1[slc]
            xs.append(s.unflatten(vi))

        return self._from_components(tuple(xs))

    def flatten_batch(self, xs: ProductElement) -> DenseArray:
        """Concatenate a leading-axis batch of product elements to ``(N, size)``."""
        xs_parts = self._components(xs)
        parts = tuple(s.flatten_batch(xi) for s, xi in zip(self.spaces, xs_parts))
        if len(parts) == 1:
            return parts[0]
        if self._concatenate_uses_dim:
            return self._concatenate(parts, dim=1)
        return self._concatenate(parts, axis=1)

    def unflatten_batch(self, vs: DenseArray) -> ProductElement:
        """Split rows of shape ``(N, size)`` into batched component elements."""
        if self._checks_at_least("cheap"):
            vs = self.ctx.assert_dense(vs)
        parts = tuple(s.unflatten_batch(vs[:, slc]) for s, slc in zip(self.spaces, self._slices))
        return self._from_components(parts)

    def tree_flatten(self):
        """Flatten this space for JAX pytree registration."""
        return (), (self.spaces, self.ctx, self._structure)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this space from pytree aux data."""
        spaces, ctx, structure = aux
        return cls(spaces, ctx, structure=structure)


class _ProductInnerProductMixin:
    """Inner-product operations for products whose components all support them."""

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: ProductElement, y: ProductElement) -> Any:
        """Return the sum of component inner products."""
        x_parts = self._components(x)
        y_parts = self._components(y)
        acc = None
        for s, xi, yi in zip(self.spaces, x_parts, y_parts):
            v = s.inner(xi, yi)
            acc = v if acc is None else (acc + v)
        return acc

    def riesz(self, x: ProductElement) -> ProductElement:
        """Apply each component space's Riesz map."""
        parts = self._components(x)
        out = tuple(s.riesz(xi) for s, xi in zip(self.spaces, parts))
        return self._from_components(out)

    def riesz_inverse(self, x: ProductElement) -> ProductElement:
        """Apply each component space's inverse Riesz map."""
        parts = self._components(x)
        out = tuple(s.riesz_inverse(xi) for s, xi in zip(self.spaces, parts))
        return self._from_components(out)

    @property
    def is_euclidean(self) -> bool:
        """Return whether every product component is Euclidean."""
        return all(s.is_euclidean for s in self.spaces)


class _ProductStarMixin:
    """Star operation for products whose components all support it."""

    def star(self, x: ProductElement) -> ProductElement:
        """Return the componentwise star operation."""
        parts = self._components(x)
        out = tuple(s.star(xi) for s, xi in zip(self.spaces, parts))
        return self._from_components(out)


class _ProductJordanMixin:
    """Jordan operations for products whose components all support them."""

    @checked_method(in_space="self", arg_positions=(0, 1))
    def jordan(self, x: ProductElement, y: ProductElement) -> ProductElement:
        """Return the componentwise Jordan product."""
        x_parts = self._components(x)
        y_parts = self._components(y)
        out = tuple(s.jordan(xi, yi) for s, xi, yi in zip(self.spaces, x_parts, y_parts))
        return self._from_components(out)

    def _spectral_size(self, space: Space) -> int:
        """Return the trailing spectral rank of one component space."""
        spectrum = space.spectrum(space.zeros())
        return int(getattr(spectrum, "shape", (0,))[-1])

    def spectrum(self, x: ProductElement) -> DenseArray:
        """Return the concatenated Jordan spectrum of product components."""
        x_parts = self._components(x)
        parts = tuple(s.spectrum(xi) for s, xi in zip(self.spaces, x_parts))
        if len(parts) == 1:
            return parts[0]
        return self.ops.concatenate(parts, axis=-1)

    def spectral_decompose(self, x: ProductElement) -> ProductSpectralDecomposition:
        """Return componentwise spectral data independent of element structure."""
        parts = self._components(x)
        decompositions = tuple(s.spectral_decompose(xi) for s, xi in zip(self.spaces, parts))
        return ProductSpectralDecomposition(
            eigvals=tuple(eigvals for eigvals, _frame in decompositions),
            frames=tuple(frame for _eigvals, frame in decompositions),
        )

    def from_spectrum(
        self,
        decomposition: ProductSpectralDecomposition,
        frame: Any = None,
    ) -> ProductElement:
        """Reconstruct a product element from explicit product spectral data."""
        if frame is not None:
            raise TypeError("ProductSpace.from_spectrum expects ProductSpectralDecomposition only.")
        if not isinstance(decomposition, ProductSpectralDecomposition):
            raise TypeError(
                "ProductSpace.from_spectrum expects ProductSpectralDecomposition; "
                f"got {type(decomposition).__name__}."
            )
        if len(decomposition.eigvals) != self.arity or len(decomposition.frames) != self.arity:
            raise ValueError("ProductSpace.from_spectrum decomposition arity mismatch.")
        out = tuple(
            s.from_spectrum(component_eigvals, component_frame)
            for s, component_eigvals, component_frame in zip(
                self.spaces,
                decomposition.eigvals,
                decomposition.frames,
            )
        )
        return self._from_components(out)

    @checked_method(in_space="self", out_space="self")
    def spectral_apply(self, x: ProductElement, f: Callable[[Any], Any]) -> ProductElement:
        """Apply each component space's spectral calculus independently."""
        parts = self._components(x)
        if self.arity == 2:
            return self._from_components(
                (
                    self.spaces[0].spectral_apply(parts[0], f),
                    self.spaces[1].spectral_apply(parts[1], f),
                )
            )
        out = tuple(s.spectral_apply(xi, f) for s, xi in zip(self.spaces, parts))
        return self._from_components(out)


@jax_pytree_class
class _ProductInnerProductSpace(_ProductInnerProductMixin, ProductSpace, InnerProductSpace):
    """Product space whose components all support inner products."""

    def __init__(self, spaces, ctx=None, structure=None):
        spaces = _validate_product_spaces(spaces, type(self).__name__)
        _require_all_components(spaces, InnerProductSpace, type(self).__name__)
        super().__init__(spaces, ctx=ctx, structure=structure)


@jax_pytree_class
class _ProductStarSpace(_ProductStarMixin, ProductSpace, StarSpace):
    """Product space whose components all support a star operation."""

    def __init__(self, spaces, ctx=None, structure=None):
        spaces = _validate_product_spaces(spaces, type(self).__name__)
        _require_all_components(spaces, StarSpace, type(self).__name__)
        super().__init__(spaces, ctx=ctx, structure=structure)


@jax_pytree_class
class _ProductJordanAlgebraSpace(_ProductJordanMixin, ProductSpace, JordanAlgebraSpace):
    """Product space whose components all support Jordan algebra operations."""

    def __init__(self, spaces, ctx=None, structure=None):
        spaces = _validate_product_spaces(spaces, type(self).__name__)
        _require_all_components(spaces, JordanAlgebraSpace, type(self).__name__)
        super().__init__(spaces, ctx=ctx, structure=structure)


@jax_pytree_class
class _ProductEuclideanJordanAlgebraSpace(
    _ProductInnerProductMixin,
    _ProductJordanMixin,
    ProductSpace,
    EuclideanJordanAlgebraSpace,
):
    """Product space whose components all support Euclidean Jordan algebra operations."""

    def __init__(self, spaces, ctx=None, structure=None):
        spaces = _validate_product_spaces(spaces, type(self).__name__)
        _require_all_components(spaces, EuclideanJordanAlgebraSpace, type(self).__name__)
        super().__init__(spaces, ctx=ctx, structure=structure)
        _require_all_components(self.spaces, EuclideanJordanAlgebraSpace, type(self).__name__)


@jax_pytree_class
class _ProductInnerProductStarSpace(
    _ProductInnerProductMixin,
    _ProductStarMixin,
    ProductSpace,
    InnerProductSpace,
    StarSpace,
):
    """Product implementation for inner-product plus star capability."""


@jax_pytree_class
class _ProductInnerProductJordanSpace(
    _ProductInnerProductMixin,
    _ProductJordanMixin,
    ProductSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
):
    """Product implementation for inner-product plus Jordan capability."""


@jax_pytree_class
class _ProductStarJordanSpace(
    _ProductStarMixin,
    _ProductJordanMixin,
    ProductSpace,
    StarSpace,
    JordanAlgebraSpace,
):
    """Product implementation for star plus Jordan capability."""


@jax_pytree_class
class _ProductInnerProductStarJordanSpace(
    _ProductInnerProductMixin,
    _ProductStarMixin,
    _ProductJordanMixin,
    ProductSpace,
    InnerProductSpace,
    StarSpace,
    JordanAlgebraSpace,
):
    """Product implementation for inner-product, star, and Jordan capability."""


@jax_pytree_class
class _ProductEuclideanJordanStarSpace(
    _ProductStarMixin,
    _ProductEuclideanJordanAlgebraSpace,
    StarSpace,
):
    """Product implementation for Euclidean-Jordan plus star capability."""


_PRODUCT_REGISTRY.update(
    {
        frozenset(): ProductSpace,
        frozenset({_CAP_INNER}): _ProductInnerProductSpace,
        frozenset({_CAP_STAR}): _ProductStarSpace,
        frozenset({_CAP_JORDAN}): _ProductJordanAlgebraSpace,
        frozenset({_CAP_INNER, _CAP_STAR}): _ProductInnerProductStarSpace,
        frozenset({_CAP_INNER, _CAP_JORDAN}): _ProductInnerProductJordanSpace,
        frozenset({_CAP_STAR, _CAP_JORDAN}): _ProductStarJordanSpace,
        frozenset({_CAP_INNER, _CAP_STAR, _CAP_JORDAN}): _ProductInnerProductStarJordanSpace,
        frozenset(
            {_CAP_INNER, _CAP_JORDAN, _CAP_EUCLIDEAN_JORDAN}
        ): _ProductEuclideanJordanAlgebraSpace,
        frozenset(
            {_CAP_INNER, _CAP_STAR, _CAP_JORDAN, _CAP_EUCLIDEAN_JORDAN}
        ): _ProductEuclideanJordanStarSpace,
    }
)


__all__ = [
    "ProductSpace",
    "ProductSpectralDecomposition",
    "_space_capabilities",
    "_product_capabilities",
]
