from __future__ import annotations

from typing import Any, Tuple, List, Sequence, Callable

from ..base import (
    CoordinateSpace,
    EuclideanJordanAlgebraSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
    Space,
    StarSpace,
)
from ..checks import ProductComponentCheck, ProductStructureCheck
from .._structure import ProductStructure, TupleStructure, PytreeStructure
from ._dense_coordinate import DenseCoordinateSpace
from ._dense_vector import DenseVectorSpace, ElementwiseJordanSpace
from ..._checks import checked_method
from ...types import DenseArray
from ...backend import Context, jax_pytree_class

from ..._contextual import resolve_context_priority


ProductElement = Any


def _prod_int(shape: Tuple[int, ...]) -> int:
    """Return the integer product of a shape tuple."""
    p = 1
    for d in shape:
        p *= int(d)
    return int(p)


@jax_pytree_class
class ProductSpace(CoordinateSpace):
    r"""
    Represent a Cartesian product of spaces.

    Product elements are tuples ``(x1, ..., xk)`` by default. Advanced callers
    can opt into a registered pytree/dataclass representation with
    ``ProductSpace.from_template(...)`` or an explicit ``PytreeStructure``.
    Operations still run componentwise, and element-returning operations
    rebuild the same representation as this product space's structure.

    Dense coordinates are representation-neutral: :meth:`flatten` concatenates
    the flattened coordinates of each component, so equal ordered components
    produce the same flat vector whether the product element is represented as
    a tuple or as a registered pytree/dataclass.

    Parameters
    ----------
    spaces : tuple of Space
        Nonempty tuple of component spaces.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from components.
    structure : ProductStructure or None, optional
        Element representation adapter. Default uses tuple elements.

    Attributes
    ----------
    spaces : tuple of Space
        Component spaces converted to ``ctx``.
    arity : int
        Number of component spaces.

    Notes
    -----
    ``shape`` is the one-dimensional coordinate length of the concatenated
    flattening. The Jordan spectrum of a product/direct sum is the last-axis
    concatenation of component spectra. Product spectral decompositions are
    component-delegating rather than one fused reconstruction frame.

    The product inner product is the sum of component inner products. Riesz
    and inverse Riesz maps are applied componentwise and return the product
    element representation configured by ``structure``. ``is_euclidean`` is
    true if and only if every component space is Euclidean. Although
    :class:`Space` stores a ``geometry`` attribute, ``ProductSpace`` uses these
    componentwise overrides as its effective geometry.
    """


    def __new__(
        cls,
        spaces: Tuple[Space, ...],
        ctx: Context | str | None = None,
        structure: ProductStructure | None = None,
    ):
        if cls is ProductSpace:
            spaces_tuple = tuple(spaces) if isinstance(spaces, Sequence) else ()
            if spaces_tuple and all(isinstance(sp, EuclideanJordanAlgebraSpace) and isinstance(sp, StarSpace) for sp in spaces_tuple):
                cls = ProductEuclideanJordanAlgebraSpace
            elif spaces_tuple and all(isinstance(sp, JordanAlgebraSpace) for sp in spaces_tuple):
                cls = ProductJordanAlgebraSpace
            elif spaces_tuple and all(isinstance(sp, StarSpace) for sp in spaces_tuple):
                cls = ProductStarSpace
            elif spaces_tuple and all(isinstance(sp, InnerProductSpace) for sp in spaces_tuple):
                cls = ProductInnerProductSpace
        return super(ProductSpace, cls).__new__(cls)

    def _convert(self, new_ctx: Context) -> Space:
        """Convert all component spaces to ``new_ctx``."""
        new_spaces = []
        for sp in self.spaces:
            new_spaces.append(sp.convert(new_ctx))
        return type(self)(tuple(new_spaces), new_ctx, structure=self._structure)

    def _local_checks(self):
        """Return membership checks local to product spaces."""
        return ProductStructureCheck(), ProductComponentCheck()

    def __init__(
        self,
        spaces: Tuple[Space, ...],
        ctx: Context | str | None = None,
        structure: ProductStructure | None = None,
    ) -> None:
        if len(spaces) == 0:
            raise ValueError("ProductSpace requires at least one subspace.")
        if structure is None:
            structure = TupleStructure()
        if not isinstance(structure, ProductStructure):
            raise TypeError(
                "ProductSpace structure must be a ProductStructure, "
                f"got {type(structure).__name__}."
            )

        spaces = self._validate_spaces(spaces)
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
        self._vector_fast_path = all(type(sp) in (DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace) for sp in uniform_spaces)
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
        if type(other) is type(self):
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
        """Build a product whose elements match a registered pytree template.

        For dataclasses, register the class with JAX first, for example via
        ``jax.tree_util.register_dataclass(MyState)``.
        """
        structure = PytreeStructure(template_element)
        product = cls(spaces, ctx=ctx, structure=structure)
        structure.to_components(template_element, arity=product.arity)
        return product

    def _validate_spaces(self, spaces: Any) -> Tuple[Space, ...]:
        """Validate and normalize product component spaces."""
        if isinstance(spaces, Sequence):
            spaces = tuple(spaces)
            for i, sp in enumerate(spaces):
                if isinstance(sp, Space):
                    continue
                else:
                    raise TypeError(f"ProductSpace requires a sequence of spaces, got {type(sp)!r} at index {i}.")
            return spaces
        else:
            raise TypeError(f"ProductSpace requires a sequence of spaces, got {type(spaces)!r}.")

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

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: ProductElement, y: ProductElement) -> Any:
        r"""Return the sum of component inner products."""
        x_parts = self._components(x)
        y_parts = self._components(y)
        # Accumulate via backend ops (vdot works for scalars too, but sum is enough)
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


    def star(self, x: ProductElement) -> ProductElement:
        """Return the componentwise star operation."""
        parts = self._components(x)
        out = tuple(s.star(xi) for s, xi in zip(self.spaces, parts))
        return self._from_components(out)

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

    def spectral_decompose(self, x: ProductElement) -> ProductElement:
        """Return componentwise spectral decompositions aligned to components."""
        parts = self._components(x)
        out = tuple(s.spectral_decompose(xi) for s, xi in zip(self.spaces, parts))
        return self._from_components(out)

    def from_spectrum(self, eigvals: Any, frame: Any = None) -> ProductElement:
        """Reconstruct components from component-delegating spectral data."""
        if frame is None:
            decompositions = self._components(eigvals)
            if len(decompositions) != self.arity:
                raise ValueError("ProductSpace.from_spectrum decomposition arity mismatch.")
            out = tuple(
                s.from_spectrum(component_eigvals, component_frame)
                for s, (component_eigvals, component_frame) in zip(self.spaces, decompositions)
            )
            return self._from_components(out)

        frame_parts = self._components(frame)

        if isinstance(eigvals, tuple) or not self.ctx.ops.is_dense(eigvals):
            eigval_parts = self._components(eigvals)
            if len(eigval_parts) != self.arity:
                raise ValueError("ProductSpace.from_spectrum eigval arity mismatch.")
            out = tuple(
                s.from_spectrum(component_eigvals, component_frame)
                for s, component_eigvals, component_frame in zip(self.spaces, eigval_parts, frame_parts)
            )
            return self._from_components(out)

        if self._enable_checks:
            eigvals = self.ctx.assert_dense(eigvals)
        components = []
        offset = 0
        for s, component_frame in zip(self.spaces, frame_parts):
            size = self._spectral_size(s)
            vals = eigvals[..., offset: offset + size]
            components.append(s.from_spectrum(vals, component_frame))
            offset += size
        if offset != int(getattr(eigvals, "shape", (offset,))[-1]):
            raise ValueError("ProductSpace.from_spectrum received extra eigenvalues.")
        return self._from_components(tuple(components))

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
            if self._enable_checks:
                vi = self.ctx.assert_dense(vi)
            parts.append(vi)

        if len(parts) == 1:
            return parts[0]

        if self._concatenate_uses_dim:
            return self._concatenate(parts, dim=0)
        return self._concatenate(parts, axis=0)

    def unflatten(self, v: DenseArray) -> ProductElement:
        """Split dense coordinates into component-space elements."""
        if self._enable_checks:
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
        if self._enable_checks:
            vs = self.ctx.assert_dense(vs)
        parts = tuple(
            s.unflatten_batch(vs[:, slc])
            for s, slc in zip(self.spaces, self._slices)
        )
        return self._from_components(parts)

    @checked_method(in_space="self", out_space="self")
    def spectral_apply(self, x: ProductElement, f: Callable[[Any], Any]) -> ProductElement:
        r"""
        Apply a function to each component of a product-space element.

        For a product space
        $$
        X = X_1 \times \cdots \times X_m,
        $$
        and an element
        $$
        x = (x_1,\dots,x_m), \qquad x_i \in X_i,
        $$
        this method returns
        $$
        f(x) := \bigl(f_{X_1}(x_1), \dots, f_{X_m}(x_m)\bigr),
        $$
        where ``f_{X_i}`` denotes application according to the logic of the
        corresponding component space ``X_i``.

        Parameters
        ----------
        x:
            Product element in this space's configured representation. Tuple is
            the default representation; registered pytree/dataclass elements
            are accepted only when the space was built from a template or
            explicit ``PytreeStructure``.
        f:
            Callable to apply to each component. The meaning of application is
            delegated to each component space via ``spaces[i].apply``.

        Returns
        -------
        Any
            Product element with transformed components, rebuilt using this
            product space's structure.

        Raises
        ------
        TypeError
            If ``x`` is not a valid product-space element.
        ValueError
            If ``x`` has the wrong arity for this product space.

        Notes
        -----
        This method does not define a new joint functional calculus on the
        product space. It applies the existing functional calculus of each
        factor space independently, component by component.
        """
        parts = self._components(x)
        if self._arity == 2:
            return self._from_components((
                self.spaces[0].apply(parts[0], f),
                self.spaces[1].apply(parts[1], f),
            ))
        out = tuple(s.apply(xi, f) for s, xi in zip(self.spaces, parts))
        return self._from_components(out)

    def tree_flatten(self):
        """Flatten this space for JAX pytree registration."""
        return (), (self.spaces, self.ctx, self._structure)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this space from pytree aux data."""
        spaces, ctx, structure = aux
        return cls(spaces, ctx, structure=structure)

    @checked_method(in_space="self", out_space="self")
    def apply(self, x: ProductElement, f: Callable[[Any], Any]) -> ProductElement:
        """Backward-compatible alias for spectral application."""
        return self.spectral_apply(x, f)


def _require_all_components(space: ProductSpace, capability: type, name: str) -> None:
    bad = [type(component).__name__ for component in space.spaces if not isinstance(component, capability)]
    if bad:
        raise TypeError(f"{name} requires every component to be {capability.__name__}; got incompatible components {bad}.")


@jax_pytree_class
class ProductInnerProductSpace(ProductSpace, InnerProductSpace):
    """Product space whose components all support inner products."""

    def __init__(self, spaces, ctx=None, structure=None):
        super().__init__(spaces, ctx=ctx, structure=structure)
        _require_all_components(self, InnerProductSpace, type(self).__name__)


@jax_pytree_class
class ProductStarSpace(ProductSpace, StarSpace):
    """Product space whose components all support a star operation."""

    def __init__(self, spaces, ctx=None, structure=None):
        super().__init__(spaces, ctx=ctx, structure=structure)
        _require_all_components(self, StarSpace, type(self).__name__)


@jax_pytree_class
class ProductJordanAlgebraSpace(ProductSpace, JordanAlgebraSpace):
    """Product space whose components all support Jordan algebra operations."""

    def __init__(self, spaces, ctx=None, structure=None):
        super().__init__(spaces, ctx=ctx, structure=structure)
        _require_all_components(self, JordanAlgebraSpace, type(self).__name__)


@jax_pytree_class
class ProductEuclideanJordanAlgebraSpace(ProductSpace, StarSpace, EuclideanJordanAlgebraSpace):
    """Product space whose components all support Euclidean Jordan algebra operations."""

    def __init__(self, spaces, ctx=None, structure=None):
        super().__init__(spaces, ctx=ctx, structure=structure)
        _require_all_components(self, EuclideanJordanAlgebraSpace, type(self).__name__)
        _require_all_components(self, StarSpace, type(self).__name__)
