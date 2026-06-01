from __future__ import annotations

from typing import Any, Tuple, List, Sequence, Callable

from ._base import Space
from ._checks import ProductComponentCheck, ProductStructureCheck
from ._vector import VectorSpace
from .._checks import checked_method
from ..types import DenseArray
from ..backend import Context

from .._contextual import resolve_context_priority


def _prod_int(shape: Tuple[int, ...]) -> int:
    """Return the integer product of a shape tuple."""
    p = 1
    for d in shape:
        p *= int(d)
    return int(p)


class ProductSpace(Space):
    r"""
    Represent a Cartesian product of spaces.

    Elements are tuples ``(x1, ..., xk)`` with ``xi`` in ``spaces[i]``.
    Dense coordinates concatenate the flattened coordinates of each component.

    Parameters
    ----------
    spaces : tuple of Space
        Nonempty tuple of component spaces.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from components.

    Attributes
    ----------
    spaces : tuple of Space
        Component spaces converted to ``ctx``.
    arity : int
        Number of component spaces.

    Notes
    -----
    ``shape`` is the one-dimensional coordinate length of the concatenated
    flattening. ``eigh`` has no canonical meaning and raises by default.

    The product inner product is the sum of component inner products. Riesz
    and inverse Riesz maps are applied componentwise to product tuple elements,
    and ``is_euclidean`` is true if and only if every component space is
    Euclidean. Although :class:`Space` stores a ``geometry`` attribute,
    ``ProductSpace`` uses these componentwise overrides as its effective
    geometry.
    """

    def _convert(self, new_ctx: Context) -> Space:
        """Convert all component spaces to ``new_ctx``."""
        new_spaces = []
        for sp in self.spaces:
            new_spaces.append(sp.convert(new_ctx))
        return ProductSpace(tuple(new_spaces), new_ctx)

    def _local_checks(self):
        """Return membership checks local to product spaces."""
        return ProductStructureCheck(), ProductComponentCheck()

    def __init__(self, spaces: Tuple[Space, ...], ctx: Context | str | None = None) -> None:
        if len(spaces) == 0:
            raise ValueError("ProductSpace requires at least one subspace.")

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
        self._arity = len(uniform_spaces)
        self._vector_fast_path = all(type(sp) is VectorSpace for sp in uniform_spaces)
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
            return self.ctx == other.ctx and self.spaces == other.spaces
        return False

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
        return self._arity

    def zeros(self) -> Tuple[Any, ...]:
        """Return the product-space zero tuple."""
        return tuple(s.zeros() for s in self.spaces)

    @checked_method(in_space="self", arg_positions=(0, 1))
    def add(self, x: Tuple[Any, ...], y: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Return the componentwise product-space sum."""
        return tuple(s.add(xi, yi) for s, xi, yi in zip(self.spaces, x, y))

    def add_batch(self, x: Tuple[Any, ...], y: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Return the componentwise leading-axis batch sum."""
        return tuple(s.add_batch(xi, yi) for s, xi, yi in zip(self.spaces, x, y))

    @checked_method(in_space="self", arg_positions=(1,))
    def scale(self, a: Any, x: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Return the componentwise scalar product."""
        return tuple(s.scale(a, xi) for s, xi in zip(self.spaces, x))

    def scale_batch(self, a: Any, x: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Return the componentwise leading-axis batch scalar product."""
        return tuple(s.scale_batch(a, xi) for s, xi in zip(self.spaces, x))

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: Tuple[Any, ...], y: Tuple[Any, ...]) -> Any:
        r"""Return the sum of component inner products."""
        # Accumulate via backend ops (vdot works for scalars too, but sum is enough)
        acc = None
        for s, xi, yi in zip(self.spaces, x, y):
            v = s.inner(xi, yi)
            acc = v if acc is None else (acc + v)
        return acc

    def riesz(self, x: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Apply each component space's Riesz map."""
        return tuple(s.riesz(xi) for s, xi in zip(self.spaces, x))

    def riesz_inverse(self, x: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Apply each component space's inverse Riesz map."""
        return tuple(s.riesz_inverse(xi) for s, xi in zip(self.spaces, x))

    @property
    def is_euclidean(self) -> bool:
        """Return whether every product component is Euclidean."""
        return all(s.is_euclidean for s in self.spaces)

    def eigh(self, x: Any, k: int = None) -> Any:
        """Raise because product spaces do not define a canonical eigendecomposition."""
        raise NotImplementedError(
            "ProductSpace.eigh is not defined. "
            "Call eigh on a specific component space, or define a custom convention."
        )

    @checked_method(in_space="self")
    def flatten(self, x: Tuple[Any, ...]) -> DenseArray:
        """Concatenate component coordinate vectors into one dense vector."""
        if self._vector_fast_path:
            if self._arity == 1:
                return x[0] if self._component_is_flat[0] else x[0].reshape((-1,))
            if self._arity == 2:
                x0 = x[0] if self._is_flat0 else x[0].reshape((-1,))
                x1 = x[1] if self._is_flat1 else x[1].reshape((-1,))
                if self._concatenate_uses_dim:
                    return self._concatenate((x0, x1), dim=0)
                return self._concatenate((x0, x1), axis=0)
            parts = tuple(
                xi if is_flat else xi.reshape((-1,))
                for xi, is_flat in zip(x, self._component_is_flat)
            )
            if self._concatenate_uses_dim:
                return self._concatenate(parts, dim=0)
            return self._concatenate(parts, axis=0)

        parts = []
        for s, xi in zip(self.spaces, x):
            vi = s.flatten(xi)
            if self._enable_checks:
                vi = self.ctx.assert_dense(vi)
            parts.append(vi)

        if len(parts) == 1:
            return parts[0]

        if self._concatenate_uses_dim:
            return self._concatenate(parts, dim=0)
        return self._concatenate(parts, axis=0)

    def unflatten(self, v: DenseArray) -> Tuple[Any, ...]:
        """Split dense coordinates into component-space elements."""
        if self._enable_checks:
            v = self.ctx.assert_dense(v)
            v1 = v if tuple(getattr(v, "shape", ())) == self.shape else v.reshape((-1,))
        else:
            v1 = v

        if self._vector_fast_path:
            if self._arity == 1:
                x0 = v1[self._slice0]
                return (x0 if self._is_flat0 else x0.reshape(self._shape0),)
            if self._arity == 2:
                x0 = v1[self._slice0]
                x1 = v1[self._slice1]
                if not self._is_flat0:
                    x0 = x0.reshape(self._shape0)
                if not self._is_flat1:
                    x1 = x1.reshape(self._shape1)
                return x0, x1
            return tuple(
                v1[slc] if is_flat else v1[slc].reshape(shape)
                for slc, shape, is_flat in zip(
                    self._slices, self._component_shapes, self._component_is_flat
                )
            )

        xs: List[Any] = []
        for s, slc in zip(self.spaces, self._slices):
            vi = v1[slc]
            xs.append(s.unflatten(vi))

        return tuple(xs)

    def flatten_batch(self, xs: Tuple[Any, ...]) -> DenseArray:
        """Concatenate a leading-axis batch of product elements to ``(N, size)``."""
        parts = tuple(s.flatten_batch(xi) for s, xi in zip(self.spaces, xs))
        if len(parts) == 1:
            return parts[0]
        if self._concatenate_uses_dim:
            return self._concatenate(parts, dim=1)
        return self._concatenate(parts, axis=1)

    def unflatten_batch(self, vs: DenseArray) -> Tuple[Any, ...]:
        """Split rows of shape ``(N, size)`` into batched component elements."""
        if self._enable_checks:
            vs = self.ctx.assert_dense(vs)
        return tuple(
            s.unflatten_batch(vs[:, slc])
            for s, slc in zip(self.spaces, self._slices)
        )

    @checked_method(in_space="self", out_space="self")
    def apply(self, x: Tuple[Any, ...], f: Callable[[Any], Any]) -> Tuple[Any, ...]:
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
            Tuple representing an element of this product space. Its length must
            equal the arity of the product space, and each component must be a
            valid member of the corresponding factor space.
        f:
            Callable to apply to each component. The meaning of application is
            delegated to each component space via ``spaces[i].apply``.

        Returns
        -------
        tuple[Any, ...]
            Tuple of transformed components, one for each factor space.

        Raises
        ------
        TypeError
            If ``x`` is not a valid product-space element.
        ValueError
            If ``x`` has the wrong tuple length.

        Notes
        -----
        This method does not define a new joint functional calculus on the
        product space. It applies the existing functional calculus of each
        factor space independently, component by component.
        """
        if self._arity == 2:
            return self.spaces[0].apply(x[0], f), self.spaces[1].apply(x[1], f)
        return tuple(s.apply(xi, f) for s, xi in zip(self.spaces, x))
