from __future__ import annotations

from typing import Any, Tuple, List, Sequence, Callable

from ._base import Space
from ._checks import ProductComponentCheck, ProductStructureCheck
from ._vector import VectorSpace
from ..types import DenseArray
from ..backend import Context

from .._contextual.manager import ctx_manager


def _prod_int(shape: Tuple[int, ...]) -> int:
    p = 1
    for d in shape:
        p *= int(d)
    return int(p)


class ProductSpace(Space):
    """
    Cartesian product space X = X1 × ... × Xk.

    Elements are tuples:
        x = (x1, ..., xk) with xi ∈ Xi

    Canonical dense coordinates:
        flatten(x) = concat(flatten_i(xi))

    Notes:
      - `shape` for this space is the *1D coordinate length* of the concatenated flattening.
      - `eigh` has no canonical meaning here and raises by default.
    """

    def _convert(self, new_ctx: Context) -> Space:
        new_spaces = []
        for sp in self.spaces:
            new_spaces.append(sp.convert(new_ctx))
        return ProductSpace(tuple(new_spaces), new_ctx)

    def _local_checks(self):
        return ProductStructureCheck(), ProductComponentCheck()

    def __init__(self, spaces: Tuple[Space, ...], ctx: Context | str | None = None) -> None:
        if len(spaces) == 0:
            raise ValueError("ProductSpace requires at least one subspace.")

        spaces = self._validate_spaces(spaces)
        ctx = ctx_manager.resolve_context_priority(ctx, *spaces)

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
        if raw_array_ops is None:
            raw_array_ops = getattr(self.ctx.ops, "jnp", None)
        self._concatenate = (
            raw_array_ops.concatenate if raw_array_ops is not None else self.ctx.ops.concatenate
        )
        if self._arity >= 1:
            self._slice0 = self._slices[0]
            self._shape0 = self._component_shapes[0]
            self._is_flat0 = self._component_is_flat[0]
        if self._arity >= 2:
            self._slice1 = self._slices[1]
            self._shape1 = self._component_shapes[1]
            self._is_flat1 = self._component_is_flat[1]

    def _validate_spaces(self, spaces: Any) -> Tuple[Space, ...]:
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
        return self._arity

    def zeros(self) -> Tuple[Any, ...]:
        return tuple(s.zeros() for s in self.spaces)

    def add(self, x: Tuple[Any, ...], y: Tuple[Any, ...]) -> Tuple[Any, ...]:
        self.check_member(x)
        self.check_member(y)
        return tuple(s.add(xi, yi) for s, xi, yi in zip(self.spaces, x, y))

    def scale(self, a: Any, x: Tuple[Any, ...]) -> Tuple[Any, ...]:
        self.check_member(x)
        return tuple(s.scale(a, xi) for s, xi in zip(self.spaces, x))

    def inner(self, x: Tuple[Any, ...], y: Tuple[Any, ...]) -> Any:
        self.check_member(x)
        self.check_member(y)

        # Accumulate via backend ops (vdot works for scalars too, but sum is enough)
        acc = None
        for s, xi, yi in zip(self.spaces, x, y):
            v = s.inner(xi, yi)
            acc = v if acc is None else (acc + v)
        return acc

    def eigh(self, x: Any, k: int = None) -> Any:
        raise NotImplementedError(
            "ProductSpace.eigh is not defined. "
            "Call eigh on a specific component space, or define a custom convention."
        )

    def flatten(self, x: Tuple[Any, ...]) -> DenseArray:
        self.check_member(x)

        if self._vector_fast_path:
            if self._arity == 1:
                return x[0] if self._component_is_flat[0] else x[0].reshape((-1,))
            if self._arity == 2:
                x0 = x[0] if self._is_flat0 else x[0].reshape((-1,))
                x1 = x[1] if self._is_flat1 else x[1].reshape((-1,))
                return self._concatenate((x0, x1), axis=0)
            parts = tuple(
                xi if is_flat else xi.reshape((-1,))
                for xi, is_flat in zip(x, self._component_is_flat)
            )
            return self._concatenate(parts, axis=0)

        parts = []
        for s, xi in zip(self.spaces, x):
            vi = s.flatten(xi)
            if self._enable_checks:
                vi = self.ctx.assert_dense(vi)
            parts.append(vi)

        if len(parts) == 1:
            return parts[0]

        return self._concatenate(parts, axis=0)

    def unflatten(self, v: DenseArray) -> Tuple[Any, ...]:
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
        self.check_member(x)
        if self._arity == 2:
            return self.spaces[0].apply(x[0], f), self.spaces[1].apply(x[1], f)
        return tuple(s.apply(xi, f) for s, xi in zip(self.spaces, x))
