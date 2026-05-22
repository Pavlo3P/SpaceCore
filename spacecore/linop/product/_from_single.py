from __future__ import annotations

from typing import Any, Tuple

from ._base import ProductLinOp
from .._base import LinOp, Domain
from ...space import ProductSpace, VectorSpace
from ...backend import jax_pytree_class, Context


@jax_pytree_class
class StackedLinOp(ProductLinOp[Domain, ProductSpace]):
    """
    Stack of operators from a single domain into a product codomain.

    dom = X
    cod = Y1 × ... × Yk

    ``ops[i] : X -> Yi``
    ``apply(x)  = (ops[i](x))_i``
    ``rapply(y) = sum_i ops[i]^*(y_i)``
    """

    def _check_layout(self) -> None:
        if not isinstance(self.cod, ProductSpace):
            raise TypeError("StackedLinOp expects cod to be ProductSpace.")

        if len(self.parts) != len(self.cod.spaces):
            raise ValueError("Number of ops must match codomain product arity.")

        for i, A in enumerate(self.parts):
            if A.dom == self.dom and A.cod == self.cod.spaces[i]:
                continue
            else:
                raise TypeError(f"Component op {i} must map dom -> cod.spaces[{i}].")

    def apply(self, x: Any) -> Any:
        if self._enable_checks:
            self.dom._check_member(x)
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        if self._num_parts == 2:
            return self._apply_parts[0](x), self._apply_parts[1](x)
        return tuple(apply(x) for apply in self._apply_parts)

    def rapply(self, y: Any) -> Any:
        if self._enable_checks:
            self.cod._check_member(y)
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: Any) -> Any:
        if self._num_parts == 2:
            x0 = self._rapply_parts[0](y[0])
            x1 = self._rapply_parts[1](y[1])
            return x0 + x1 if type(self.dom) is VectorSpace else self.dom.add(x0, x1)
        acc = None
        use_direct_add = type(self.dom) is VectorSpace
        for rapply, yi in zip(self._rapply_parts, y):
            xi = rapply(yi)
            acc = xi if acc is None else (acc + xi if use_direct_add else self.dom.add(xi, acc))
        return acc

    def vapply(self, x: Any, batch_space=None) -> Any:
        in_space = self._input_batch_space(self.domain, x, batch_space)
        if self._enable_checks:
            in_space._check_member(x)
        batch_shape = in_space.batch_shape
        batch_axes = in_space.batch_axes
        return tuple(
            op.vapply(x, op.domain.batch(batch_shape, batch_axes))
            for op in self.parts
        )

    def rvapply(self, y: Any, batch_space=None) -> Any:
        in_space = self._input_batch_space(self.codomain, y, batch_space)
        if self._enable_checks:
            in_space._check_member(y)
        batch_shape = in_space.batch_shape
        batch_axes = in_space.batch_axes
        out_space = self.domain.batch(batch_shape, batch_axes)
        acc = None
        for op, yi in zip(self.parts, y):
            xi = op.rvapply(yi, op.codomain.batch(batch_shape, batch_axes))
            acc = xi if acc is None else out_space.add(acc, xi)
        return acc

    @classmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> StackedLinOp:
        if not parts:
            raise ValueError("Parts must be non-empty.")

        cod = ProductSpace(tuple(op.cod for op in parts))
        dom = parts[0].dom

        return cls(dom, cod, parts)

    def _convert(self, new_ctx: Context) -> StackedLinOp:
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_parts = [op.convert(new_ctx) for op in self.parts]
        return StackedLinOp(new_dom, new_cod, new_parts)
