from __future__ import annotations

from typing import Any, Tuple

from ._base import ProductLinOp
from .._base import LinOp, Codomain
from ...space import ProductSpace, VectorSpace
from ...backend import jax_pytree_class, Context


@jax_pytree_class
class SumToSingleLinOp(ProductLinOp[ProductSpace, Codomain]):
    """
    Sum of component operators from a product domain into a single codomain.

    dom = X1 × ... × Xk
    cod = Y

    ``ops[i] : Xi -> Y``
    ``apply(x)  = sum_i ops[i](x_i)``
    ``rapply(y) = (ops[i]^*(y))_i``
    """

    def _check_layout(self) -> None:
        if not isinstance(self.dom, ProductSpace):
            raise TypeError("SumToSingleLinOp expects dom to be ProductSpace.")

        if len(self.parts) != len(self.dom.spaces):
            raise ValueError("Number of ops must match product arity.")

        for i, A in enumerate(self.parts):
            if A.dom == self.dom.spaces[i] and A.cod == self.cod:
                continue
            else:
                raise TypeError(f"Component op {i} must map dom.spaces[{i}] -> cod.")

    def apply(self, x: Any) -> Any:
        if self._enable_checks:
            self.dom._check_member(x)
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        if self._num_parts == 2:
            y0 = self._apply_parts[0](x[0])
            y1 = self._apply_parts[1](x[1])
            return y0 + y1 if type(self.cod) is VectorSpace else self.cod.add(y0, y1)
        acc = None
        use_direct_add = type(self.cod) is VectorSpace
        for apply, xi in zip(self._apply_parts, x):
            yi = apply(xi)
            acc = yi if acc is None else (acc + yi if use_direct_add else self.cod.add(yi, acc))
        return acc

    def rapply(self, y: Any) -> Any:
        if self._enable_checks:
            self.cod._check_member(y)
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: Any) -> Any:
        if self._num_parts == 2:
            return self._rapply_parts[0](y), self._rapply_parts[1](y)
        return tuple(rapply(y) for rapply in self._rapply_parts)

    @classmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> SumToSingleLinOp:
        if not parts:
            raise ValueError("Parts must be non-empty.")

        dom = ProductSpace(tuple(op.dom for op in parts))
        cod = parts[0].cod

        return cls(dom, cod, parts)

    def _convert(self, new_ctx: Context) -> SumToSingleLinOp:
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_parts = [op.convert(new_ctx) for op in self.parts]
        return SumToSingleLinOp(new_dom, new_cod, new_parts)
