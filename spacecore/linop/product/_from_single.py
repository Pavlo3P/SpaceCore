from __future__ import annotations

from typing import Any, Tuple

from ._base import ProductLinOp
from .._base import LinOp, Domain
from ...space import ProductSpace
from ...backend import jax_pytree_class, Context


@jax_pytree_class
class StackedLinOp(ProductLinOp[Domain, ProductSpace]):
    """
    Stack of operators from a single domain into a product codomain.

    dom = X
    cod = Y1 × ... × Yk

    ops[i] : X -> Yi
    apply(x)  = (ops[i](x))_i
    rapply(y) = sum_i ops[i]^*(y_i)
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
        self.assert_domain(x)
        return tuple(A.apply(x) for A in self.parts)

    def rapply(self, y: Any) -> Any:
        self.assert_codomain(y)
        acc = None
        for A, yi in zip(self.parts, y):
            xi = A.rapply(yi)
            acc = xi if acc is None else self.dom.add(xi, acc)
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